from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import subprocess
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton.viewer


SCENE_MODULES = {
    "density_compare": "newton.examples.fluid.example_fluid_quasi2d_density_compare",
    "box_compare": "newton.examples.fluid.example_fluid_quasi2d_box_compare",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Panel-export config JSON.")
    parser.add_argument("--output-root", type=Path, default=None, help="Optional override for cache root directory.")
    parser.add_argument("--device", type=str, default=None, help="Warp device, for example cuda:0.")
    parser.add_argument("--record-frames", type=int, default=None, help="Optional override for recorded frame count.")
    parser.add_argument("--case-key", type=str, default=None, help="Export only one case key from the panel config.")
    parser.add_argument(
        "--isolate-cases",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Export each case in a fresh Python process for better Warp stability.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite the output root if it already exists.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def resolve_output_root(args: argparse.Namespace, config: dict[str, Any]) -> Path:
    if args.output_root is not None:
        return args.output_root.resolve()
    cache_dir = config.get("cache_dir")
    if cache_dir is None:
        raise ValueError("Config must define 'cache_dir' when --output-root is not provided.")
    return Path(cache_dir).resolve()


def prepare_output_root(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output root already exists: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def build_case_argv(
    *,
    panel_config: dict[str, Any],
    case_config: dict[str, Any],
    device: str | None,
) -> list[str]:
    argv: list[str] = []
    if device:
        argv.extend(["--device", device])

    argv.extend(["--fluid-solver", str(case_config["fluid_solver"])])
    if "iterations" in case_config:
        argv.extend(["--iterations", str(int(case_config["iterations"]))])
    if "sim_substeps" in case_config:
        argv.extend(["--sim-substeps", str(int(case_config["sim_substeps"]))])

    box_clamp = bool(panel_config.get("box_clamp", True))
    argv.append("--box-clamp" if box_clamp else "--no-box-clamp")

    if panel_config["scene"] == "box_compare":
        box_case = str(case_config.get("box_case", panel_config.get("box_case", "float")))
        argv.extend(["--box-case", box_case])
        if "box_density" in case_config:
            argv.extend(["--box-density", str(float(case_config["box_density"]))])
        if "coupling_mode" in panel_config:
            argv.extend(["--coupling-mode", str(panel_config["coupling_mode"])])
        if "coupling_iterations" in panel_config:
            argv.extend(["--coupling-iterations", str(int(panel_config["coupling_iterations"]))])

    return argv


def instantiate_example(
    *,
    panel_config: dict[str, Any],
    case_config: dict[str, Any],
    record_frames: int,
    device: str | None,
):
    module_name = SCENE_MODULES[str(panel_config["scene"])]
    module = importlib.import_module(module_name)
    parser = module.Example.create_parser()
    argv = build_case_argv(panel_config=panel_config, case_config=case_config, device=device)
    args = parser.parse_args(argv)
    if getattr(args, "device", None):
        wp.set_device(args.device)
    viewer = newton.viewer.ViewerNull(num_frames=max(record_frames, 1))
    example = module.Example(viewer, args)
    if hasattr(example, "enable_diagnostics"):
        example.enable_diagnostics = False
    return example


def fluid_namespace(example):
    if hasattr(example, "_fluid_state_namespace"):
        return example._fluid_state_namespace(example.state_0)
    raise AttributeError("Example does not expose a fluid-state namespace helper.")


def capture_payload(example, *, record_index: int) -> dict[str, np.ndarray]:
    fluid_state = fluid_namespace(example)
    payload: dict[str, np.ndarray] = {
        "record_index": np.asarray(record_index, dtype=np.int32),
        "frame_index": np.asarray(int(getattr(example, "frame_index", record_index)), dtype=np.int32),
        "sim_time": np.asarray(float(example.sim_time), dtype=np.float64),
        "fluid_positions": example.state_0.particle_q.numpy().astype(np.float32, copy=False),
        "fluid_density": fluid_state.density.numpy().astype(np.float32, copy=False),
        "fluid_colors": example.particle_colors.numpy().astype(np.float32, copy=False),
        "fluid_render_radii": example.particle_radii.numpy().astype(np.float32, copy=False),
    }

    if hasattr(example, "float_box_body"):
        body_index = int(example.float_box_body)
        payload["box_body_q"] = example.state_0.body_q.numpy()[body_index : body_index + 1].astype(np.float32, copy=False)
        payload["box_body_qd"] = example.state_0.body_qd.numpy()[body_index : body_index + 1].astype(
            np.float32, copy=False
        )
    return payload


def write_frame_payload(case_dir: Path, *, record_index: int, payload: dict[str, np.ndarray]) -> None:
    frame_dir = case_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(frame_dir / f"frame_{record_index:04d}.npz", **payload)


def build_case_metadata(
    *,
    example,
    panel_config: dict[str, Any],
    case_config: dict[str, Any],
    record_frames: int,
    record_initial_frame: bool,
    sim_ms: list[float],
) -> dict[str, Any]:
    scene_config = getattr(example, "scene", None) or getattr(example, "config", None)
    if scene_config is None:
        raise AttributeError("Example does not expose scene/config data.")

    radii = example.particle_radii.numpy().astype(np.float32, copy=False)
    metadata: dict[str, Any] = {
        "scene": str(panel_config["scene"]),
        "label": str(case_config["label"]),
        "key": str(case_config["key"]),
        "fluid_solver": str(case_config["fluid_solver"]),
        "iterations": int(case_config["iterations"]),
        "sim_substeps": int(case_config["sim_substeps"]),
        "record_frames": int(record_frames),
        "record_initial_frame": bool(record_initial_frame),
        "frame_dt": float(example.frame_dt),
        "sim_dt": float(example.sim_dt),
        "fps": int(example.fps),
        "rest_density": float(scene_config["rest_density"]),
        "particle_count": int(example.model.particle_count),
        "particle_display_radius": float(np.median(radii)) if radii.size else 0.0,
        "tank": {
            "half_width": float(example.container_half_width),
            "half_depth": float(example.container_half_depth),
            "floor_y": float(example.floor_y),
            "top_y": float(example.top_y),
        },
        "avg_sim_ms_per_frame": float(np.mean(np.asarray(sim_ms, dtype=np.float64))) if sim_ms else 0.0,
        "sim_frame_ms": [float(value) for value in sim_ms],
        "axis_conversion": "newton-y-up-to-blender-z-up",
    }

    if hasattr(example, "box_half_extent"):
        metadata["box"] = {
            "half_extent": [float(example.box_half_extent[0]), float(example.box_half_extent[1]), float(example.box_half_extent[2])],
            "density": float(scene_config["box_density"]),
            "box_case": str(getattr(example, "box_case", panel_config.get("box_case", "float"))),
        }

    return metadata


def export_case(
    *,
    panel_config: dict[str, Any],
    case_config: dict[str, Any],
    case_dir: Path,
    record_frames: int,
    record_initial_frame: bool,
    device: str | None,
) -> dict[str, Any]:
    print(f"[Fluid Compare] Export case: {case_config['key']}")
    example = instantiate_example(
        panel_config=panel_config,
        case_config=case_config,
        record_frames=record_frames,
        device=device,
    )

    sim_ms: list[float] = []
    record_index = 0

    if record_initial_frame:
        write_frame_payload(case_dir, record_index=record_index, payload=capture_payload(example, record_index=record_index))
        record_index += 1

    while record_index < record_frames:
        try:
            wp.synchronize_device(example.model.device)
            start = time.perf_counter()
            example.step()
            wp.synchronize_device(example.model.device)
            sim_ms.append((time.perf_counter() - start) * 1000.0)
            write_frame_payload(case_dir, record_index=record_index, payload=capture_payload(example, record_index=record_index))
            record_index += 1
        except Exception as exc:
            raise RuntimeError(
                f"Case '{case_config['key']}' failed while exporting record frame {record_index}."
            ) from exc

    metadata = build_case_metadata(
        example=example,
        panel_config=panel_config,
        case_config=case_config,
        record_frames=record_frames,
        record_initial_frame=record_initial_frame,
        sim_ms=sim_ms,
    )
    save_json(case_dir / "metadata.json", metadata)

    example.viewer.close()
    del example
    gc.collect()
    return metadata


def build_export_subprocess_command(
    *,
    args: argparse.Namespace,
    output_root: Path,
    case_key: str,
    record_frames: int,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--config",
        str(args.config.resolve()),
        "--output-root",
        str(output_root),
        "--record-frames",
        str(record_frames),
        "--case-key",
        case_key,
        "--no-isolate-cases",
    ]
    if args.device:
        command.extend(["--device", args.device])
    return command


def export_cases_in_subprocesses(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    output_root: Path,
    record_frames: int,
) -> dict[str, Any]:
    panel_metadata: dict[str, Any] = {
        "name": str(config["name"]),
        "scene": str(config["scene"]),
        "record_frames": record_frames,
        "record_initial_frame": bool(config.get("record_initial_frame", True)),
        "cases": [],
        "axis_conversion": "newton-y-up-to-blender-z-up",
        "config_path": str(args.config.resolve()),
    }

    for case_config in config["cases"]:
        case_key = str(case_config["key"])
        command = build_export_subprocess_command(
            args=args,
            output_root=output_root,
            case_key=case_key,
            record_frames=record_frames,
        )
        child_env = os.environ.copy()
        print(f"[Fluid Compare] Spawn case process: {case_key}")
        subprocess.run(command, check=True, env=child_env)
        metadata = load_json(output_root / case_key / "metadata.json")
        panel_metadata["cases"].append(
            {
                "key": case_key,
                "label": str(case_config["label"]),
                "path": str((output_root / case_key).resolve()),
                "fluid_solver": metadata["fluid_solver"],
                "iterations": metadata["iterations"],
                "sim_substeps": metadata["sim_substeps"],
            }
        )

    return panel_metadata


def main() -> None:
    args = parse_args()
    wp.config.use_precompiled_headers = False
    config = load_json(args.config)
    record_frames = int(args.record_frames or config.get("record_frames", 180))
    record_initial_frame = bool(config.get("record_initial_frame", True))
    output_root = resolve_output_root(args, config)
    if args.case_key is None:
        prepare_output_root(output_root, args.overwrite)
    elif not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=True)

    if args.case_key is None and args.isolate_cases and len(config["cases"]) > 1:
        panel_metadata = export_cases_in_subprocesses(
            args=args,
            config=config,
            output_root=output_root,
            record_frames=record_frames,
        )
        save_json(output_root / "panel_metadata.json", panel_metadata)
        return

    if args.case_key is not None:
        matching_cases = [case for case in config["cases"] if str(case["key"]) == args.case_key]
        if not matching_cases:
            raise KeyError(f"Case key not found in config: {args.case_key}")
        cases_to_export = matching_cases
    else:
        cases_to_export = list(config["cases"])

    panel_metadata: dict[str, Any] = {
        "name": str(config["name"]),
        "scene": str(config["scene"]),
        "record_frames": record_frames,
        "record_initial_frame": record_initial_frame,
        "cases": [],
        "axis_conversion": "newton-y-up-to-blender-z-up",
        "config_path": str(args.config.resolve()),
    }

    for case_config in cases_to_export:
        case_key = str(case_config["key"])
        case_dir = output_root / case_key
        case_dir.mkdir(parents=True, exist_ok=True)
        metadata = export_case(
            panel_config=config,
            case_config=case_config,
            case_dir=case_dir,
            record_frames=record_frames,
            record_initial_frame=record_initial_frame,
            device=args.device or config.get("device"),
        )
        panel_metadata["cases"].append(
            {
                "key": case_key,
                "label": str(case_config["label"]),
                "path": str(case_dir),
                "fluid_solver": metadata["fluid_solver"],
                "iterations": metadata["iterations"],
                "sim_substeps": metadata["sim_substeps"],
            }
        )

    save_json(output_root / "panel_metadata.json", panel_metadata)


if __name__ == "__main__":
    main()
