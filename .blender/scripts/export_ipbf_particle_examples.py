from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton.examples
import newton.viewer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export headless IPBF example particle caches for thesis renders.")
    parser.add_argument("--config", type=Path, required=True, help="IPBF particle-example config JSON.")
    parser.add_argument("--output-root", type=Path, default=None, help="Optional override for cache root directory.")
    parser.add_argument("--device", type=str, default=None, help="Warp device, for example cuda:0.")
    parser.add_argument("--record-frames", type=int, default=None, help="Override the recorded frame count for all cases.")
    parser.add_argument("--case-key", type=str, default=None, help="Export only one case key from the config.")
    parser.add_argument(
        "--test-example-config",
        action="store_true",
        help="Pass --test to the Newton examples so the cache export is fast enough for smoke tests.",
    )
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


def example_parser(module: Any):
    create_parser = getattr(module.Example, "create_parser", None)
    if create_parser is not None:
        return create_parser()
    return newton.examples.create_parser()


def build_example_argv(
    *,
    case_config: dict[str, Any],
    device: str | None,
    test_example_config: bool,
) -> list[str]:
    argv: list[str] = []
    if device:
        argv.extend(["--device", device])
    if test_example_config:
        argv.append("--test")
    argv.extend(str(item) for item in case_config.get("example_args", []))
    return argv


def instantiate_example(
    *,
    case_config: dict[str, Any],
    record_frames: int,
    device: str | None,
    test_example_config: bool,
):
    module = importlib.import_module(str(case_config["module"]))
    parser = example_parser(module)
    args = parser.parse_args(
        build_example_argv(case_config=case_config, device=device, test_example_config=test_example_config)
    )
    if getattr(args, "device", None):
        wp.set_device(args.device)
    viewer = newton.viewer.ViewerNull(num_frames=max(record_frames, 1))
    example = module.Example(viewer, args)
    return example


def ipbf_state(example):
    state = getattr(example.state_0, "ipbf", None)
    if state is None:
        raise AttributeError("Example state does not expose an IPBF namespace.")
    return state


def color_from_density_ratio(ratio: np.ndarray) -> np.ndarray:
    ratio = np.asarray(ratio, dtype=np.float32)
    colors = np.empty((ratio.size, 3), dtype=np.float32)
    colors[:] = np.array([0.10, 0.45, 1.00], dtype=np.float32)
    t = np.clip((ratio - 0.95) / 0.30, 0.0, 1.0)[:, None]
    hot = np.array([1.00, 0.18, 0.05], dtype=np.float32)
    cool = np.array([0.00, 0.70, 1.00], dtype=np.float32)
    colors[:] = (1.0 - t) * cool + t * hot
    return colors


def capture_payload(example, *, record_index: int) -> dict[str, np.ndarray]:
    fluid_state = ipbf_state(example)
    positions = example.state_0.particle_q.numpy().astype(np.float32, copy=False)
    velocities = example.state_0.particle_qd.numpy().astype(np.float32, copy=False)
    density = fluid_state.density.numpy().astype(np.float32, copy=False)
    radii = example.model.particle_radius.numpy().astype(np.float32, copy=False)
    rest_density = 1000.0
    solver_config = getattr(getattr(example, "solver", None), "config", None)
    if solver_config is not None and hasattr(solver_config, "rest_density"):
        rest_density = float(solver_config.rest_density)
    ratio = density / max(rest_density, 1.0e-8)

    return {
        "record_index": np.asarray(record_index, dtype=np.int32),
        "frame_index": np.asarray(int(getattr(example, "frame_index", record_index)), dtype=np.int32),
        "sim_time": np.asarray(float(example.sim_time), dtype=np.float64),
        "fluid_positions": positions,
        "fluid_velocities": velocities,
        "fluid_density": density,
        "fluid_colors": color_from_density_ratio(ratio),
        "fluid_render_radii": radii,
    }


def write_frame_payload(case_dir: Path, *, record_index: int, payload: dict[str, np.ndarray]) -> None:
    frame_dir = case_dir / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(frame_dir / f"frame_{record_index:04d}.npz", **payload)


def array_list(value: Any) -> list[list[float]]:
    if value is None:
        return []
    array = value.numpy() if hasattr(value, "numpy") else np.asarray(value)
    return np.asarray(array, dtype=np.float32).reshape(-1, 3).tolist()


def tank_metadata(example) -> dict[str, float]:
    half_width = float(getattr(example, "container_half_width", getattr(example, "container_half_extent", 1.0)))
    half_depth = float(getattr(example, "container_half_depth", getattr(example, "container_half_extent", 1.0)))
    return {
        "half_width": half_width,
        "half_depth": half_depth,
        "floor_y": float(getattr(example, "floor_y", 0.0)),
        "top_y": float(getattr(example, "top_y", 1.0)),
    }


def solver_metadata(example) -> dict[str, float | int]:
    config = getattr(getattr(example, "solver", None), "config", None)
    if config is None:
        return {"rest_density": 1000.0, "iterations": 0, "smoothing_radius": 0.0}
    return {
        "rest_density": float(getattr(config, "rest_density", 1000.0)),
        "iterations": int(getattr(config, "iterations", 0)),
        "smoothing_radius": float(getattr(config, "smoothing_radius", 0.0)),
    }


def build_case_metadata(
    *,
    example,
    panel_config: dict[str, Any],
    case_config: dict[str, Any],
    record_frames: int,
    record_initial_frame: bool,
    sim_ms: list[float],
    test_example_config: bool,
) -> dict[str, Any]:
    solver_meta = solver_metadata(example)
    radii = example.model.particle_radius.numpy().astype(np.float32, copy=False)
    return {
        "name": str(panel_config["name"]),
        "scene": str(case_config["key"]),
        "label": str(case_config["label"]),
        "key": str(case_config["key"]),
        "module": str(case_config["module"]),
        "fluid_solver": "ipbf",
        "iterations": int(solver_meta["iterations"]),
        "sim_substeps": int(getattr(example, "sim_substeps", 0)),
        "record_frames": int(record_frames),
        "record_initial_frame": bool(record_initial_frame),
        "test_example_config": bool(test_example_config),
        "frame_dt": float(example.frame_dt),
        "sim_dt": float(getattr(example, "sim_dt", 0.0)),
        "fps": int(getattr(example, "fps", 60)),
        "rest_density": float(solver_meta["rest_density"]),
        "smoothing_radius": float(solver_meta["smoothing_radius"]),
        "particle_count": int(example.model.particle_count),
        "particle_display_radius": float(np.median(radii)) if radii.size else 0.0,
        "tank": tank_metadata(example),
        "wireframe_starts": array_list(getattr(example, "box_wire_starts", None)),
        "wireframe_ends": array_list(getattr(example, "box_wire_ends", None)),
        "avg_sim_ms_per_frame": float(np.mean(np.asarray(sim_ms, dtype=np.float64))) if sim_ms else 0.0,
        "sim_frame_ms": [float(value) for value in sim_ms],
        "axis_conversion": "newton-y-up-to-blender-z-up",
    }


def case_record_frames(global_frames: int | None, config: dict[str, Any], case_config: dict[str, Any]) -> int:
    if global_frames is not None:
        return int(global_frames)
    if "record_frames" in case_config:
        return int(case_config["record_frames"])
    return int(config.get("record_frames", 180))


def export_case(
    *,
    panel_config: dict[str, Any],
    case_config: dict[str, Any],
    case_dir: Path,
    record_frames: int,
    record_initial_frame: bool,
    device: str | None,
    test_example_config: bool,
) -> dict[str, Any]:
    print(f"[IPBF Particles] Export case: {case_config['key']}")
    example = instantiate_example(
        case_config=case_config,
        record_frames=record_frames,
        device=device,
        test_example_config=test_example_config,
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
        test_example_config=test_example_config,
    )
    save_json(case_dir / "metadata.json", metadata)

    example.viewer.close()
    del example
    gc.collect()
    return metadata


def subprocess_command(
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
    if args.test_example_config:
        command.append("--test-example-config")
    return command


def export_cases_in_subprocesses(
    *,
    args: argparse.Namespace,
    config: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    panel_metadata: dict[str, Any] = {
        "name": str(config["name"]),
        "record_initial_frame": bool(config.get("record_initial_frame", True)),
        "cases": [],
        "axis_conversion": "newton-y-up-to-blender-z-up",
        "config_path": str(args.config.resolve()),
    }
    max_record_frames = 0
    for case_config in config["cases"]:
        case_key = str(case_config["key"])
        record_frames = case_record_frames(args.record_frames, config, case_config)
        max_record_frames = max(max_record_frames, record_frames)
        print(f"[IPBF Particles] Spawn case process: {case_key}")
        subprocess.run(
            subprocess_command(args=args, output_root=output_root, case_key=case_key, record_frames=record_frames),
            check=True,
            env=os.environ.copy(),
        )
        metadata = load_json(output_root / case_key / "metadata.json")
        panel_metadata["cases"].append(
            {
                "key": case_key,
                "label": str(case_config["label"]),
                "path": str((output_root / case_key).resolve()),
                "fluid_solver": "ipbf",
                "iterations": metadata["iterations"],
                "sim_substeps": metadata["sim_substeps"],
                "record_frames": metadata["record_frames"],
            }
        )
    panel_metadata["record_frames"] = max_record_frames
    return panel_metadata


def main() -> None:
    args = parse_args()
    wp.config.use_precompiled_headers = False
    config = load_json(args.config)
    record_initial_frame = bool(config.get("record_initial_frame", True))
    output_root = resolve_output_root(args, config)

    if args.case_key is None:
        prepare_output_root(output_root, args.overwrite)
    elif not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=True)

    if args.case_key is None and args.isolate_cases and len(config["cases"]) > 1:
        panel_metadata = export_cases_in_subprocesses(args=args, config=config, output_root=output_root)
        save_json(output_root / "panel_metadata.json", panel_metadata)
        return

    if args.case_key is None:
        cases_to_export = list(config["cases"])
    else:
        cases_to_export = [case for case in config["cases"] if str(case["key"]) == args.case_key]
        if not cases_to_export:
            raise KeyError(f"Case key not found in config: {args.case_key}")

    panel_metadata: dict[str, Any] = {
        "name": str(config["name"]),
        "record_initial_frame": record_initial_frame,
        "cases": [],
        "axis_conversion": "newton-y-up-to-blender-z-up",
        "config_path": str(args.config.resolve()),
    }
    max_record_frames = 0
    for case_config in cases_to_export:
        case_key = str(case_config["key"])
        record_frames = case_record_frames(args.record_frames, config, case_config)
        max_record_frames = max(max_record_frames, record_frames)
        case_dir = output_root / case_key
        case_dir.mkdir(parents=True, exist_ok=True)
        metadata = export_case(
            panel_config=config,
            case_config=case_config,
            case_dir=case_dir,
            record_frames=record_frames,
            record_initial_frame=record_initial_frame,
            device=args.device or config.get("device"),
            test_example_config=args.test_example_config,
        )
        panel_metadata["cases"].append(
            {
                "key": case_key,
                "label": str(case_config["label"]),
                "path": str(case_dir.resolve()),
                "fluid_solver": "ipbf",
                "iterations": metadata["iterations"],
                "sim_substeps": metadata["sim_substeps"],
                "record_frames": metadata["record_frames"],
            }
        )
    panel_metadata["record_frames"] = max_record_frames
    save_json(output_root / "panel_metadata.json", panel_metadata)


if __name__ == "__main__":
    main()
