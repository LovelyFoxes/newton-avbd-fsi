from __future__ import annotations

import argparse
import importlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton
import newton.examples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export FSI example caches for Blender thesis figures.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--case-key", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--record-frames", type=int, default=None)
    parser.add_argument("--test-example-config", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--compress-cache", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def array_or_empty(value: Any, *, dtype=np.float32, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if value is None:
        return np.empty(shape or (0,), dtype=dtype)
    return np.asarray(value, dtype=dtype)


def wp_numpy(value: Any) -> np.ndarray:
    if value is None:
        return np.empty((0,), dtype=np.float32)
    return value.numpy()


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = [float(v) for v in q1]
    x2, y2, z2, w2 = [float(v) for v in q2]
    return np.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=np.float32,
    )


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([float(v[0]), float(v[1]), float(v[2]), 0.0], dtype=np.float32)
    qi = np.array([-float(q[0]), -float(q[1]), -float(q[2]), float(q[3])], dtype=np.float32)
    return quat_mul(quat_mul(q, qv), qi)[:3]


def transform_mul(parent: np.ndarray, child: np.ndarray) -> np.ndarray:
    parent = np.asarray(parent, dtype=np.float32)
    child = np.asarray(child, dtype=np.float32)
    out = np.empty(7, dtype=np.float32)
    out[:3] = parent[:3] + quat_rotate(parent[3:7], child[:3])
    out[3:7] = quat_mul(parent[3:7], child[3:7])
    norm = float(np.linalg.norm(out[3:7]))
    if norm > 1.0e-8:
        out[3:7] /= norm
    return out


def shape_type_name(type_value: int) -> str:
    try:
        return str(newton.GeoType(int(type_value)).name).lower()
    except Exception:
        return str(int(type_value))


def selected_cases(config: dict[str, Any], case_key: str | None) -> list[dict[str, Any]]:
    cases = [case for case in config["cases"] if case_key is None or str(case["key"]) == case_key]
    if not cases:
        raise KeyError(f"Case key not found: {case_key}")
    return cases


def build_example(case: dict[str, Any], *, device: str, record_frames: int, test_config: bool):
    module = importlib.import_module(str(case["module"]))
    parser = module.Example.create_parser()
    argv = ["--viewer", "null", "--num-frames", str(record_frames), "--device", device, "--headless"]
    if test_config:
        argv.append("--test")
    argv.extend(str(item) for item in case.get("example_args", []))
    args = parser.parse_args(argv)
    if getattr(args, "quiet", False):
        wp.config.quiet = True
    if args.device:
        wp.set_device(args.device)
    viewer = newton.viewer.ViewerNull(num_frames=record_frames)
    return module.Example(viewer, args)


def particle_slice(example) -> slice:
    start = int(getattr(example, "fluid_particle_start", 0))
    count = getattr(example, "fluid_particle_count", None)
    if count is None:
        count = int(getattr(example.model, "particle_count", 0))
        start = 0
    return slice(start, start + int(count))


def sliced_density(example, sl: slice) -> np.ndarray:
    state = getattr(example.state_0, "ipbf", None)
    if state is None or not hasattr(state, "density"):
        return np.empty((0,), dtype=np.float32)
    density = state.density.numpy().astype(np.float32, copy=False).reshape(-1)
    if density.shape[0] >= sl.stop:
        return density[sl]
    return density[: max(0, sl.stop - sl.start)]


def shape_world_transforms(example) -> np.ndarray:
    model = example.model
    local = wp_numpy(model.shape_transform).astype(np.float32, copy=False).reshape(-1, 7)
    if len(local) == 0:
        return local
    body = wp_numpy(model.shape_body).astype(np.int32, copy=False).reshape(-1)
    body_q = wp_numpy(example.state_0.body_q).astype(np.float32, copy=False).reshape(-1, 7)
    world = np.empty_like(local)
    for index, local_tf in enumerate(local):
        body_index = int(body[index]) if index < len(body) else -1
        if body_index >= 0 and body_index < len(body_q):
            world[index] = transform_mul(body_q[body_index], local_tf)
        else:
            world[index] = local_tf
    return world


def wireframes(example) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    names = sorted(name[:-7] for name in dir(example) if name.endswith("_starts"))
    for prefix in names:
        starts = getattr(example, f"{prefix}_starts", None)
        ends = getattr(example, f"{prefix}_ends", None)
        if starts is None or ends is None:
            continue
        starts_np = np.asarray(starts, dtype=np.float32).reshape(-1, 3)
        ends_np = np.asarray(ends, dtype=np.float32).reshape(-1, 3)
        if starts_np.size == 0 or starts_np.shape != ends_np.shape:
            continue
        records.append({"name": prefix, "starts": starts_np.tolist(), "ends": ends_np.tolist()})
    return records


def cloth_metadata(example) -> dict[str, Any]:
    start = getattr(example, "cloth_particle_start", None)
    count = getattr(example, "cloth_particle_count", None)
    tri_start = getattr(example, "cloth_triangle_start", None)
    tri_count = getattr(example, "cloth_triangle_count", None)
    if start is None or count is None or int(count) <= 0:
        return {}
    tri_indices = wp_numpy(getattr(example.model, "tri_indices", None)).astype(np.int32, copy=False)
    if tri_indices.size == 0:
        return {}
    tri_indices = tri_indices.reshape(-1, 3)
    if tri_start is not None and tri_count is not None:
        tri_indices = tri_indices[int(tri_start) : int(tri_start) + int(tri_count)]
    local = tri_indices - int(start)
    mask = np.all((local >= 0) & (local < int(count)), axis=1)
    return {
        "particle_start": int(start),
        "particle_count": int(count),
        "triangle_start": None if tri_start is None else int(tri_start),
        "triangle_count": int(len(local[mask])),
        "indices": local[mask].astype(np.int32, copy=False).tolist(),
    }


def shape_metadata(example) -> dict[str, Any]:
    model = example.model
    shape_type = wp_numpy(model.shape_type).astype(np.int32, copy=False).reshape(-1)
    shape_body = wp_numpy(model.shape_body).astype(np.int32, copy=False).reshape(-1)
    shape_scale = wp_numpy(model.shape_scale).astype(np.float32, copy=False).reshape(-1, 3)
    shape_local = wp_numpy(model.shape_transform).astype(np.float32, copy=False).reshape(-1, 7)
    labels = list(getattr(model, "shape_label", []))
    body_labels = list(getattr(model, "body_label", []))
    records = []
    for index, type_value in enumerate(shape_type):
        type_name = shape_type_name(int(type_value))
        if type_name == "plane":
            continue
        body_index = int(shape_body[index]) if index < len(shape_body) else -1
        records.append(
            {
                "index": index,
                "type": int(type_value),
                "type_name": type_name,
                "body": body_index,
                "body_label": body_labels[body_index] if 0 <= body_index < len(body_labels) else "static",
                "label": labels[index] if index < len(labels) else f"shape_{index}",
                "scale": shape_scale[index].tolist() if index < len(shape_scale) else [1.0, 1.0, 1.0],
                "local_transform": shape_local[index].tolist() if index < len(shape_local) else [0, 0, 0, 0, 0, 0, 1],
            }
        )
    return {"shapes": records, "body_labels": body_labels}


def capture_frame(example, record_index: int) -> dict[str, np.ndarray]:
    sl = particle_slice(example)
    particle_q = example.state_0.particle_q.numpy().astype(np.float32, copy=False)
    particle_qd = example.state_0.particle_qd.numpy().astype(np.float32, copy=False)
    payload: dict[str, np.ndarray] = {
        "record_index": np.asarray(record_index, dtype=np.int32),
        "frame_index": np.asarray(record_index, dtype=np.int32),
        "sim_time": np.asarray(float(getattr(example, "sim_time", 0.0)), dtype=np.float32),
        "fluid_positions": particle_q[sl],
        "fluid_velocities": particle_qd[sl],
        "fluid_density": sliced_density(example, sl),
        "shape_world_q": shape_world_transforms(example),
    }
    if getattr(example.state_0, "body_q", None) is not None:
        payload["body_q"] = example.state_0.body_q.numpy().astype(np.float32, copy=False)
    if getattr(example.state_0, "body_qd", None) is not None:
        payload["body_qd"] = example.state_0.body_qd.numpy().astype(np.float32, copy=False)
    radii = wp_numpy(getattr(example.model, "particle_radius", None)).astype(np.float32, copy=False).reshape(-1)
    if radii.shape[0] >= sl.stop:
        payload["fluid_radii"] = radii[sl]
    if hasattr(example, "cloth_particle_start") and hasattr(example, "cloth_particle_count"):
        start = int(getattr(example, "cloth_particle_start"))
        count = int(getattr(example, "cloth_particle_count"))
        payload["cloth_positions"] = particle_q[start : start + count]
        payload["cloth_velocities"] = particle_qd[start : start + count]
    boundary = getattr(example, "boundary_model", None)
    if boundary is not None:
        if hasattr(boundary, "body_force"):
            payload["body_force"] = boundary.body_force.numpy().astype(np.float32, copy=False)
        if hasattr(boundary, "sample_force"):
            payload["sample_force"] = boundary.sample_force.numpy().astype(np.float32, copy=False)
    return payload


def metadata(example, case: dict[str, Any], config: dict[str, Any], record_frames: int) -> dict[str, Any]:
    sl = particle_slice(example)
    solver_config = getattr(getattr(example, "fluid_solver", None), "config", None)
    scene_config = getattr(example, "config", {})
    particle_radius = wp_numpy(getattr(example.model, "particle_radius", None)).astype(np.float32, copy=False).reshape(-1)
    default_radius = float(np.median(particle_radius[sl])) if particle_radius.size and sl.stop <= len(particle_radius) else 0.006
    meta = {
        "key": str(case["key"]),
        "label": str(case.get("label", case["key"])),
        "module": str(case["module"]),
        "group": str(case.get("group", "")),
        "method": "IPBF-VBD FSI",
        "record_frames": int(record_frames),
        "record_initial_frame": bool(config.get("record_initial_frame", True)),
        "render_frames": [int(value) for value in case.get("render_frames", [])],
        "frame_dt": float(getattr(example, "frame_dt", 0.0)),
        "sim_dt": float(getattr(example, "sim_dt", 0.0)),
        "sim_substeps": int(getattr(example, "sim_substeps", 0)),
        "fps": int(getattr(example, "fps", config.get("render", {}).get("fps", 60))),
        "fluid": {
            "particle_start": int(sl.start),
            "particle_count": int(sl.stop - sl.start),
            "particle_radius": default_radius,
            "rest_density": float(getattr(solver_config, "rest_density", scene_config.get("rest_density", 1000.0))),
            "smoothing_radius": float(getattr(solver_config, "smoothing_radius", scene_config.get("smoothing_radius", 0.0))),
        },
        "coordinate_system": {"blender_recommended_axis_conversion": config.get("axis_conversion", "newton-y-up-to-blender-z-up")},
        "axis_conversion": config.get("axis_conversion", "newton-y-up-to-blender-z-up"),
        "example_config": jsonable(scene_config),
        "camera": jsonable(case.get("camera", {})),
        "wireframes": wireframes(example),
        "cloth": cloth_metadata(example),
        "export": {"actual_video_fps": int(config.get("render", {}).get("fps", 60))},
    }
    meta.update(shape_metadata(example))
    return meta


def save_npz(path: Path, payload: dict[str, np.ndarray], *, compressed: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if compressed:
        np.savez_compressed(path, **payload)
    else:
        np.savez(path, **payload)


def export_case(case: dict[str, Any], config: dict[str, Any], args: argparse.Namespace, output_root: Path) -> dict[str, Any]:
    record_frames = int(args.record_frames or case.get("record_frames", config.get("record_frames", 180)))
    case_dir = output_root / str(case["key"])
    if case_dir.exists() and args.overwrite:
        shutil.rmtree(case_dir)
    frames_dir = case_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    print(f"[FSI Example Cache] Export {case['key']} ({record_frames} frames)")
    example = build_example(case, device=args.device, record_frames=record_frames, test_config=args.test_example_config)
    save_npz(frames_dir / "frame_0000.npz", capture_frame(example, 0), compressed=args.compress_cache)
    for record_index in range(1, record_frames):
        example.step()
        save_npz(
            frames_dir / f"frame_{record_index:04d}.npz",
            capture_frame(example, record_index),
            compressed=args.compress_cache,
        )
        if record_index % 25 == 0:
            print(f"[FSI Example Cache] {case['key']}: {record_index}/{record_frames}")

    meta = metadata(example, case, config, record_frames)
    write_json(case_dir / "metadata.json", jsonable(meta))
    return {
        "key": str(case["key"]),
        "label": str(case.get("label", case["key"])),
        "path": str(case_dir.resolve()),
        "record_frames": record_frames,
        "group": str(case.get("group", "")),
    }


def main() -> None:
    args = parse_args()
    config = load_json(args.config)
    output_root = (args.output_root or Path(config.get("cache_root", ".blender/cache/fsi_experiment_scenes"))).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    panel = {"name": config.get("name", "fsi_experiment_scenes"), "cases": []}
    for case in selected_cases(config, args.case_key):
        panel["cases"].append(export_case(case, config, args, output_root))
    write_json(output_root / "panel_metadata.json", jsonable(panel))


if __name__ == "__main__":
    main()
