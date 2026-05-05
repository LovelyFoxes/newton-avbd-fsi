# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Export Newton FSI fluid caches to splashsurf-compatible particle sequences."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        required=True,
        help="Directory containing metadata.json and frames/frame_*.npz exported by .blender/scripts/export_fsi_cache.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for the exported splashsurf particle sequence. Defaults to <cache-dir>/particles.",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="First cached frame to export.")
    parser.add_argument("--end-frame", type=int, default=None, help="Last cached frame to export, inclusive.")
    parser.add_argument("--stride", type=int, default=1, help="Stride used when selecting cached frames.")
    parser.add_argument(
        "--axis-conversion",
        choices=["newton-y-up-to-blender-z-up", "none"],
        default=None,
        help="Coordinate conversion applied before writing PLY files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove existing frame_*.ply files in the output directory before exporting.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir
    metadata = _load_required_json(cache_dir / "metadata.json")
    output_dir = args.output_dir if args.output_dir is not None else cache_dir / "particles"
    frame_paths = _selected_frame_paths(cache_dir, args.start_frame, args.end_frame, args.stride)
    if not frame_paths:
        raise FileNotFoundError(f"No cache frames found in {cache_dir / 'frames'} for the selected range.")

    axis_mode = _resolve_axis_mode(args.axis_conversion, metadata)
    _prepare_output_dir(output_dir, overwrite=args.overwrite)

    exported_frames: list[dict[str, Any]] = []
    for export_index, frame_path in enumerate(frame_paths):
        positions = _load_positions(frame_path)
        positions = _convert_positions(positions, axis_mode)
        output_path = output_dir / f"frame_{export_index:04d}.ply"
        _write_binary_ply(output_path, positions)

        source_index = _parse_frame_index(frame_path)
        source_sim_time = _load_scalar(frame_path, "sim_time")
        exported_frames.append(
            {
                "export_index": export_index,
                "source_frame": source_index,
                "source_path": str(frame_path),
                "output_path": str(output_path),
                "sim_time": source_sim_time,
                "particle_count": int(len(positions)),
            }
        )
        if not args.quiet and (export_index + 1) % 25 == 0:
            print(f"Exported {export_index + 1}/{len(frame_paths)} splashsurf particle frames")

    splashsurf_metadata = _build_splashsurf_metadata(
        cache_dir=cache_dir,
        output_dir=output_dir,
        source_metadata=metadata,
        axis_mode=axis_mode,
        frame_paths=frame_paths,
        exported_frames=exported_frames,
    )
    _write_json(output_dir / "splashsurf_particles_metadata.json", splashsurf_metadata)

    if not args.quiet:
        recommended = splashsurf_metadata["recommended_reconstruct"]
        print(f"Exported {len(exported_frames)} particle frames to {output_dir}")
        print(
            "Recommended splashsurf reconstruct args: "
            f"-r {recommended['particle_radius']:.6f} "
            f"-l {recommended['smoothing_length']:.6f} "
            f"-c {recommended['cube_size']:.6f} "
            f"-t {recommended['surface_threshold']:.6f}"
        )


def _load_required_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            f"Required cache metadata not found: {path}. "
            "Regenerate the Newton/FSI cache before exporting splashsurf particles."
        )
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _frame_paths(cache_dir: Path) -> list[Path]:
    frame_dir = cache_dir / "frames"
    if not frame_dir.exists():
        raise FileNotFoundError(f"Expected cache frame directory does not exist: {frame_dir}")
    return sorted(frame_dir.glob("frame_*.npz"))


def _selected_frame_paths(cache_dir: Path, start_frame: int, end_frame: int | None, stride: int) -> list[Path]:
    all_paths = _frame_paths(cache_dir)
    if not all_paths:
        return []
    start = max(0, start_frame)
    stop = None if end_frame is None else max(start, end_frame) + 1
    return all_paths[start : stop : max(1, stride)]


def _resolve_axis_mode(requested_mode: str | None, metadata: dict[str, Any]) -> str:
    if requested_mode is not None:
        return requested_mode
    coordinate_system = metadata.get("coordinate_system", {})
    return str(coordinate_system.get("blender_recommended_axis_conversion", "newton-y-up-to-blender-z-up"))


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        existing = list(output_dir.glob("frame_*.ply"))
        if existing and not overwrite:
            raise FileExistsError(
                f"{output_dir} already contains splashsurf particle files. Pass --overwrite to replace them."
            )
        for path in existing:
            path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _load_positions(frame_path: Path) -> np.ndarray:
    with np.load(frame_path, allow_pickle=False) as data:
        positions = np.asarray(data["fluid_positions"], dtype=np.float32)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError(f"{frame_path} does not contain fluid_positions with shape [N, 3].")
    return positions


def _load_scalar(frame_path: Path, key: str) -> float | None:
    with np.load(frame_path, allow_pickle=False) as data:
        if key not in data:
            return None
        value = np.asarray(data[key]).reshape(-1)
        if value.size == 0:
            return None
        return float(value[0])


def _convert_positions(positions: np.ndarray, axis_mode: str) -> np.ndarray:
    if axis_mode == "none":
        return np.asarray(positions, dtype=np.float32)
    converted = np.empty_like(positions, dtype=np.float32)
    converted[:, 0] = positions[:, 0]
    converted[:, 1] = -positions[:, 2]
    converted[:, 2] = positions[:, 1]
    return converted


def _write_binary_ply(path: Path, positions: np.ndarray) -> None:
    points = np.asarray(positions, dtype="<f4", order="C")
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment generated by .blender/scripts/export_splashsurf_particles.py\n"
        f"element vertex {len(points)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with path.open("wb") as file:
        file.write(header.encode("ascii"))
        file.write(points.tobytes())


def _parse_frame_index(frame_path: Path) -> int:
    stem = frame_path.stem
    suffix = stem.split("_")[-1]
    return int(suffix)


def _build_splashsurf_metadata(
    *,
    cache_dir: Path,
    output_dir: Path,
    source_metadata: dict[str, Any],
    axis_mode: str,
    frame_paths: list[Path],
    exported_frames: list[dict[str, Any]],
) -> dict[str, Any]:
    fluid_metadata = source_metadata.get("fluid", {})
    export_metadata = source_metadata.get("export", {})
    particle_radius = float(fluid_metadata.get("particle_radius", 0.0) or 0.0)
    if particle_radius <= 0.0:
        particle_radius = _first_frame_radius(frame_paths)
    support_radius = float(fluid_metadata.get("smoothing_radius", 0.0) or 0.0)
    if support_radius <= 0.0 and particle_radius > 0.0:
        support_radius = 4.0 * particle_radius
    if particle_radius <= 0.0:
        raise ValueError(
            "Invalid splashsurf particle radius: 0. "
            "The cache metadata must contain fluid.particle_radius or frames must contain positive fluid_radii."
        )
    if support_radius <= 0.0:
        raise ValueError("Invalid splashsurf smoothing radius: 0.")
    rest_density = float(fluid_metadata.get("rest_density", 1000.0))
    smoothing_length = support_radius / max(2.0 * particle_radius, 1.0e-12)

    return {
        "source_cache_dir": str(cache_dir),
        "source_scene": source_metadata.get("scene"),
        "source_method": source_metadata.get("method"),
        "axis_conversion": axis_mode,
        "axis_conversion_note": (
            "newton-y-up-to-blender-z-up means (x, y, z)_Newton -> (x, -z, y)_Blender."
            if axis_mode != "none"
            else "No axis conversion was applied."
        ),
        "frame_count": len(exported_frames),
        "frames": exported_frames,
        "sequence_pattern": str(output_dir / "frame_{}.ply"),
        "recommended_reconstruct": {
            "particle_radius": particle_radius,
            "rest_density": rest_density,
            "smoothing_length": smoothing_length,
            "support_radius": support_radius,
            "cube_size": 0.6,
            "surface_threshold": 0.6,
            "mesh_smoothing_iters": 25,
            "mesh_smoothing_weights": True,
            "mesh_cleanup": True,
            "normals": True,
            "normals_smoothing_iters": 10,
            "start_index": 0,
            "end_index": max(0, len(exported_frames) - 1),
            "video_fps": export_metadata.get("actual_video_fps"),
            "recorded_video_duration": export_metadata.get("recorded_video_duration"),
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


def _first_frame_radius(frame_paths: list[Path]) -> float:
    for frame_path in frame_paths[: min(len(frame_paths), 8)]:
        with np.load(frame_path, allow_pickle=False) as data:
            if "fluid_radii" not in data:
                continue
            radii = np.asarray(data["fluid_radii"], dtype=np.float32).reshape(-1)
            radii = radii[np.isfinite(radii) & (radii > 0.0)]
            if radii.size:
                return float(np.median(radii))
    return 0.0


if __name__ == "__main__":
    main()
