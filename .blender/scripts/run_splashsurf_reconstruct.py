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

"""Run pysplashsurf reconstruct using metadata exported from Newton caches."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--particles-dir",
        type=Path,
        required=True,
        help="Directory containing frame_*.ply and splashsurf_particles_metadata.json.",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where splashsurf writes mesh files.")
    parser.add_argument(
        "--pysplashsurf-exe",
        default="pysplashsurf",
        help="Command used to invoke splashsurf, e.g. pysplashsurf or a full executable path.",
    )
    parser.add_argument("--particle-radius", type=float, default=None)
    parser.add_argument("--rest-density", type=float, default=None)
    parser.add_argument("--smoothing-length", type=float, default=None)
    parser.add_argument("--cube-size", type=float, default=None)
    parser.add_argument("--surface-threshold", type=float, default=None)
    parser.add_argument("--start-index", type=int, default=None)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--mesh-smoothing-iters", type=int, default=None)
    parser.add_argument("--normals-smoothing-iters", type=int, default=None)
    parser.add_argument("--mesh-smoothing-weights", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mesh-cleanup", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--normals", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mt-files", choices=["off", "on"], default="off")
    parser.add_argument("--mt-particles", choices=["off", "on"], default="on")
    parser.add_argument("--num-threads", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    particles_metadata = _load_json(args.particles_dir / "splashsurf_particles_metadata.json")
    recommended = particles_metadata.get("recommended_reconstruct", {})

    particle_radius = _resolve_float(args.particle_radius, recommended, "particle_radius")
    rest_density = _resolve_float(args.rest_density, recommended, "rest_density")
    smoothing_length = _resolve_float(args.smoothing_length, recommended, "smoothing_length")
    cube_size = _resolve_float(args.cube_size, recommended, "cube_size")
    surface_threshold = _resolve_float(args.surface_threshold, recommended, "surface_threshold")
    start_index = _resolve_int(args.start_index, recommended, "start_index")
    end_index = _resolve_int(args.end_index, recommended, "end_index")
    mesh_smoothing_iters = _resolve_int(args.mesh_smoothing_iters, recommended, "mesh_smoothing_iters")
    normals_smoothing_iters = _resolve_int(args.normals_smoothing_iters, recommended, "normals_smoothing_iters")
    mesh_smoothing_weights = _resolve_bool(args.mesh_smoothing_weights, recommended, "mesh_smoothing_weights")
    mesh_cleanup = _resolve_bool(args.mesh_cleanup, recommended, "mesh_cleanup")
    normals = _resolve_bool(args.normals, recommended, "normals")

    _prepare_output_dir(args.output_dir, overwrite=args.overwrite)

    command = [
        args.pysplashsurf_exe,
        "reconstruct",
        str(args.particles_dir / "frame_{}.ply"),
        "--output-dir",
        str(args.output_dir),
        "--output-file",
        "frame_{}.obj",
        "--particle-radius",
        _fmt(particle_radius),
        "--rest-density",
        _fmt(rest_density),
        "--smoothing-length",
        _fmt(smoothing_length),
        "--cube-size",
        _fmt(cube_size),
        "--surface-threshold",
        _fmt(surface_threshold),
        "--start-index",
        str(start_index),
        "--end-index",
        str(end_index),
        f"--mesh-smoothing-weights={'on' if mesh_smoothing_weights else 'off'}",
        f"--mesh-cleanup={'on' if mesh_cleanup else 'off'}",
        f"--normals={'on' if normals else 'off'}",
        f"--mt-files={args.mt_files}",
        f"--mt-particles={args.mt_particles}",
    ]
    if mesh_smoothing_iters > 0:
        command.extend(["--mesh-smoothing-iters", str(mesh_smoothing_iters)])
    if normals and normals_smoothing_iters > 0:
        command.extend(["--normals-smoothing-iters", str(normals_smoothing_iters)])
    if args.num_threads is not None:
        command.extend(["--num-threads", str(args.num_threads)])
    if args.quiet:
        command.append("--quiet")

    _write_json(
        args.output_dir / "splashsurf_reconstruct_command.json",
        {
            "command": command,
            "particles_dir": str(args.particles_dir),
            "output_dir": str(args.output_dir),
            "particles_metadata": particles_metadata,
        },
    )

    if args.dry_run:
        print("Dry run command:")
        print(" ".join(command))
        return

    if not args.quiet:
        print("Running splashsurf reconstruct:")
        print(" ".join(command))
    subprocess.run(command, check=True)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _resolve_float(value: float | None, recommended: dict[str, Any], key: str) -> float:
    result = value if value is not None else recommended.get(key)
    if result is None:
        raise ValueError(f"Missing required splashsurf parameter: {key}")
    return float(result)


def _resolve_int(value: int | None, recommended: dict[str, Any], key: str) -> int:
    result = value if value is not None else recommended.get(key)
    if result is None:
        raise ValueError(f"Missing required splashsurf parameter: {key}")
    return int(result)


def _resolve_bool(value: bool | None, recommended: dict[str, Any], key: str) -> bool:
    if value is not None:
        return bool(value)
    return bool(recommended.get(key, False))


def _prepare_output_dir(output_dir: Path, *, overwrite: bool) -> None:
    if output_dir.exists():
        existing = list(output_dir.glob("frame_*.obj"))
        if existing and not overwrite:
            raise FileExistsError(f"{output_dir} already contains splashsurf meshes. Pass --overwrite to replace them.")
        for path in existing:
            path.unlink()
    output_dir.mkdir(parents=True, exist_ok=True)


def _fmt(value: float) -> str:
    return f"{float(value):.8g}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
        file.write("\n")


if __name__ == "__main__":
    main()
