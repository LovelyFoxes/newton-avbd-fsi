from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


COLUMNS = [
    "case_key",
    "label",
    "record_index",
    "sim_time",
    "density_error_rms",
    "density_error_max",
    "fluid_center_y",
    "fluid_max_speed",
    "rigid_center_y",
    "rigid_max_speed",
    "cloth_center_y",
    "cloth_max_displacement",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot basic metrics from FSI example caches.")
    parser.add_argument("--cache-root", type=Path, default=Path(".blender/cache/fsi_experiment_scenes"))
    parser.add_argument("--output-dir", type=Path, default=Path(".blender/renders/fsi_experiment_metrics"))
    parser.add_argument("--case-key", action="append", default=None)
    parser.add_argument("--include-initial-frame", action="store_true")
    parser.add_argument("--formats", nargs="+", default=["png"])
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def frame_paths(case_dir: Path) -> list[Path]:
    return sorted((case_dir / "frames").glob("frame_*.npz"))


def scalar(payload, key: str, fallback: float = 0.0) -> float:
    if key not in payload:
        return fallback
    value = np.asarray(payload[key])
    return float(value.reshape(-1)[0]) if value.size else fallback


def array(payload, key: str, *, dtype=np.float64, shape: tuple[int, ...]) -> np.ndarray:
    if key not in payload:
        empty_shape = tuple(0 if int(value) < 0 else int(value) for value in shape)
        return np.empty(empty_shape, dtype=dtype)
    return np.asarray(payload[key], dtype=dtype).reshape(shape)


def rows_for_case(case_dir: Path, include_initial: bool) -> list[dict[str, Any]]:
    metadata = load_json(case_dir / "metadata.json")
    fluid_meta = metadata.get("fluid", {})
    rest_density = float(fluid_meta.get("rest_density", 1000.0))
    label = str(metadata.get("label", case_dir.name))
    initial_cloth = None
    rows: list[dict[str, Any]] = []
    for path in frame_paths(case_dir):
        with np.load(path, allow_pickle=True) as payload:
            record_index = int(scalar(payload, "record_index", len(rows)))
            if record_index == 0 and not include_initial:
                continue
            positions = array(payload, "fluid_positions", shape=(-1, 3))
            velocities = array(payload, "fluid_velocities", shape=(-1, 3))
            density = array(payload, "fluid_density", shape=(-1,))
            if density.size:
                ratio = density / max(rest_density, 1.0e-8)
                density_error_rms = float(np.sqrt(np.mean(np.square(ratio - 1.0))))
                density_error_max = float(np.max(np.abs(ratio - 1.0)))
            else:
                density_error_rms = np.nan
                density_error_max = np.nan
            body_q = array(payload, "body_q", shape=(-1, 7))
            body_qd = array(payload, "body_qd", shape=(-1, 6))
            cloth = array(payload, "cloth_positions", shape=(-1, 3))
            if initial_cloth is None and cloth.size:
                initial_cloth = cloth.copy()
            cloth_disp = np.nan
            if initial_cloth is not None and cloth.shape == initial_cloth.shape:
                cloth_disp = float(np.linalg.norm(cloth - initial_cloth, axis=1).max())
            rows.append(
                {
                    "case_key": str(metadata.get("key", case_dir.name)),
                    "label": label,
                    "record_index": record_index,
                    "sim_time": scalar(payload, "sim_time", 0.0),
                    "density_error_rms": density_error_rms,
                    "density_error_max": density_error_max,
                    "fluid_center_y": float(np.mean(positions[:, 1])) if positions.size else np.nan,
                    "fluid_max_speed": float(np.linalg.norm(velocities, axis=1).max()) if velocities.size else np.nan,
                    "rigid_center_y": float(np.mean(body_q[:, 1])) if body_q.size else np.nan,
                    "rigid_max_speed": float(np.linalg.norm(body_qd[:, :3], axis=1).max()) if body_qd.size else np.nan,
                    "cloth_center_y": float(np.mean(cloth[:, 1])) if cloth.size else np.nan,
                    "cloth_max_displacement": cloth_disp,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in COLUMNS})


def plot_metric(rows: list[dict[str, Any]], metric: str, output_base: Path, formats: list[str], dpi: int) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        value = row.get(metric, np.nan)
        if not np.isfinite(value):
            continue
        grouped.setdefault(str(row["case_key"]), []).append(row)
    if not grouped:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for key, case_rows in grouped.items():
        case_rows = sorted(case_rows, key=lambda item: float(item["sim_time"]))
        ax.plot(
            [float(row["sim_time"]) for row in case_rows],
            [float(row[metric]) for row in case_rows],
            linewidth=2.0,
            label=str(case_rows[0]["label"]),
        )
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(metric.replace("_", " "))
    ax.grid(True, color="#D8DDE6", linewidth=0.8, alpha=0.8)
    ax.legend(fontsize=9)
    for fmt in formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=dpi)
    plt.close(fig)


def case_dirs(cache_root: Path, keys: list[str] | None) -> list[Path]:
    if keys:
        return [cache_root / key for key in keys]
    return [path for path in sorted(cache_root.iterdir()) if (path / "metadata.json").exists()]


def main() -> None:
    args = parse_args()
    all_rows: list[dict[str, Any]] = []
    for case_dir in case_dirs(args.cache_root, args.case_key):
        rows = rows_for_case(case_dir, include_initial=args.include_initial_frame)
        if rows:
            write_csv(args.output_dir / f"{case_dir.name}_metrics.csv", rows)
            all_rows.extend(rows)
    write_csv(args.output_dir / "fsi_experiment_metrics.csv", all_rows)
    for metric in [
        "density_error_rms",
        "density_error_max",
        "fluid_center_y",
        "fluid_max_speed",
        "rigid_center_y",
        "rigid_max_speed",
        "cloth_center_y",
        "cloth_max_displacement",
    ]:
        plot_metric(all_rows, metric, args.output_dir / f"fsi_experiment_{metric}", args.formats, args.dpi)
    print(f"[FSI Metrics] Wrote {len(all_rows)} metric rows to {args.output_dir}")


if __name__ == "__main__":
    main()
