from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator


METRIC_COLUMNS = [
    "panel",
    "case_key",
    "label",
    "record_index",
    "frame_index",
    "sim_time",
    "density_error_max",
    "density_error_p95",
    "density_error_rms",
    "density_ratio_min",
    "density_ratio_mean",
    "density_ratio_max",
    "particle_center_y",
    "particle_std_x",
    "particle_std_y",
    "particle_std_z",
    "particle_std_xz",
    "particle_mean_abs_x",
    "particle_center_band_fraction",
    "particle_min_y",
    "particle_max_y",
    "mean_speed",
    "max_speed",
]


CASE_COLORS = {
    "ipbf_double_dam_break": "#1F77B4",
    "ipbf_block_flop": "#2CA02C",
    "ipbf_3d_compression": "#D62728",
}

TITLE_FONT_SIZE = 18.0
AXIS_LABEL_FONT_SIZE = 16.0
TICK_LABEL_FONT_SIZE = 13.0
LEGEND_FONT_SIZE = 13.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot metrics from IPBF particle-example caches.")
    parser.add_argument("--cache-root", type=Path, default=Path(".blender/cache/ipbf_particle_examples"))
    parser.add_argument("--output-dir", type=Path, default=Path(".blender/renders/ipbf_particle_metrics"))
    parser.add_argument("--config", type=Path, default=Path(".blender/config/ipbf_particle_examples.json"))
    parser.add_argument("--case-key", action="append", default=None, help="Only plot the selected case key. Repeatable.")
    parser.add_argument("--include-initial-frame", action="store_true")
    parser.add_argument("--formats", type=str, default="pdf,png,svg")
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def scalar(payload: np.lib.npyio.NpzFile, key: str, fallback: float | int = 0) -> float:
    if key not in payload:
        return float(fallback)
    value = np.asarray(payload[key])
    return float(value.reshape(-1)[0]) if value.size else float(fallback)


def frame_paths(case_dir: Path) -> list[Path]:
    paths = sorted((case_dir / "frames").glob("frame_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No frame_*.npz files found in {case_dir / 'frames'}")
    return paths


def case_path(panel_dir: Path, case_entry: dict[str, Any]) -> Path:
    raw_value = str(case_entry.get("path", ""))
    raw = Path(raw_value)
    if raw_value and raw.exists():
        return raw
    return panel_dir / str(case_entry["key"])


def load_panel_metadata(cache_root: Path, config_path: Path) -> tuple[str, list[dict[str, Any]]]:
    panel_metadata = load_json(cache_root / "panel_metadata.json") if (cache_root / "panel_metadata.json").exists() else {}
    config = load_json(config_path) if config_path.exists() else {}
    panel_name = str(config.get("name", panel_metadata.get("name", cache_root.name)))
    cases = config.get("cases") or panel_metadata.get("cases", [])
    return panel_name, cases


def compute_case_rows(
    *,
    panel_name: str,
    case_dir: Path,
    case_entry: dict[str, Any],
    include_initial_frame: bool,
) -> list[dict[str, Any]]:
    metadata = load_json(case_dir / "metadata.json")
    rest_density = float(metadata.get("rest_density", 1000.0))
    label = str(metadata.get("label", case_entry.get("label", case_entry["key"]))).replace("\n", " ")
    rows: list[dict[str, Any]] = []
    for path in frame_paths(case_dir):
        with np.load(path) as payload:
            record_index = int(scalar(payload, "record_index", len(rows)))
            if not include_initial_frame and record_index == 0:
                continue
            positions = np.asarray(payload["fluid_positions"], dtype=np.float64).reshape(-1, 3)
            density = np.asarray(payload["fluid_density"], dtype=np.float64).reshape(-1)
            if "fluid_velocities" in payload:
                velocities = np.asarray(payload["fluid_velocities"], dtype=np.float64).reshape(-1, 3)
            else:
                velocities = np.zeros_like(positions)
            ratio = density / max(rest_density, 1.0e-8)
            abs_error = np.abs(ratio - 1.0)
            speeds = np.linalg.norm(velocities, axis=1)
            row = {
                "panel": panel_name,
                "case_key": str(metadata.get("key", case_entry["key"])),
                "label": label,
                "record_index": record_index,
                "frame_index": int(scalar(payload, "frame_index", record_index)),
                "sim_time": scalar(payload, "sim_time", record_index * float(metadata.get("frame_dt", 0.0))),
                "density_error_max": float(np.max(abs_error)),
                "density_error_p95": float(np.percentile(abs_error, 95.0)),
                "density_error_rms": float(np.sqrt(np.mean(np.square(ratio - 1.0)))),
                "density_ratio_min": float(np.min(ratio)),
                "density_ratio_mean": float(np.mean(ratio)),
                "density_ratio_max": float(np.max(ratio)),
                "particle_center_y": float(np.mean(positions[:, 1])),
                "particle_std_x": float(np.std(positions[:, 0])),
                "particle_std_y": float(np.std(positions[:, 1])),
                "particle_std_z": float(np.std(positions[:, 2])),
                "particle_std_xz": 0.5 * float(np.std(positions[:, 0]) + np.std(positions[:, 2])),
                "particle_mean_abs_x": float(np.mean(np.abs(positions[:, 0]))),
                "particle_center_band_fraction": float(np.mean(np.abs(positions[:, 0]) <= 0.15)),
                "particle_min_y": float(np.min(positions[:, 1])),
                "particle_max_y": float(np.max(positions[:, 1])),
                "mean_speed": float(np.mean(speeds)),
                "max_speed": float(np.max(speeds)),
            }
            rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in METRIC_COLUMNS})


def grouped(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for row in rows:
        label = str(row["label"])
        if label not in groups:
            order.append(label)
            groups[label] = []
        groups[label].append(row)
    return [(label, groups[label]) for label in order]


def plot_metric(
    *,
    rows: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    output_base: Path,
    formats: list[str],
    dpi: int,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    for label, case_rows in grouped(rows):
        ordered = sorted(case_rows, key=lambda item: (int(item["frame_index"]), int(item["record_index"])))
        xs = np.asarray([float(item["frame_index"]) for item in ordered], dtype=np.float64)
        ys = np.asarray([float(item[metric]) for item in ordered], dtype=np.float64)
        key = str(ordered[0]["case_key"])
        ax.plot(xs, ys, linewidth=2.2, color=CASE_COLORS.get(key), label=label)
    ax.set_title(title, fontsize=TITLE_FONT_SIZE, pad=10)
    ax.set_xlabel("Frame", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.legend(fontsize=LEGEND_FONT_SIZE)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=dpi)
    plt.close(fig)


def safe_stem(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in name).strip("_") or "ipbf_examples"


def title_for(rows: list[dict[str, Any]], metric_title: str) -> str:
    labels = {str(row["label"]) for row in rows}
    if len(labels) == 1:
        return f"{next(iter(labels))}: {metric_title}"
    return f"IPBF particle examples: {metric_title}"


def main() -> None:
    args = parse_args()
    panel_name, case_entries = load_panel_metadata(args.cache_root, args.config)
    selected_keys = set(args.case_key or [])
    prefix = safe_stem(panel_name if not selected_keys else "_".join(sorted(selected_keys)))
    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]

    all_rows: list[dict[str, Any]] = []
    for case_entry in case_entries:
        if selected_keys and str(case_entry["key"]) not in selected_keys:
            continue
        case_dir = case_path(args.cache_root, case_entry)
        rows = compute_case_rows(
            panel_name=panel_name,
            case_dir=case_dir,
            case_entry=case_entry,
            include_initial_frame=args.include_initial_frame,
        )
        write_csv(args.output_dir / f"{case_entry['key']}_metrics.csv", rows)
        all_rows.extend(rows)

    write_csv(args.output_dir / f"{prefix}_metrics.csv", all_rows)
    if not all_rows:
        print(f"[IPBF Metrics] No metric rows found for {sorted(selected_keys)} in {args.cache_root}")
        return
    plot_metric(
        rows=all_rows,
        metric="density_error_rms",
        ylabel="RMS relative density error",
        title=title_for(all_rows, "RMS density error"),
        output_base=args.output_dir / f"{prefix}_density_error_rms",
        formats=formats,
        dpi=args.dpi,
    )
    plot_metric(
        rows=all_rows,
        metric="density_error_max",
        ylabel="Max relative density error",
        title=title_for(all_rows, "max density error"),
        output_base=args.output_dir / f"{prefix}_density_error_max",
        formats=formats,
        dpi=args.dpi,
    )
    plot_metric(
        rows=all_rows,
        metric="particle_center_y",
        ylabel="Mean particle height (m)",
        title=title_for(all_rows, "mean particle height"),
        output_base=args.output_dir / f"{prefix}_particle_center_y",
        formats=formats,
        dpi=args.dpi,
    )
    plot_metric(
        rows=all_rows,
        metric="particle_std_xz",
        ylabel="Horizontal spread (m)",
        title=title_for(all_rows, "horizontal spread"),
        output_base=args.output_dir / f"{prefix}_particle_spread_xz",
        formats=formats,
        dpi=args.dpi,
    )
    plot_metric(
        rows=all_rows,
        metric="max_speed",
        ylabel="Max particle speed (m/s)",
        title=title_for(all_rows, "max particle speed"),
        output_base=args.output_dir / f"{prefix}_max_speed",
        formats=formats,
        dpi=args.dpi,
    )
    print(f"[IPBF Metrics] Wrote {len(all_rows)} metric rows to {args.output_dir}")


if __name__ == "__main__":
    main()
