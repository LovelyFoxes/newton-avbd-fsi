from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


METRIC_COLUMNS = [
    "panel",
    "case_key",
    "label",
    "fluid_solver",
    "iterations",
    "sim_substeps",
    "record_index",
    "frame_index",
    "sim_time",
    "density_error_max",
    "density_error_rms",
    "density_error_p95",
    "density_ratio_min",
    "density_ratio_mean",
    "density_ratio_max",
    "box_x",
    "box_y",
    "box_z",
    "box_vx",
    "box_vy",
    "box_vz",
]

TRUNCATION_COLUMNS = [
    "panel",
    "case_key",
    "label",
    "fluid_solver",
    "truncated",
    "kept_frame_rows",
    "total_frame_rows",
    "truncate_record_index",
    "truncate_sim_time",
    "truncate_density_error_max",
    "truncate_density_error_rms",
    "truncate_density_error_p95",
    "truncate_reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot density-error curves from fluid comparison cache directories.")
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path(".blender/cache"),
        help="A panel cache directory, or a parent directory containing panel_metadata.json files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(".blender/renders/density_error_curves"),
        help="Directory for CSV files and plotted figures.",
    )
    parser.add_argument(
        "--panel",
        action="append",
        default=None,
        help="Optional panel directory name to include. Can be passed more than once.",
    )
    parser.add_argument(
        "--include-initial-frame",
        action="store_true",
        help="Include record_index 0. By default it is skipped because it may be captured before density evaluation.",
    )
    parser.add_argument(
        "--no-truncate-blowup",
        action="store_true",
        help="Do not truncate case curves after the first blow-up frame.",
    )
    parser.add_argument(
        "--blowup-error-threshold",
        type=float,
        default=3.0,
        help="Truncate after the first frame whose max relative density error exceeds this threshold.",
    )
    parser.add_argument(
        "--truncate-solvers",
        type=str,
        default="pbf",
        help="Comma-separated solver names to truncate on blow-up, or 'all'. Default: pbf.",
    )
    parser.add_argument("--dpi", type=int, default=220, help="DPI for PNG output.")
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["svg", "png"],
        choices=["svg", "png", "pdf"],
        help="Figure formats to write.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_panel_dirs(cache_root: Path, panel_names: list[str] | None) -> list[Path]:
    root = cache_root.resolve()
    if (root / "panel_metadata.json").exists():
        panel_dirs = [root]
    else:
        panel_dirs = sorted(path for path in root.iterdir() if (path / "panel_metadata.json").exists())

    if panel_names:
        wanted = set(panel_names)
        panel_dirs = [path for path in panel_dirs if path.name in wanted]
    return panel_dirs


def frame_paths(case_dir: Path) -> list[Path]:
    paths = sorted((case_dir / "frames").glob("frame_*.npz"))
    if not paths:
        raise FileNotFoundError(f"No frame_*.npz files found in {case_dir / 'frames'}")
    return paths


def case_path(panel_dir: Path, case_entry: dict[str, Any]) -> Path:
    raw_path = Path(str(case_entry.get("path", "")))
    if raw_path.exists():
        return raw_path
    return panel_dir / str(case_entry["key"])


def scalar_from_payload(payload: np.lib.npyio.NpzFile, key: str, fallback: float | int = 0) -> float:
    if key not in payload:
        return float(fallback)
    value = np.asarray(payload[key])
    return float(value.reshape(-1)[0]) if value.size else float(fallback)


def extract_box_values(payload: np.lib.npyio.NpzFile) -> dict[str, float | None]:
    values: dict[str, float | None] = {
        "box_x": None,
        "box_y": None,
        "box_z": None,
        "box_vx": None,
        "box_vy": None,
        "box_vz": None,
    }

    if "box_body_q" in payload:
        body_q = np.asarray(payload["box_body_q"], dtype=np.float64).reshape(-1)
        if body_q.size >= 3:
            values["box_x"] = float(body_q[0])
            values["box_y"] = float(body_q[1])
            values["box_z"] = float(body_q[2])

    if "box_body_qd" in payload:
        body_qd = np.asarray(payload["box_body_qd"], dtype=np.float64).reshape(-1)
        if body_qd.size >= 3:
            values["box_vx"] = float(body_qd[0])
            values["box_vy"] = float(body_qd[1])
            values["box_vz"] = float(body_qd[2])

    return values


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
    for frame_path in frame_paths(case_dir):
        with np.load(frame_path) as payload:
            record_index = int(scalar_from_payload(payload, "record_index", len(rows)))
            if not include_initial_frame and record_index == 0:
                continue

            if "fluid_density" not in payload:
                raise KeyError(f"{frame_path} does not contain 'fluid_density'.")
            density = np.asarray(payload["fluid_density"], dtype=np.float64).reshape(-1)
            if density.size == 0:
                continue

            ratio = density / max(rest_density, 1.0e-8)
            abs_error = np.abs(ratio - 1.0)
            row: dict[str, Any] = {
                "panel": panel_name,
                "case_key": str(metadata.get("key", case_entry["key"])),
                "label": label,
                "fluid_solver": str(metadata.get("fluid_solver", case_entry.get("fluid_solver", ""))),
                "iterations": int(metadata.get("iterations", case_entry.get("iterations", 0))),
                "sim_substeps": int(metadata.get("sim_substeps", case_entry.get("sim_substeps", 0))),
                "record_index": record_index,
                "frame_index": int(scalar_from_payload(payload, "frame_index", record_index)),
                "sim_time": scalar_from_payload(payload, "sim_time", record_index * float(metadata.get("frame_dt", 0.0))),
                "density_error_max": float(np.max(abs_error)),
                "density_error_rms": float(np.sqrt(np.mean(np.square(ratio - 1.0)))),
                "density_error_p95": float(np.percentile(abs_error, 95.0)),
                "density_ratio_min": float(np.min(ratio)),
                "density_ratio_mean": float(np.mean(ratio)),
                "density_ratio_max": float(np.max(ratio)),
            }
            row.update(extract_box_values(payload))
            rows.append(row)

    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in METRIC_COLUMNS})


def parse_solver_filter(value: str) -> set[str] | None:
    solvers = {item.strip().lower() for item in value.split(",") if item.strip()}
    if not solvers or "all" in solvers:
        return None
    if "none" in solvers:
        return set()
    return solvers


def should_truncate_solver(row: dict[str, Any], solver_filter: set[str] | None) -> bool:
    if solver_filter is None:
        return True
    return str(row.get("fluid_solver", "")).lower() in solver_filter


def blowup_reason(row: dict[str, Any], threshold: float) -> str | None:
    values = [
        float(row["density_error_max"]),
        float(row["density_error_rms"]),
        float(row["density_error_p95"]),
        float(row["density_ratio_max"]),
    ]
    if not np.isfinite(values).all():
        return "nonfinite density statistic"
    if float(row["density_error_max"]) > threshold:
        return f"density_error_max>{threshold:g}"
    return None


def truncate_rows_on_blowup(
    rows: list[dict[str, Any]],
    *,
    enabled: bool,
    solver_filter: set[str] | None,
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for row in rows:
        key = str(row["case_key"])
        if key not in grouped:
            order.append(key)
            grouped[key] = []
        grouped[key].append(row)

    kept_rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for key in order:
        case_rows = sorted(grouped[key], key=lambda item: (float(item["sim_time"]), int(item["record_index"])))
        cutoff_index: int | None = None
        reason: str | None = None
        if enabled and case_rows and should_truncate_solver(case_rows[0], solver_filter):
            for index, row in enumerate(case_rows):
                reason = blowup_reason(row, threshold)
                if reason is not None:
                    cutoff_index = index
                    break

        if cutoff_index is None:
            kept = case_rows
            cutoff_row: dict[str, Any] | None = None
            reason = ""
        else:
            kept = case_rows[: cutoff_index + 1]
            cutoff_row = case_rows[cutoff_index]

        kept_rows.extend(kept)
        sample = case_rows[0] if case_rows else {}
        summaries.append(
            {
                "panel": sample.get("panel", ""),
                "case_key": key,
                "label": sample.get("label", ""),
                "fluid_solver": sample.get("fluid_solver", ""),
                "truncated": cutoff_row is not None,
                "kept_frame_rows": len(kept),
                "total_frame_rows": len(case_rows),
                "truncate_record_index": "" if cutoff_row is None else cutoff_row["record_index"],
                "truncate_sim_time": "" if cutoff_row is None else cutoff_row["sim_time"],
                "truncate_density_error_max": "" if cutoff_row is None else cutoff_row["density_error_max"],
                "truncate_density_error_rms": "" if cutoff_row is None else cutoff_row["density_error_rms"],
                "truncate_density_error_p95": "" if cutoff_row is None else cutoff_row["density_error_p95"],
                "truncate_reason": reason or "",
            }
        )

    return kept_rows, summaries


def write_truncation_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=TRUNCATION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in TRUNCATION_COLUMNS})


def series_by_case(rows: list[dict[str, Any]], metric: str) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["label"]), []).append(row)

    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for label, case_rows in grouped.items():
        ordered = sorted(case_rows, key=lambda item: (float(item["sim_time"]), int(item["record_index"])))
        xs = np.asarray([float(item["sim_time"]) for item in ordered], dtype=np.float64)
        ys = np.asarray([float(item[metric]) for item in ordered], dtype=np.float64)
        series[label] = (xs, ys)
    return series


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
    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for label, (xs, ys) in series_by_case(rows, metric).items():
        ax.plot(xs, ys, linewidth=1.8, label=label)
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.legend(fontsize=8)
    for fmt in formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=dpi)
    plt.close(fig)


def plot_box_height(
    *,
    rows: list[dict[str, Any]],
    output_base: Path,
    formats: list[str],
    dpi: int,
) -> None:
    box_rows = [row for row in rows if row.get("box_y") not in (None, "")]
    if not box_rows:
        return

    fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    grouped = series_by_case(box_rows, "box_y")
    for label, (xs, ys) in grouped.items():
        ax.plot(xs, ys, linewidth=1.8, label=label)
    ax.set_title("Rigid Body Height")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Center height (m)")
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.legend(fontsize=8)
    for fmt in formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=dpi)
    plt.close(fig)


def analyze_panel(
    panel_dir: Path,
    output_root: Path,
    include_initial_frame: bool,
    formats: list[str],
    dpi: int,
    truncate_blowup: bool,
    solver_filter: set[str] | None,
    blowup_threshold: float,
) -> None:
    panel_metadata = load_json(panel_dir / "panel_metadata.json")
    panel_name = str(panel_metadata.get("name", panel_dir.name))
    panel_output = output_root / panel_name

    full_rows: list[dict[str, Any]] = []
    for case_entry in panel_metadata["cases"]:
        full_rows.extend(
            compute_case_rows(
                panel_name=panel_name,
                case_dir=case_path(panel_dir, case_entry),
                case_entry=case_entry,
                include_initial_frame=include_initial_frame,
            )
        )

    rows, truncation_summary = truncate_rows_on_blowup(
        full_rows,
        enabled=truncate_blowup,
        solver_filter=solver_filter,
        threshold=blowup_threshold,
    )

    write_csv(panel_output / "density_metrics_full.csv", full_rows)
    write_csv(panel_output / "density_metrics.csv", rows)
    write_truncation_summary(panel_output / "truncation_summary.csv", truncation_summary)
    plot_metric(
        rows=rows,
        metric="density_error_max",
        ylabel="Max relative density error",
        title=f"{panel_name}: max density error",
        output_base=panel_output / "density_error_max",
        formats=formats,
        dpi=dpi,
    )
    plot_metric(
        rows=rows,
        metric="density_error_rms",
        ylabel="RMS relative density error",
        title=f"{panel_name}: RMS density error",
        output_base=panel_output / "density_error_rms",
        formats=formats,
        dpi=dpi,
    )
    plot_metric(
        rows=rows,
        metric="density_error_p95",
        ylabel="95th percentile relative density error",
        title=f"{panel_name}: p95 density error",
        output_base=panel_output / "density_error_p95",
        formats=formats,
        dpi=dpi,
    )
    plot_box_height(
        rows=rows,
        output_base=panel_output / "box_height",
        formats=formats,
        dpi=dpi,
    )

    truncated_cases = sum(1 for item in truncation_summary if item["truncated"])
    print(
        f"[Density Curves] {panel_name}: wrote {len(rows)} plotted frame rows "
        f"({len(full_rows)} full rows, {truncated_cases} truncated cases) to {panel_output}"
    )


def main() -> None:
    args = parse_args()
    panel_dirs = find_panel_dirs(args.cache_root, args.panel)
    if not panel_dirs:
        raise FileNotFoundError(f"No panel cache directories found under {args.cache_root}")

    solver_filter = parse_solver_filter(args.truncate_solvers)
    for panel_dir in panel_dirs:
        analyze_panel(
            panel_dir=panel_dir,
            output_root=args.output_root,
            include_initial_frame=args.include_initial_frame,
            formats=args.formats,
            dpi=args.dpi,
            truncate_blowup=not args.no_truncate_blowup,
            solver_filter=solver_filter,
            blowup_threshold=args.blowup_error_threshold,
        )


if __name__ == "__main__":
    main()
