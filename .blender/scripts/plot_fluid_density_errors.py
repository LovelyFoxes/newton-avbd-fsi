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

CASE_STYLES = {
    ("pbf", 8, 4): {"color": "#F28E2B"},
    ("pbf", 8, 8): {"color": "#D62728"},
    ("ipbf", 4, 4): {"color": "#56B4E9"},
    ("ipbf", 8, 4): {"color": "#0072B2"},
    ("ipbf", 8, 8): {"color": "#004E64"},
}

SOLVER_FALLBACK_COLORS = {
    "pbf": "#C44E52",
    "ipbf": "#1F77B4",
}

PANEL_DISPLAY_TITLES = {
    "fluid_quasi2d_density_panel": "Quasi-2D density",
    "fluid_quasi2d_box_float_panel": "Float box",
    "fluid_quasi2d_box_sink_panel": "Sink box",
}

PANEL_FILE_PREFIXES = {
    "fluid_quasi2d_density_panel": "quasi2d_density",
    "fluid_quasi2d_box_float_panel": "box_float",
    "fluid_quasi2d_box_sink_panel": "box_sink",
}

TITLE_FONT_SIZE = 18.0
AXIS_LABEL_FONT_SIZE = 16.0
TICK_LABEL_FONT_SIZE = 13.0


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
        help="Do not detect blow-up frames; full curves will determine the plot y-axis.",
    )
    parser.add_argument(
        "--hide-after-blowup",
        action="store_true",
        help="Hide frames after the first blow-up frame. By default full curves are drawn but y-limits ignore exploded tails.",
    )
    parser.add_argument(
        "--blowup-error-threshold",
        type=float,
        default=3.0,
        help="Truncate after the first frame whose max relative density error exceeds this threshold.",
    )
    parser.add_argument(
        "--axis-error-threshold",
        type=float,
        default=None,
        help=(
            "Optional extra y-axis filter. If set, only frames below this max relative density error determine "
            "plot y-limits. By default only detected blow-up tails are excluded."
        ),
    )
    parser.add_argument(
        "--axis-percentile",
        type=float,
        default=99.5,
        help="Percentile of in-range samples used for plot y-limits.",
    )
    parser.add_argument("--legend-font-size", type=float, default=13.0, help="Legend font size for plots.")
    parser.add_argument("--line-width", type=float, default=2.6, help="Line width for plotted curves.")
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


def safe_file_stem(value: str) -> str:
    chars = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        else:
            chars.append("_")
    stem = "".join(chars).strip("_")
    while "__" in stem:
        stem = stem.replace("__", "_")
    return stem or "panel"


def panel_file_prefix(panel_name: str) -> str:
    return PANEL_FILE_PREFIXES.get(panel_name, safe_file_stem(panel_name))


def panel_display_title(panel_name: str) -> str:
    return PANEL_DISPLAY_TITLES.get(panel_name, panel_name.replace("_", " "))


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
        case_rows = sorted(grouped[key], key=lambda item: (int(item["frame_index"]), int(item["record_index"])))
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
        ordered = sorted(case_rows, key=lambda item: (int(item["frame_index"]), int(item["record_index"])))
        xs = np.asarray([float(item["frame_index"]) for item in ordered], dtype=np.float64)
        ys = np.asarray([float(item[metric]) for item in ordered], dtype=np.float64)
        series[label] = (xs, ys)
    return series


def case_style(case_rows: list[dict[str, Any]]) -> dict[str, Any]:
    sample = case_rows[0]
    solver = str(sample.get("fluid_solver", "")).lower()
    iterations = int(sample.get("iterations", 0))
    sim_substeps = int(sample.get("sim_substeps", 0))
    style = dict(CASE_STYLES.get((solver, iterations, sim_substeps), {}))
    style.setdefault("color", SOLVER_FALLBACK_COLORS.get(solver, "#4D4D4D"))
    return style


def grouped_case_rows(rows: list[dict[str, Any]]) -> list[tuple[str, list[dict[str, Any]]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    order: list[str] = []
    for row in rows:
        label = str(row["label"])
        if label not in grouped:
            order.append(label)
            grouped[label] = []
        grouped[label].append(row)
    return [(label, grouped[label]) for label in order]


def case_series(case_rows: list[dict[str, Any]], metric: str) -> tuple[np.ndarray, np.ndarray]:
    ordered = sorted(case_rows, key=lambda item: (int(item["frame_index"]), int(item["record_index"])))
    xs = np.asarray([float(item["frame_index"]) for item in ordered], dtype=np.float64)
    ys = np.asarray([float(item[metric]) for item in ordered], dtype=np.float64)
    return xs, ys


def plot_metric(
    *,
    rows: list[dict[str, Any]],
    axis_rows: list[dict[str, Any]],
    truncation_summary: list[dict[str, Any]],
    metric: str,
    ylabel: str,
    title: str,
    output_base: Path,
    formats: list[str],
    dpi: int,
    axis_error_threshold: float | None,
    axis_percentile: float,
    legend_font_size: float,
    line_width: float,
) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    for label, case_rows in grouped_case_rows(rows):
        xs, ys = case_series(case_rows, metric)
        ax.plot(xs, ys, linewidth=line_width, label=label, **case_style(case_rows))

    has_blowup = any(bool(item.get("truncated")) for item in truncation_summary)
    axis_candidates = [row for row in axis_rows if np.isfinite(float(row[metric]))]
    if has_blowup and axis_error_threshold is not None:
        filtered = [row for row in axis_candidates if float(row["density_error_max"]) <= axis_error_threshold]
        if filtered:
            axis_candidates = filtered
    if not axis_candidates:
        axis_candidates = [row for row in axis_rows if np.isfinite(float(row[metric]))]
    axis_values = np.asarray([float(row[metric]) for row in axis_candidates], dtype=np.float64)
    if axis_values.size:
        percentile = min(max(axis_percentile, 0.0), 100.0)
        y_max = float(np.percentile(axis_values, percentile))
        if y_max > 0.0:
            ax.set_ylim(bottom=0.0, top=y_max * 1.18)

    ax.set_title(title, fontsize=TITLE_FONT_SIZE, pad=10)
    ax.set_xlabel("Frame", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel(ylabel, fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.legend(fontsize=legend_font_size)
    for fmt in formats:
        fig.savefig(output_base.with_suffix(f".{fmt}"), dpi=dpi)
    plt.close(fig)


def plot_box_height(
    *,
    rows: list[dict[str, Any]],
    axis_rows: list[dict[str, Any]],
    truncation_summary: list[dict[str, Any]],
    output_base: Path,
    formats: list[str],
    dpi: int,
    axis_error_threshold: float | None,
    legend_font_size: float,
    line_width: float,
) -> None:
    box_rows = [row for row in rows if row.get("box_y") not in (None, "")]
    if not box_rows:
        return

    fig, ax = plt.subplots(figsize=(7.6, 4.6), constrained_layout=True)
    for label, case_rows in grouped_case_rows(box_rows):
        xs, ys = case_series(case_rows, "box_y")
        ax.plot(xs, ys, linewidth=line_width, label=label, **case_style(case_rows))

    has_blowup = any(bool(item.get("truncated")) for item in truncation_summary)
    axis_box_rows = [row for row in axis_rows if row.get("box_y") not in (None, "")]
    if has_blowup and axis_error_threshold is not None:
        filtered = [row for row in axis_box_rows if float(row["density_error_max"]) <= axis_error_threshold]
        if filtered:
            axis_box_rows = filtered
    if not axis_box_rows:
        axis_box_rows = [row for row in axis_rows if row.get("box_y") not in (None, "")]
    axis_values = np.asarray([float(row["box_y"]) for row in axis_box_rows], dtype=np.float64)
    if axis_values.size:
        y_min = float(np.min(axis_values))
        y_max = float(np.max(axis_values))
        padding = max((y_max - y_min) * 0.08, 1.0e-3)
        ax.set_ylim(y_min - padding, y_max + padding)

    ax.set_title("Rigid body height", fontsize=TITLE_FONT_SIZE, pad=10)
    ax.set_xlabel("Frame", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.set_ylabel("Center height (m)", fontsize=AXIS_LABEL_FONT_SIZE)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FONT_SIZE)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
    ax.grid(True, linewidth=0.35, alpha=0.35)
    ax.legend(fontsize=legend_font_size)
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
    hide_after_blowup: bool,
    solver_filter: set[str] | None,
    blowup_threshold: float,
    axis_error_threshold: float | None,
    axis_percentile: float,
    legend_font_size: float,
    line_width: float,
) -> None:
    panel_metadata = load_json(panel_dir / "panel_metadata.json")
    panel_name = str(panel_metadata.get("name", panel_dir.name))
    file_prefix = panel_file_prefix(panel_name)
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

    write_csv(panel_output / f"{file_prefix}_density_metrics_full.csv", full_rows)
    write_csv(panel_output / f"{file_prefix}_density_metrics.csv", rows)
    write_truncation_summary(panel_output / f"{file_prefix}_truncation_summary.csv", truncation_summary)

    plot_rows = rows if hide_after_blowup else full_rows
    axis_rows = rows if truncate_blowup else full_rows
    plot_metric(
        rows=plot_rows,
        axis_rows=axis_rows,
        truncation_summary=truncation_summary,
        metric="density_error_max",
        ylabel="Max relative density error",
        title=f"{panel_display_title(panel_name)}: max density error",
        output_base=panel_output / f"{file_prefix}_density_error_max",
        formats=formats,
        dpi=dpi,
        axis_error_threshold=axis_error_threshold,
        axis_percentile=axis_percentile,
        legend_font_size=legend_font_size,
        line_width=line_width,
    )
    plot_metric(
        rows=plot_rows,
        axis_rows=axis_rows,
        truncation_summary=truncation_summary,
        metric="density_error_rms",
        ylabel="RMS relative density error",
        title=f"{panel_display_title(panel_name)}: RMS density error",
        output_base=panel_output / f"{file_prefix}_density_error_rms",
        formats=formats,
        dpi=dpi,
        axis_error_threshold=axis_error_threshold,
        axis_percentile=axis_percentile,
        legend_font_size=legend_font_size,
        line_width=line_width,
    )
    plot_metric(
        rows=plot_rows,
        axis_rows=axis_rows,
        truncation_summary=truncation_summary,
        metric="density_error_p95",
        ylabel="95th percentile relative density error",
        title=f"{panel_display_title(panel_name)}: p95 density error",
        output_base=panel_output / f"{file_prefix}_density_error_p95",
        formats=formats,
        dpi=dpi,
        axis_error_threshold=axis_error_threshold,
        axis_percentile=axis_percentile,
        legend_font_size=legend_font_size,
        line_width=line_width,
    )
    plot_box_height(
        rows=plot_rows,
        axis_rows=axis_rows,
        truncation_summary=truncation_summary,
        output_base=panel_output / f"{file_prefix}_box_height",
        formats=formats,
        dpi=dpi,
        axis_error_threshold=axis_error_threshold,
        legend_font_size=legend_font_size,
        line_width=line_width,
    )

    truncated_cases = sum(1 for item in truncation_summary if item["truncated"])
    print(
        f"[Density Curves] {panel_name}: wrote {len(full_rows if not hide_after_blowup else rows)} plotted frame rows "
        f"({len(full_rows)} full rows, {truncated_cases} blow-up markers) to {panel_output}"
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
            hide_after_blowup=args.hide_after_blowup,
            solver_filter=solver_filter,
            blowup_threshold=args.blowup_error_threshold,
            axis_error_threshold=args.axis_error_threshold,
            axis_percentile=args.axis_percentile,
            legend_font_size=args.legend_font_size,
            line_width=args.line_width,
        )


if __name__ == "__main__":
    main()
