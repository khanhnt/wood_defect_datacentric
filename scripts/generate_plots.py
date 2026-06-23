#!/usr/bin/env python3
"""Generate publication-ready Phase 3 negative-aware evaluation plots."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any


VARIANT_LABELS = {
    "baseline": "Baseline",
    "p2_illumination": "P2 illumination",
    "a1_crop": "A1 crop",
    "a2_colorjitter": "A2 color jitter",
    "p4_a4_combined": "P4+A4 combined",
}
VARIANT_ORDER = {variant: index for index, variant in enumerate(VARIANT_LABELS)}
METRIC_PANELS = (
    ("ap50", "AP50"),
    ("precision", "Precision"),
    ("recall", "Recall"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("results/negative_aware"))
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--annotate-operating-points",
        action="store_true",
        help="Annotate zero-FP operating points with variant names and recall values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else data_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = data_dir / "threshold_sweep" / "summary_aggregated.csv"
    sweet_spot_path = data_dir / "operational_sweet_spots.csv"
    if not summary_path.exists():
        raise SystemExit(f"Missing threshold summary CSV: {summary_path}")

    summary_rows = read_csv(summary_path)
    sweet_spots = read_csv(sweet_spot_path) if sweet_spot_path.exists() else []

    configure_matplotlib()
    paths = []
    paths.extend(plot_detection_performance(summary_rows, output_dir, dpi=args.dpi))
    paths.extend(plot_false_positive_behavior(summary_rows, output_dir, dpi=args.dpi))
    paths.extend(
        plot_operational_selection(
            summary_rows,
            sweet_spots,
            output_dir,
            dpi=args.dpi,
            annotate_operating_points=args.annotate_operating_points,
        )
    )
    for path in paths:
        print(f"Wrote: {path}")


def configure_matplotlib() -> None:
    import os
    import tempfile

    cache_dir = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "wood_defect_datacentric_mpl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def plot_detection_performance(rows: list[dict[str, str]], output_dir: Path, *, dpi: int) -> list[Path]:
    import matplotlib.pyplot as plt

    by_variant = group_by_variant(rows)
    colors = variant_colors()
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.6), sharex=True)
    for axis, (metric, title) in zip(axes, METRIC_PANELS):
        for variant, variant_rows in by_variant.items():
            x = [parse_float(row["threshold"]) for row in variant_rows]
            mean = [parse_float(row[f"{metric}_mean"]) for row in variant_rows]
            std = [parse_float(row[f"{metric}_std"]) for row in variant_rows]
            lower = [max(m - s, 0.0) for m, s in zip(mean, std)]
            upper = [min(m + s, 1.0) for m, s in zip(mean, std)]
            axis.plot(x, mean, marker="o", markersize=2.5, linewidth=1.4, color=colors[variant], label=VARIANT_LABELS.get(variant, variant))
            axis.fill_between(x, lower, upper, color=colors[variant], alpha=0.13, linewidth=0)
        axis.set_title(title)
        axis.set_xlabel("Confidence threshold")
        axis.set_ylim(0.0, 1.02)
        axis.set_xlim(0.05, 0.95)
    axes[0].set_ylabel("Metric value")
    axes[-1].legend(loc="lower left", bbox_to_anchor=(1.02, 0.0), frameon=False)
    fig.tight_layout()
    return save_figure(fig, output_dir / "detection_performance_vs_threshold", dpi=dpi)


def plot_false_positive_behavior(rows: list[dict[str, str]], output_dir: Path, *, dpi: int) -> list[Path]:
    import matplotlib.pyplot as plt

    by_variant = group_by_variant(rows)
    colors = variant_colors()
    fig, axis = plt.subplots(figsize=(3.5, 2.7))
    for variant, variant_rows in by_variant.items():
        x = [parse_float(row["threshold"]) for row in variant_rows]
        y = [parse_float(row["fp_image_rate_mean"]) for row in variant_rows]
        lower = [parse_float(row.get("fp_image_rate_ci_lower", "nan")) for row in variant_rows]
        upper = [parse_float(row.get("fp_image_rate_ci_upper", "nan")) for row in variant_rows]
        axis.plot(x, y, marker="o", markersize=2.8, linewidth=1.5, color=colors[variant], label=VARIANT_LABELS.get(variant, variant))
        if all(value == value for value in lower + upper):
            axis.fill_between(x, lower, upper, color=colors[variant], alpha=0.15, linewidth=0)
    axis.set_xlabel("Confidence threshold")
    axis.set_ylabel("FP image rate on knot-free")
    axis.set_ylim(bottom=0.0)
    axis.set_xlim(0.05, 0.95)
    axis.legend(frameon=False)
    fig.tight_layout()
    return save_figure(fig, output_dir / "false_positive_behavior_vs_threshold", dpi=dpi)


def plot_operational_selection(
    rows: list[dict[str, str]],
    sweet_spots: list[dict[str, str]],
    output_dir: Path,
    *,
    dpi: int,
    annotate_operating_points: bool = False,
) -> list[Path]:
    import matplotlib.pyplot as plt

    by_variant = group_by_variant(rows)
    sweet_by_variant = {row["variant"]: row for row in sweet_spots}
    colors = variant_colors()
    fig, axis_recall = plt.subplots(figsize=(7.2, 3.25))
    axis_fp = axis_recall.twinx()

    recall_handles = []
    for variant, variant_rows in by_variant.items():
        color = colors[variant]
        x = [parse_float(row["threshold"]) for row in variant_rows]
        recall = [parse_float(row["recall_mean"]) for row in variant_rows]
        fp_rate = [parse_float(row["fp_image_rate_mean"]) for row in variant_rows]
        recall_line = axis_recall.plot(x, recall, color=color, linewidth=1.6, marker="o", markersize=2.5, label=VARIANT_LABELS.get(variant, variant))[0]
        axis_fp.plot(x, fp_rate, color=color, linewidth=1.2, linestyle="--", alpha=0.85)
        recall_handles.append(recall_line)

        sweet = sweet_by_variant.get(variant)
        if sweet and sweet.get("zero_fp_threshold"):
            threshold = parse_float(sweet["zero_fp_threshold"])
            recall_value = parse_float(sweet["recall_at_that_threshold"])
            axis_recall.scatter([threshold], [recall_value], s=38, color=color, edgecolor="black", linewidth=0.5, zorder=5)
            if annotate_operating_points:
                axis_recall.annotate(
                    f"{VARIANT_LABELS.get(variant, variant)}\nR={recall_value:.2f}",
                    xy=(threshold, recall_value),
                    xytext=(4, 5),
                    textcoords="offset points",
                    fontsize=7,
                    color=color,
                )

    axis_recall.set_xlabel("Confidence threshold")
    axis_recall.set_ylabel("Recall")
    axis_fp.set_ylabel("FP image rate on knot-free")
    axis_recall.set_ylim(0.0, 1.02)
    axis_fp.set_ylim(bottom=0.0)
    axis_recall.set_xlim(0.05, 0.95)
    axis_recall.legend(
        handles=recall_handles,
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        edgecolor="0.85",
        title="Variant",
    )
    fig.text(
        0.08,
        0.06,
        "Note: solid lines = recall; dashed lines = FP image rate; black-edged markers = zero-FP operating point.",
        ha="left",
        va="bottom",
        fontsize=7,
        color="0.25",
    )
    fig.subplots_adjust(left=0.08, right=0.91, bottom=0.25, top=0.95)
    return save_figure(fig, output_dir / "operational_selection_recall_fp_tradeoff", dpi=dpi)


def group_by_variant(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["variant"]].append(row)
    return {
        variant: sorted(grouped[variant], key=lambda row: parse_float(row["threshold"]))
        for variant in sorted(grouped, key=variant_sort_key)
    }


def variant_colors() -> dict[str, Any]:
    import matplotlib.pyplot as plt

    cmap = plt.get_cmap("tab10")
    return {variant: cmap(index) for index, variant in enumerate(sorted(VARIANT_LABELS, key=variant_sort_key))}


def save_figure(fig: Any, stem: Path, *, dpi: int) -> list[Path]:
    paths = [stem.with_suffix(".png"), stem.with_suffix(".pdf")]
    for path in paths:
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return paths


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def parse_float(value: Any) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return float("nan")


def variant_sort_key(variant: str) -> int:
    return VARIANT_ORDER.get(str(variant), 99)


if __name__ == "__main__":
    main()
