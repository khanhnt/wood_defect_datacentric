#!/usr/bin/env python3
"""Plot retained AP50 against tolerated clean-image false-positive rate."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUTS = [
    ("VNWoodKnot", PROJECT_ROOT / "results" / "tables" / "vnwoodknot_sensitivity.csv"),
    ("VSB clean wood", PROJECT_ROOT / "results" / "tables" / "vsb_clean_sensitivity.csv"),
]
VARIANT_ORDER = ["baseline", "p2_illumination", "a1_crop", "a2_colorjitter", "p4_a4_combined"]
VARIANT_LABELS = {
    "baseline": "Baseline",
    "p1_clahe": "P1 CLAHE",
    "p2_illumination": "P2 illumination",
    "p3_unsharp": "P3 unsharp",
    "a1_crop": "A1 crop",
    "a2_colorjitter": "A2 colour jitter",
    "p4_a4_combined": "P4+A4 combined",
}
COLORS = {
    "baseline": "#4C78A8",
    "p1_clahe": "#9C755F",
    "p2_illumination": "#59A14F",
    "p3_unsharp": "#B07AA1",
    "a1_crop": "#E15759",
    "a2_colorjitter": "#F28E2B",
    "p4_a4_combined": "#76B7B2",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vnwoodknot", type=Path, default=DEFAULT_INPUTS[0][1])
    parser.add_argument("--vsb-clean", type=Path, default=DEFAULT_INPUTS[1][1])
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "figures")
    args = parser.parse_args()

    datasets = [
        ("VNWoodKnot", read_rows(args.vnwoodknot)),
        ("VSB clean wood", read_rows(args.vsb_clean)),
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    draw(datasets, args.output_dir / "ap50_vs_tolerance_vnwk_vsb.pdf")
    draw(datasets, args.output_dir / "ap50_vs_tolerance_vnwk_vsb.png")
    print(f"Wrote: {args.output_dir / 'ap50_vs_tolerance_vnwk_vsb.pdf'}")
    print(f"Wrote: {args.output_dir / 'ap50_vs_tolerance_vnwk_vsb.png'}")


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing sensitivity CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def draw(datasets: list[tuple[str, list[dict[str, str]]]], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)
    for ax, (title, rows) in zip(axes, datasets):
        variants = ordered_variants(rows)
        for variant in variants:
            points = sorted(
                [row for row in rows if row["variant"] == variant],
                key=lambda row: float(row["epsilon"]),
            )
            x = [100.0 * float(row["epsilon"]) for row in points]
            y = [float(row["retained_AP50_mean"]) for row in points]
            ax.plot(
                x,
                y,
                marker="o",
                linewidth=1.6,
                markersize=4,
                color=COLORS.get(variant),
                label=VARIANT_LABELS.get(variant, variant),
            )
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xlabel("Tolerated clean-image FP rate (%)", fontsize=9)
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.tick_params(labelsize=8)
        ax.set_xlim(-0.2, 5.2)
        ax.set_ylim(bottom=0.0)
    axes[0].set_ylabel("Retained AP50", fontsize=9)
    axes[1].legend(loc="lower right", fontsize=7, frameon=True, framealpha=0.92)
    fig.tight_layout(w_pad=1.2)
    save_kwargs = {"bbox_inches": "tight"}
    if output.suffix.lower() == ".png":
        save_kwargs["dpi"] = 300
    fig.savefig(output, **save_kwargs)
    mirror = PROJECT_ROOT / "results" / "figures" / output.name
    mirror.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(mirror, **save_kwargs)
    plt.close(fig)


def ordered_variants(rows: list[dict[str, str]]) -> list[str]:
    present = {row["variant"] for row in rows}
    ordered = [variant for variant in VARIANT_ORDER if variant in present]
    ordered.extend(sorted(present - set(ordered)))
    return ordered


if __name__ == "__main__":
    main()
