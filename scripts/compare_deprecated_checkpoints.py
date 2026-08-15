#!/usr/bin/env python3
"""Quantify fair-evaluation changes from deprecated to corrected checkpoints."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics


METRICS = ("precision", "recall", "mAP50", "mAP50_95")
VARIANTS = {"a1_crop", "a2_colorjitter", "p4_a4_combined"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrected", type=Path, required=True)
    parser.add_argument("--deprecated", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read(path: Path) -> dict[tuple[str, str, int, str], dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            (row["dataset"], row["variant"], int(row["seed"]), row["split"]): row
            for row in csv.DictReader(handle)
            if row["variant"] in VARIANTS
        }


def write(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    corrected = read(args.corrected.expanduser().resolve())
    deprecated = read(args.deprecated.expanduser().resolve())
    keys = sorted(set(corrected) | set(deprecated))
    missing = [key for key in keys if key not in corrected or key not in deprecated]
    if missing:
        raise SystemExit(f"Corrected/deprecated fair rows do not align: {missing[:10]}")

    detail: list[dict[str, object]] = []
    for dataset, variant, seed, split in keys:
        row: dict[str, object] = {"dataset": dataset, "variant": variant, "seed": seed, "split": split}
        for metric in METRICS:
            new = float(corrected[(dataset, variant, seed, split)][metric])
            old = float(deprecated[(dataset, variant, seed, split)][metric])
            row[f"corrected_{metric}"] = new
            row[f"deprecated_{metric}"] = old
            row[f"delta_{metric}"] = new - old
        detail.append(row)

    summary: list[dict[str, object]] = []
    groups = sorted({(row["dataset"], row["variant"], row["split"]) for row in detail})
    for dataset, variant, split in groups:
        members = [row for row in detail if (row["dataset"], row["variant"], row["split"]) == (dataset, variant, split)]
        row = {"dataset": dataset, "variant": variant, "split": split, "n_seeds": len(members)}
        for metric in METRICS:
            for prefix in ("corrected", "deprecated", "delta"):
                values = [float(member[f"{prefix}_{metric}"]) for member in members]
                row[f"{prefix}_{metric}_mean"] = statistics.mean(values)
                row[f"{prefix}_{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(row)

    output = args.output_dir.expanduser().resolve()
    write(output / "deprecated_vs_corrected_per_seed.csv", detail)
    write(output / "deprecated_vs_corrected_summary.csv", summary)
    print("DEPRECATED CHECKPOINT IMPACT (corrected minus deprecated)")
    for row in summary:
        print(
            f"- {row['dataset']} {row['variant']} {row['split']}: "
            f"delta_mAP50={float(row['delta_mAP50_mean']):+.4f}, "
            f"delta_mAP50-95={float(row['delta_mAP50_95_mean']):+.4f}"
        )
    print(f"Wrote: {output}")


if __name__ == "__main__":
    main()
