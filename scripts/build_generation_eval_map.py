#!/usr/bin/env python3
"""Write explicit fair-evaluation YAML maps for one rebuilt dataset tree."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


VARIANTS = (
    ("baseline", None),
    ("p1_clahe", "P1_CLAHE_luminance"),
    ("p2_illumination", "P2_illumination_normalization"),
    ("p3_unsharp", "P3_mild_unsharp"),
    ("a1_crop", None),
    ("a2_colorjitter", None),
    ("p4_a4_combined", "P4_combined_safe"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuilt-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def dataset_yaml(root: Path, dataset: str, preprocessing: str | None) -> Path:
    if preprocessing:
        return root / "variants" / dataset / "preprocessing" / preprocessing / "dataset.yaml"
    return root / "canonical" / dataset / "dataset.yaml"


def write_map(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "variant", "data_yaml"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote: {path}")


def main() -> None:
    args = parse_args()
    root = args.rebuilt_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    fair_rows: list[dict[str, str]] = []
    for dataset in ("vnwoodknot", "vsb_rarefirst"):
        for variant, preprocessing in VARIANTS:
            path = dataset_yaml(root, dataset, preprocessing).resolve()
            if not path.exists():
                raise SystemExit(f"Missing fair-evaluation YAML: {path}")
            fair_rows.append({"dataset": dataset, "variant": variant, "data_yaml": str(path)})

    clean_rows: list[dict[str, str]] = []
    for variant, preprocessing in VARIANTS:
        path = dataset_yaml(root, "vsb_strict_clean", preprocessing).resolve()
        if not path.exists():
            raise SystemExit(f"Missing strict-clean YAML: {path}")
        clean_rows.append({"dataset": "vsb_strict_clean", "variant": variant, "data_yaml": str(path)})

    write_map(output / "fair_eval_dataset_map.csv", fair_rows)
    write_map(output / "vsb_strict_clean_eval_dataset_map.csv", clean_rows)


if __name__ == "__main__":
    main()
