#!/usr/bin/env python3
"""Verify configured dataset splits without modifying datasets."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent

if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from wood_defect_datacentric.datasets.statistics import verify_splits, write_verification_csv
from wood_defect_datacentric.scripts.dataset_stats import load_config, load_configured_datasets
from wood_defect_datacentric.datasets.adapters import resolve_repo_path


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "project.yaml"
    config = load_config(config_path)
    datasets = load_configured_datasets(config)
    output_root = resolve_repo_path(config["paths"]["output_root"], PROJECT_ROOT)
    output_path = output_root / "dataset_split_verification.csv"
    write_verification_csv(datasets, output_path)

    exit_code = 0
    for dataset in datasets:
        print(f"[{dataset.dataset_key}]")
        for row in verify_splits(dataset):
            print(f"{row['status']:>4} {row['check']} count={row['count']} - {row['message']}")
            if row["status"] == "fail":
                exit_code = 1
    print(f"Wrote: {output_path}")
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()

