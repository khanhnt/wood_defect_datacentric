#!/usr/bin/env python3
"""Generate read-only dataset statistics and audit reports."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent

if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from wood_defect_datacentric.datasets.adapters import load_manifest_dataset, resolve_repo_path
from wood_defect_datacentric.datasets.statistics import (
    render_markdown_report,
    write_markdown_report,
    write_stats_csv,
    write_verification_csv,
)


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_configured_datasets(config: dict[str, Any]):
    paths = config["paths"]
    datasets = config["datasets"]

    vsb_manifest = resolve_repo_path(paths["vsb_manifest_path"], PROJECT_ROOT)
    vn_manifest = resolve_repo_path(paths["vnwoodknot_manifest_path"], PROJECT_ROOT)

    vsb = load_manifest_dataset(
        dataset_key="vsb",
        manifest_path=vsb_manifest,
        expected_classes=datasets["vsb_curated"]["classes"],
        check_image_exists=True,
    )
    vn = load_manifest_dataset(
        dataset_key="vnwoodknot",
        manifest_path=vn_manifest,
        expected_classes=datasets["vnwoodknot"]["positive_classes"],
        negative_source_category=datasets["vnwoodknot"]["negative_class"],
        check_image_exists=True,
    )
    return vsb, vn


def main() -> None:
    config_path = PROJECT_ROOT / "configs" / "project.yaml"
    config = load_config(config_path)
    output_root = resolve_repo_path(config["paths"]["output_root"], PROJECT_ROOT)
    docs_root = resolve_repo_path(config["paths"].get("docs_root", PROJECT_ROOT / "docs"), PROJECT_ROOT)

    vsb, vn = load_configured_datasets(config)

    vsb_csv = output_root / "dataset_stats_vsb.csv"
    vn_csv = output_root / "dataset_stats_vnwoodknot.csv"
    verification_csv = output_root / "dataset_split_verification.csv"
    report_md = docs_root / "dataset_audit.md"

    write_stats_csv(vsb, vsb_csv)
    write_stats_csv(vn, vn_csv)
    write_verification_csv((vsb, vn), verification_csv)
    report = render_markdown_report(
        load_results=(vsb, vn),
        output_paths={
            "vsb_csv": vsb_csv,
            "vn_csv": vn_csv,
            "verification_csv": verification_csv,
        },
    )
    write_markdown_report(report, report_md)

    print(f"Wrote: {vsb_csv}")
    print(f"Wrote: {vn_csv}")
    print(f"Wrote: {verification_csv}")
    print(f"Wrote: {report_md}")


if __name__ == "__main__":
    main()

