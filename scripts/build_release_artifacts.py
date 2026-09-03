#!/usr/bin/env python3
"""Package final lightweight table artifacts from frozen analysis generations."""

from __future__ import annotations

import argparse
import hashlib
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

YOLO_ROOT_FILES = (
    "base_ap_reproduction.csv",
    "calibration_per_seed.csv",
    "calibration_summary.csv",
    "clean_fp_sweep_per_seed.csv",
    "clean_fp_sweep_summary.csv",
    "clean_max_confidence_cdf.csv",
    "clean_max_confidence_per_image.csv",
    "clean_max_confidence_per_seed.csv",
    "clean_max_confidence_summary.csv",
    "fair_metrics_per_seed.csv",
    "fair_metrics_summary.csv",
    "locked_test_operating_points_per_seed.csv",
    "locked_test_operating_points_summary.csv",
    "locked_test_sensitivity_summary.csv",
    "reliability_bins.csv",
    "validation_threshold_selection.csv",
)

FRCNN_STANDARD_FILES = (
    "checkpoint_registry.csv",
    "finalization_report.json",
    "per_seed_metrics.csv",
    "prediction_audit.csv",
    "summary.csv",
)

FRCNN_NEGATIVE_FILES = (
    "analysis_report.json",
    "epsilon_rank_audit.csv",
    "operational_stability_audit.json",
    "test_operating_metrics_per_seed.csv",
    "test_operating_summary.csv",
    "test_threshold_raw.csv",
    "validation_selected_thresholds.csv",
    "validation_threshold_raw.csv",
    "zero_fp_binding_audit.csv",
)

FRCNN_PROVENANCE_FILES = (
    "analysis_code_commit.txt",
    "job_manifest.json",
    "runtime_preflight.json",
    "runtime_preflight_training.json",
    "training_code_commit.txt",
    "verification_gate.csv",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yolo-analysis-dir",
        type=Path,
        default=PROJECT_ROOT / "revised" / "analysis" / "access_r1_g2",
        help="Frozen YOLOv8s analysis directory.",
    )
    parser.add_argument(
        "--fasterrcnn-generation-dir",
        type=Path,
        default=PROJECT_ROOT / "revised" / "generations" / "access_r1_g3_fasterrcnn",
        help="Frozen Faster R-CNN generation directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "results" / "tables",
        help="Destination for the lightweight public table package.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    yolo_src = args.yolo_analysis_dir.resolve()
    frcnn_src = args.fasterrcnn_generation_dir.resolve()
    output = args.output_root.resolve()

    require_directory(yolo_src, "YOLOv8s analysis")
    require_directory(frcnn_src, "Faster R-CNN generation")

    yolo_dst = output / "yolov8s"
    frcnn_dst = output / "fasterrcnn"
    copy_named(yolo_src, yolo_dst, YOLO_ROOT_FILES)
    copy_matching(yolo_src / "deprecated_impact", yolo_dst, ("*.csv",))
    copy_matching(yolo_src / "reviewer_offline", yolo_dst / "reviewer_audits", ("*.csv",))
    copy_matching(yolo_src / "latex", yolo_dst / "latex", ("*.tex",))

    analysis_src = frcnn_src / "fasterrcnn" / "analysis"
    negative_src = frcnn_src / "fasterrcnn" / "negative_aware"
    provenance_src = frcnn_src / "provenance"
    copy_named(analysis_src, frcnn_dst / "standard", FRCNN_STANDARD_FILES)
    copy_named(negative_src, frcnn_dst / "negative_aware", FRCNN_NEGATIVE_FILES)
    copy_named(provenance_src, frcnn_dst / "provenance", FRCNN_PROVENANCE_FILES)
    copy_file(frcnn_src / "fasterrcnn" / "run_log.csv", frcnn_dst / "provenance" / "run_log.csv")

    checksum_path = output / "SHA256SUMS"
    write_checksums(output, checksum_path)
    packaged = sum(1 for path in output.rglob("*") if path.is_file())
    print(f"Wrote public result package: {output}")
    print(f"Packaged files: {packaged}")
    print(f"Checksums: {checksum_path}")


def require_directory(path: Path, label: str) -> None:
    if not path.is_dir():
        raise SystemExit(f"Missing {label} directory: {path}")


def copy_named(source: Path, destination: Path, names: tuple[str, ...]) -> None:
    require_directory(source, "source artifact")
    for name in names:
        copy_file(source / name, destination / name)


def copy_matching(source: Path, destination: Path, patterns: tuple[str, ...]) -> None:
    require_directory(source, "source artifact")
    matches = sorted({path for pattern in patterns for path in source.glob(pattern)})
    if not matches:
        raise SystemExit(f"No matching artifacts in {source}: {patterns}")
    for path in matches:
        copy_file(path, destination / path.name)


def copy_file(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise SystemExit(f"Missing source artifact: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def write_checksums(root: Path, output: Path) -> None:
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path != output and "_deprecated" not in path.parts
    )
    lines = []
    for path in paths:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        lines.append(f"{digest}  {path.relative_to(root).as_posix()}")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
