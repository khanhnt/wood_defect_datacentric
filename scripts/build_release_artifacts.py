#!/usr/bin/env python3
"""Build lightweight public-release artifacts from local analysis outputs."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_RESULTS = PROJECT_ROOT / "results 3"
TABLES = PROJECT_ROOT / "results" / "tables"
FIGURES = PROJECT_ROOT / "results" / "figures"
DATA = PROJECT_ROOT / "data"


VARIANT_LABELS = {
    "baseline": "Baseline",
    "p1_clahe": "P1 CLAHE",
    "p2_illumination": "P2 illumination",
    "p3_unsharp": "P3 unsharp",
    "a1_crop": "A1 crop",
    "a2_colorjitter": "A2 colour jitter",
    "p4_a4_combined": "P4+A4 combined",
}


def main() -> None:
    if not SOURCE_RESULTS.exists():
        raise SystemExit(f"Missing source results folder: {SOURCE_RESULTS}")
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    build_tables()
    build_data_manifests()
    copy_figures()
    print(f"Wrote release tables to {TABLES}")
    print(f"Wrote release figures to {FIGURES}")


def build_tables() -> None:
    copy_file(
        SOURCE_RESULTS / "corrected_common_eval_fixed" / "test_best_summary.csv",
        TABLES / "vnwoodknot_fair_test_best_summary.csv",
    )
    copy_file(
        SOURCE_RESULTS / "corrected_common_eval_fixed" / "vsb" / "test_best_summary.csv",
        TABLES / "vsb_rarefirst_fair_test_best_summary.csv",
    )
    copy_file(
        SOURCE_RESULTS / "multiseed" / "all_variants_multiseed_summary.csv",
        TABLES / "standard_map_multiseed_summary.csv",
    )
    copy_file(
        SOURCE_RESULTS / "negative_aware_corrected" / "threshold_sweep" / "raw_data.csv",
        TABLES / "vnwoodknot_threshold_sweep_raw.csv",
    )
    copy_file(
        SOURCE_RESULTS / "negative_aware_corrected" / "threshold_sweep" / "summary_aggregated.csv",
        TABLES / "vnwoodknot_threshold_sweep_summary.csv",
    )
    copy_file(
        SOURCE_RESULTS / "negative_aware_corrected" / "bootstrap" / "bootstrap_summary_table.csv",
        TABLES / "vnwoodknot_fp_bootstrap_summary.csv",
    )
    copy_file(
        SOURCE_RESULTS / "calibration_step3" / "calibration_summary.csv",
        TABLES / "vnwoodknot_calibration_summary.csv",
    )
    copy_file(
        SOURCE_RESULTS / "calibration_step3" / "clean_max_confidence_summary.csv",
        TABLES / "vnwoodknot_clean_max_confidence_summary.csv",
    )
    copy_file(
        SOURCE_RESULTS / "vsb_negative_aware" / "clean_set_report.json",
        TABLES / "vsb_clean_set_report.json",
    )
    copy_file(
        SOURCE_RESULTS / "vsb_negative_aware" / "threshold_sweep" / "raw_data.csv",
        TABLES / "vsb_clean_threshold_sweep_raw.csv",
    )
    copy_file(
        SOURCE_RESULTS / "vsb_negative_aware" / "threshold_sweep" / "summary_aggregated.csv",
        TABLES / "vsb_clean_threshold_sweep_summary.csv",
    )
    copy_file(
        SOURCE_RESULTS / "vsb_negative_aware" / "bootstrap" / "bootstrap_summary_table.csv",
        TABLES / "vsb_clean_fp_bootstrap_summary.csv",
    )
    copy_file(
        SOURCE_RESULTS / "vsb_negative_aware" / "calibration" / "calibration_summary.csv",
        TABLES / "vsb_clean_calibration_summary.csv",
    )
    copy_file(
        SOURCE_RESULTS / "vsb_negative_aware" / "calibration" / "clean_max_confidence_summary.csv",
        TABLES / "vsb_clean_max_confidence_summary.csv",
    )
    copy_retained_json(
        SOURCE_RESULTS / "retained_metrics_audit.json",
        TABLES / "vnwoodknot_retained_metrics.json",
    )
    copy_retained_json(
        SOURCE_RESULTS / "vsb_negative_aware" / "retained_metrics.json",
        TABLES / "vsb_clean_retained_metrics.json",
    )
    retained_to_csv(TABLES / "vnwoodknot_retained_metrics.json", TABLES / "vnwoodknot_operational_selection.csv")
    retained_to_csv(TABLES / "vsb_clean_retained_metrics.json", TABLES / "vsb_clean_operational_selection.csv")
    sensitivity_to_csv(TABLES / "vnwoodknot_retained_metrics.json", TABLES / "vnwoodknot_sensitivity.csv")
    sensitivity_to_csv(TABLES / "vsb_clean_retained_metrics.json", TABLES / "vsb_clean_sensitivity.csv")


def build_data_manifests() -> None:
    sanitize_manifest(
        DATA / "processed" / "vnwoodknot_manifest.jsonl",
        DATA / "vnwoodknot_split" / "manifest.jsonl",
        dataset="vnwoodknot",
    )
    sanitize_manifest(
        DATA / "processed" / "large_scale_wood_surface_defects_manifest.jsonl",
        DATA / "vsb_rarefirst_split" / "source_manifest_sanitized.jsonl",
        dataset="vsb_source",
    )
    build_vsb_clean_manifest()
    build_vsb_test_manifest_from_predictions()


def copy_figures() -> None:
    figure_map = {
        SOURCE_RESULTS / "negative_aware" / "plots" / "detection_performance_vs_threshold.pdf": "detection_performance_vs_threshold.pdf",
        SOURCE_RESULTS / "negative_aware" / "plots" / "detection_performance_vs_threshold.png": "detection_performance_vs_threshold.png",
        SOURCE_RESULTS / "negative_aware" / "plots" / "false_positive_behavior_vs_threshold.pdf": "false_positive_behavior_vs_threshold.pdf",
        SOURCE_RESULTS / "negative_aware" / "plots" / "false_positive_behavior_vs_threshold.png": "false_positive_behavior_vs_threshold.png",
        SOURCE_RESULTS / "negative_aware" / "plots" / "operational_selection_recall_fp_tradeoff.pdf": "operational_selection_recall_fp_tradeoff.pdf",
        SOURCE_RESULTS / "negative_aware" / "plots" / "operational_selection_recall_fp_tradeoff.png": "operational_selection_recall_fp_tradeoff.png",
        SOURCE_RESULTS / "qualitative_corrected" / "dataset_samples.pdf": "dataset_samples.pdf",
        SOURCE_RESULTS / "qualitative_corrected" / "dataset_samples.png": "dataset_samples.png",
        SOURCE_RESULTS / "qualitative_corrected" / "detection_scenarios.pdf": "detection_scenarios.pdf",
        SOURCE_RESULTS / "qualitative_corrected" / "detection_scenarios.png": "detection_scenarios.png",
        SOURCE_RESULTS / "qualitative_corrected" / "scenario_1.pdf": "scenario_1.pdf",
        SOURCE_RESULTS / "qualitative_corrected" / "scenario_1.png": "scenario_1.png",
        SOURCE_RESULTS / "qualitative_corrected" / "scenario_2.pdf": "scenario_2.pdf",
        SOURCE_RESULTS / "qualitative_corrected" / "scenario_2.png": "scenario_2.png",
        SOURCE_RESULTS / "qualitative_corrected" / "scenario_3.pdf": "scenario_3.pdf",
        SOURCE_RESULTS / "qualitative_corrected" / "scenario_3.png": "scenario_3.png",
        SOURCE_RESULTS / "qualitative_corrected" / "scenario_4.pdf": "scenario_4.pdf",
        SOURCE_RESULTS / "qualitative_corrected" / "scenario_4.png": "scenario_4.png",
        SOURCE_RESULTS / "calibration_step3" / "reliability_curve.pdf": "reliability_curve.pdf",
        SOURCE_RESULTS / "calibration_step3" / "reliability_curve.png": "reliability_curve.png",
        SOURCE_RESULTS / "calibration_step3" / "clean_max_confidence_cdf.pdf": "clean_max_confidence_cdf.pdf",
        SOURCE_RESULTS / "calibration_step3" / "clean_max_confidence_cdf.png": "clean_max_confidence_cdf.png",
        SOURCE_RESULTS / "calibration_step3" / "reliability_curve.pdf": "vnwoodknot_reliability_curve.pdf",
        SOURCE_RESULTS / "calibration_step3" / "reliability_curve.png": "vnwoodknot_reliability_curve.png",
        SOURCE_RESULTS / "calibration_step3" / "clean_max_confidence_cdf.pdf": "vnwoodknot_clean_max_confidence_cdf.pdf",
        SOURCE_RESULTS / "calibration_step3" / "clean_max_confidence_cdf.png": "vnwoodknot_clean_max_confidence_cdf.png",
        SOURCE_RESULTS / "vsb_negative_aware" / "calibration" / "reliability_curve.pdf": "vsb_clean_reliability_curve.pdf",
        SOURCE_RESULTS / "vsb_negative_aware" / "calibration" / "reliability_curve.png": "vsb_clean_reliability_curve.png",
        SOURCE_RESULTS / "vsb_negative_aware" / "calibration" / "clean_max_confidence_cdf.pdf": "vsb_clean_max_confidence_cdf.pdf",
        SOURCE_RESULTS / "vsb_negative_aware" / "calibration" / "clean_max_confidence_cdf.png": "vsb_clean_max_confidence_cdf.png",
    }
    for src, name in figure_map.items():
        if src.exists():
            copy_file(src, FIGURES / name)
            if src.suffix == ".pdf":
                copy_file(src, PROJECT_ROOT / "figures" / name)


def retained_to_csv(src_json: Path, out_csv: Path) -> None:
    data = json.loads(src_json.read_text())
    rows = []
    for variant, payload in data["variants"].items():
        summary = payload["summary_at_tau_star"]
        rows.append(
            {
                "variant": variant,
                "variant_label": payload.get("variant_label", VARIANT_LABELS.get(variant, variant)),
                "tau_star": payload["tau_star"],
                "precision_mean": summary["precision_at_tau_mean"],
                "precision_std": summary["precision_at_tau_std"],
                "retained_recall_mean": summary["retained_recall_mean"],
                "retained_recall_std": summary["retained_recall_std"],
                "retained_AP50_mean": summary["retained_AP50_mean"],
                "retained_AP50_std": summary["retained_AP50_std"],
            }
        )
    write_csv(rows, out_csv)


def sensitivity_to_csv(src_json: Path, out_csv: Path) -> None:
    data = json.loads(src_json.read_text())
    rows = []
    for variant, payload in data["variants"].items():
        for epsilon, summary in payload["epsilon_operating_points"].items():
            rows.append(
                {
                    "variant": variant,
                    "variant_label": payload.get("variant_label", VARIANT_LABELS.get(variant, variant)),
                    "epsilon": epsilon,
                    "tau": summary["threshold"],
                    "retained_AP50_mean": summary["retained_AP50_mean"],
                    "retained_AP50_std": summary["retained_AP50_std"],
                    "retained_recall_mean": summary["retained_recall_mean"],
                    "retained_recall_std": summary["retained_recall_std"],
                }
            )
    write_csv(rows, out_csv)


def sanitize_manifest(src: Path, dst: Path, *, dataset: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            row = json.loads(line)
            image_id = str(row.get("image_id") or "")
            if dataset == "vnwoodknot":
                row["image_path"] = f"{image_id}.jpg"
                row["annotation_path"] = f"{image_id}.txt" if row.get("annotations") else None
            else:
                stem = Path(image_id).name
                row["image_path"] = f"{image_id}.bmp"
                row["annotation_path"] = f"Bouding Boxes/{stem}_anno.txt"
                row["semantic_map_path"] = f"Semantic Maps/{stem}_segm.bmp"
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_vsb_clean_manifest() -> None:
    ids_path = PROJECT_ROOT / "configs" / "datasets" / "vsb_clean_source_ids.txt"
    out_dir = DATA / "vsb_clean_manifest"
    out_dir.mkdir(parents=True, exist_ok=True)
    copy_file(ids_path, out_dir / "source_ids.txt")
    rows = []
    for source_id in ids_path.read_text().splitlines():
        if not source_id.strip():
            continue
        for x in (0, 896, 1776):
            tile_id = f"{source_id}__x{x:04d}_y0000"
            rows.append(
                {
                    "source_id": source_id,
                    "tile_id": tile_id,
                    "split": "test",
                    "image_path": f"clean/{tile_id}.bmp",
                    "x": x,
                    "y": 0,
                    "width": 1024,
                    "height": 1024,
                    "annotations": "[]",
                }
            )
    write_csv(rows, out_dir / "clean_tile_manifest.csv")


def build_vsb_test_manifest_from_predictions() -> None:
    src = SOURCE_RESULTS / "vsb_negative_aware" / "predictions" / "baseline_seed42_predictions.json"
    if not src.exists():
        return
    payload = json.loads(src.read_text())
    rows = []
    for image in payload["images"]:
        path = str(image.get("image_path", ""))
        if "/vsb_clean" in path:
            continue
        rows.append(
            {
                "split": "test",
                "image": image.get("image"),
                "canonical_id": image.get("canonical_id"),
                "image_path": Path(path).name,
                "is_empty": bool(image.get("is_knot_free", False)),
                "num_gt_boxes": len(image.get("gt_boxes", [])),
            }
        )
    write_csv(rows, DATA / "vsb_rarefirst_split" / "test_tile_manifest.csv")
    summary = {
        "note": "The full train/val/test rare-first manifest is reconstructed with data/prepare_vsb.py from the source manifest. This file records the held-out test tiles used by released prediction JSONs.",
        "test_tiles": len(rows),
        "test_positive_tiles": sum(1 for row in rows if not row["is_empty"]),
        "test_empty_tiles": sum(1 for row in rows if row["is_empty"]),
        "known_full_split_counts": {"train": 7679, "val": 977, "test": 972},
    }
    write_json(summary, DATA / "vsb_rarefirst_split" / "split_summary.json")


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"WARNING missing source artifact: {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_retained_json(src: Path, dst: Path) -> None:
    if not src.exists():
        print(f"WARNING missing source artifact: {src}")
        return
    payload = json.loads(src.read_text())
    payload.pop("predictions_dir", None)
    write_json(payload, dst)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
