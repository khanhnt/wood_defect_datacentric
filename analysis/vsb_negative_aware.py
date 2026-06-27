#!/usr/bin/env python3
"""Negative-aware evaluation on VSB clean-wood tiles.

The script builds a clean VSB test set from zero-annotation records, exports
low-confidence predictions for the trained VSB rare-first checkpoints, and
reuses the threshold/AP and calibration utilities used by the VNWoodKnot
analysis.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
from typing import Any, Iterable

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from scripts import calibration_analysis as cal  # noqa: E402
from scripts import evaluate_corrected_common as common_eval  # noqa: E402
from scripts import materialize_yolo_from_manifest as mat_yolo  # noqa: E402
from scripts import threshold_analysis as th  # noqa: E402
from scripts import threshold_sweep_inference as infer  # noqa: E402
from wood_defect_datacentric.datasets.adapters import normalize_split, read_jsonl  # noqa: E402


VSB_CLASSES = ("live_knot", "dead_knot", "resin", "knot_with_crack", "crack", "marrow", "knot_missing")
VARIANTS = ("baseline", "p1_clahe", "p2_illumination", "p3_unsharp", "a1_crop", "a2_colorjitter", "p4_a4_combined")
SEEDS = (42, 43, 44)
EPSILONS = (0.0, 0.01, 0.02, 0.05)
LABELS = {
    "baseline": "Baseline",
    "p1_clahe": "P1 CLAHE",
    "p2_illumination": "P2 illumination",
    "p3_unsharp": "P3 unsharp",
    "a1_crop": "A1 crop",
    "a2_colorjitter": "A2 colour jitter",
    "p4_a4_combined": "P4+A4 combined",
}
PREPROCESSING = {
    "p1_clahe": "P1_CLAHE_luminance",
    "p2_illumination": "P2_illumination_normalization",
    "p3_unsharp": "P3_mild_unsharp",
    "p4_a4_combined": "P4_combined_safe",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SOURCE_TILE_RE = re.compile(r"(?P<source>.+?)(?:__x\d+_y\d+|_x\d+_y\d+)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("/workspace/data/main_dataset/manifest.jsonl"))
    parser.add_argument("--images-root", type=Path, default=Path("/workspace/data/main_dataset/images"))
    parser.add_argument("--rare-first-manifest", type=Path, default=Path("/workspace/data/main_dataset/benchmarks/vsb7_3600_rare_first/manifest.jsonl"))
    parser.add_argument("--rare-first-yaml", type=Path, default=Path("/workspace/data/main_dataset/benchmarks/vsb7_3600_rare_first_yolo/dataset.yaml"))
    parser.add_argument("--eval-map", type=Path, default=Path("results/corrected_common_eval_fixed/corrected_eval_dataset_map.csv"))
    parser.add_argument("--checkpoint-root", type=Path, default=Path("results/multiseed/vsb_rarefirst/per_seed/runs"))
    parser.add_argument("--clean-output-root", type=Path, default=Path("/workspace/data/main_dataset/benchmarks/vsb_clean_wood_yolo"))
    parser.add_argument("--clean-eval-root", type=Path, default=Path("/workspace/data/wood_defect_datacentric/corrected_common_eval_yolo_fixed/vsb_clean"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/vsb_negative_aware"))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--source-splits", nargs="+", default=["all"], help="Source manifest splits to use for clean tiles, or 'all'.")
    parser.add_argument("--max-clean-tiles", type=int, default=None, help="Optional deterministic cap for quick smoke tests.")
    parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--overwrite-clean-set", action="store_true")
    parser.add_argument("--overwrite-eval-datasets", action="store_true")
    parser.add_argument("--overwrite-predictions", action="store_true")
    parser.add_argument("--skip-inference", action="store_true", help="Reuse existing prediction JSONs.")
    parser.add_argument("--prepare-only", action="store_true", help="Build clean data and eval YAMLs, then stop.")
    parser.add_argument("--allow-leakage", action="store_true", help="Do not abort when train/val overlap is detected.")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260627)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_variant_labels(args.variants)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print_reuse_report()
    clean_report = build_clean_yolo_dataset(args)
    write_json(clean_report, output_dir / "clean_set_report.json")
    print_clean_report(clean_report)
    if clean_report["leakage"]["train_val_source_overlap"] or clean_report["leakage"]["train_val_tile_overlap"]:
        if not args.allow_leakage:
            raise SystemExit("Leakage check failed. Use --allow-leakage only for debugging.")

    positive_eval_yamls = resolve_positive_eval_yamls(args)
    clean_eval_yamls = resolve_clean_eval_yamls(args, Path(clean_report["dataset_yaml"]))
    write_eval_map(output_dir / "vsb_clean_eval_dataset_map.csv", positive_eval_yamls, clean_eval_yamls)
    if args.prepare_only:
        print("Prepared clean set and eval YAMLs.")
        return

    predictions_dir = output_dir / "predictions"
    if not args.skip_inference:
        export_predictions(args, positive_eval_yamls, clean_eval_yamls, predictions_dir)

    prediction_sets = load_vsb_prediction_sets(predictions_dir, args)
    threshold_outputs = analyze_thresholds(prediction_sets, args, output_dir)
    retained_outputs = analyze_retained_operating_points(prediction_sets, args, output_dir)
    calibration_outputs = analyze_calibration(prediction_sets, args, output_dir)

    summary = build_numeric_summary(clean_report, retained_outputs, calibration_outputs)
    write_json(summary, output_dir / "numeric_summary.json")
    write_latex_tables(output_dir, threshold_outputs, retained_outputs, calibration_outputs)
    print_numeric_summary(summary)
    print_latex_tables(retained_outputs, threshold_outputs, calibration_outputs)


def configure_variant_labels(variants: Iterable[str]) -> None:
    ordered = {variant: index for index, variant in enumerate(variants)}
    th.VARIANT_LABELS = {variant: LABELS.get(variant, variant) for variant in variants}
    th.VARIANT_ORDER = ordered
    cal.VARIANT_LABELS = {variant: LABELS.get(variant, variant) for variant in variants}
    cal.VARIANT_ORDER = ordered


def print_reuse_report() -> None:
    print("REUSED FUNCTIONS/PATHS")
    print(f"- clean materialization helpers: scripts.materialize_yolo_from_manifest ({mat_yolo.__file__})")
    print(f"- YOLO test loader/inference: scripts.threshold_sweep_inference ({infer.__file__})")
    print(f"- threshold/AP evaluator: scripts.threshold_analysis.evaluate_prediction_set ({source_location(th.evaluate_prediction_set)})")
    print(f"- class-level AP estimator: scripts.threshold_analysis.evaluate_class ({source_location(th.evaluate_class)})")
    print(f"- calibration matching: scripts.calibration_analysis.match_image_predictions ({source_location(cal.match_image_predictions)})")
    print(f"- preprocessing eval materializer: scripts.evaluate_corrected_common.materialize_preprocessed_eval_dataset ({source_location(common_eval.materialize_preprocessed_eval_dataset)})")


def build_clean_yolo_dataset(args: argparse.Namespace) -> dict[str, Any]:
    manifest = args.manifest.expanduser().resolve()
    images_root = args.images_root.expanduser().resolve()
    clean_output_root = args.clean_output_root.expanduser().resolve()
    if not manifest.exists():
        raise SystemExit(f"Missing VSB manifest: {manifest}")
    if not images_root.exists():
        raise SystemExit(f"Missing VSB images root: {images_root}")

    rows = [dict(row) for row in read_jsonl(manifest)]
    source_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_has_annotations: dict[str, bool] = defaultdict(bool)
    source_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        key = source_key(row)
        source_groups[key].append(row)
        source_splits[key].add(normalize_split(row.get("split")))
        if row.get("annotations"):
            source_has_annotations[key] = True

    requested_splits = {normalize_split(item) for item in args.source_splits if item.lower() != "all"}
    image_index = mat_yolo.build_image_index(images_root)
    clean_rows = []
    missing_images = []
    split_counter: Counter[str] = Counter()
    source_counter: Counter[str] = Counter()
    for key in sorted(source_groups):
        if source_has_annotations[key]:
            continue
        for row in sorted(source_groups[key], key=tile_key):
            split = normalize_split(row.get("split"))
            if requested_splits and split not in requested_splits:
                continue
            if row.get("annotations"):
                continue
            source_image = mat_yolo.resolve_image_path(row, image_index)
            if source_image is None:
                if len(missing_images) < 20:
                    missing_images.append({"image_id": str(row.get("image_id", "")), "image_path": str(row.get("image_path", ""))})
                continue
            clean_rows.append((row, source_image))
            split_counter[split] += 1
            source_counter[key] += 1

    if args.max_clean_tiles is not None:
        clean_rows = clean_rows[: int(args.max_clean_tiles)]
    if not clean_rows:
        raise SystemExit(
            "No clean VSB tile rows found. The full VSB manifest must include zero-annotation records "
            "from defect-free source images."
        )

    leakage = leakage_check(clean_rows, args)
    dataset_yaml = clean_output_root / "dataset.yaml"
    reuse_existing = dataset_yaml.exists() and not args.overwrite_clean_set
    if reuse_existing:
        existing_images = [
            path
            for path in (clean_output_root / "images" / "test").rglob("*")
            if path.suffix.lower() in IMAGE_EXTENSIONS
        ]
        if len(existing_images) != len(clean_rows):
            raise SystemExit(
                f"Existing clean set has {len(existing_images)} test images, expected {len(clean_rows)} from manifest. "
                "Pass --overwrite-clean-set to rebuild it."
            )
        print(f"Reusing existing clean YOLO dataset: {dataset_yaml}")
    else:
        prepare_output(clean_output_root, overwrite=bool(args.overwrite_clean_set))
        materialized_rows = []
        for row, source_image in clean_rows:
            rel_image = clean_relative_image_path(row, source_image, images_root)
            target_image = clean_output_root / "images" / "test" / rel_image
            target_label = clean_output_root / "labels" / "test" / rel_image.with_suffix(".txt")
            mat_yolo.place_image(source_image, target_image, mode=args.link_mode)
            mat_yolo.write_label_file(target_label, [])
            materialized_rows.append(
                {
                    "source_key": source_key(row),
                    "tile_key": tile_key(row),
                    "split": normalize_split(row.get("split")),
                    "source_image": str(source_image),
                    "target_image": str(target_image),
                    "target_label": str(target_label),
                }
            )
        dataset_yaml = write_clean_dataset_yaml(clean_output_root)
        write_csv(materialized_rows, clean_output_root / "clean_materialized_samples.csv")
    report = {
        "manifest": str(manifest),
        "images_root": str(images_root),
        "output_root": str(clean_output_root),
        "dataset_yaml": str(dataset_yaml),
        "link_mode": args.link_mode,
        "num_manifest_records": len(rows),
        "num_clean_source_images": len({source_key(row) for row, _ in clean_rows}),
        "num_clean_tiles": len(clean_rows),
        "source_split_tile_counts": dict(sorted(split_counter.items())),
        "missing_image_examples": missing_images,
        "leakage": leakage,
        "note": "Clean records are zero-annotation tiles from source groups with no annotation in the full VSB manifest.",
    }
    write_json(report, clean_output_root / "clean_set_report.json")
    return report


def leakage_check(clean_rows: list[tuple[dict[str, Any], Path]], args: argparse.Namespace) -> dict[str, Any]:
    clean_sources = {source_key(row) for row, _ in clean_rows}
    clean_tiles = {tile_key(row) for row, _ in clean_rows}
    rare_sources_by_split, rare_tiles_by_split = load_rare_first_membership(args)
    train_val_sources = set().union(*(rare_sources_by_split.get(split, set()) for split in ("train", "val")))
    train_val_tiles = set().union(*(rare_tiles_by_split.get(split, set()) for split in ("train", "val")))
    test_sources = rare_sources_by_split.get("test", set())
    test_tiles = rare_tiles_by_split.get("test", set())
    source_overlap = sorted(clean_sources & train_val_sources)
    tile_overlap = sorted(clean_tiles & train_val_tiles)
    return {
        "clean_source_images": len(clean_sources),
        "clean_tiles": len(clean_tiles),
        "rare_train_val_source_images": len(train_val_sources),
        "rare_train_val_tiles": len(train_val_tiles),
        "train_val_source_overlap": len(source_overlap),
        "train_val_tile_overlap": len(tile_overlap),
        "test_source_overlap": len(clean_sources & test_sources),
        "test_tile_overlap": len(clean_tiles & test_tiles),
        "source_overlap_examples": source_overlap[:20],
        "tile_overlap_examples": tile_overlap[:20],
    }


def load_rare_first_membership(args: argparse.Namespace) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    sources: dict[str, set[str]] = defaultdict(set)
    tiles: dict[str, set[str]] = defaultdict(set)
    rare_manifest = args.rare_first_manifest.expanduser()
    if rare_manifest.exists():
        for row in read_jsonl(rare_manifest):
            split = normalize_split(row.get("split"))
            sources[split].add(source_key(row))
            tiles[split].add(tile_key(row))
    rare_yaml = args.rare_first_yaml.expanduser()
    if rare_yaml.exists():
        for split, image_path in yolo_split_images(rare_yaml, ("train", "val", "test")):
            key = tile_key_from_path(image_path)
            tiles[split].add(key)
            sources[split].add(source_key_from_tile_text(key))
    return sources, tiles


def resolve_positive_eval_yamls(args: argparse.Namespace) -> dict[str, Path]:
    variants = tuple(args.variants)
    mapping = read_vsb_eval_map(args.eval_map.expanduser())
    output: dict[str, Path] = {}
    for variant in variants:
        if variant in mapping and mapping[variant].exists():
            output[variant] = mapping[variant].resolve()
    raw_yaml = args.rare_first_yaml.expanduser().resolve()
    if not raw_yaml.exists():
        raise SystemExit(f"Missing VSB rare-first YAML: {raw_yaml}")
    for variant in variants:
        if variant in output:
            continue
        preprocessing = PREPROCESSING.get(variant)
        if preprocessing:
            output[variant] = common_eval.materialize_preprocessed_eval_dataset(
                source_yaml=raw_yaml,
                preprocessing=preprocessing,
                output_root=args.clean_eval_root.expanduser().resolve().parent / "vsb_positive" / preprocessing,
                overwrite=bool(args.overwrite_eval_datasets),
                splits=("test",),
            )
        else:
            output[variant] = raw_yaml
    return output


def resolve_clean_eval_yamls(args: argparse.Namespace, clean_raw_yaml: Path) -> dict[str, Path]:
    output: dict[str, Path] = {}
    root = args.clean_eval_root.expanduser().resolve()
    for variant in args.variants:
        preprocessing = PREPROCESSING.get(variant)
        if preprocessing:
            output[variant] = common_eval.materialize_preprocessed_eval_dataset(
                source_yaml=clean_raw_yaml,
                preprocessing=preprocessing,
                output_root=root / preprocessing,
                overwrite=bool(args.overwrite_eval_datasets),
                splits=("test",),
            )
        else:
            output[variant] = clean_raw_yaml
    return output


def export_predictions(
    args: argparse.Namespace,
    positive_eval_yamls: dict[str, Path],
    clean_eval_yamls: dict[str, Path],
    predictions_dir: Path,
) -> None:
    predictions_dir.mkdir(parents=True, exist_ok=True)
    class_names = VSB_CLASSES
    for variant in args.variants:
        pos_records, pos_classes = infer.load_yolo_test_records(positive_eval_yamls[variant])
        clean_records, clean_classes = infer.load_yolo_test_records(clean_eval_yamls[variant])
        if tuple(pos_classes) != tuple(clean_classes):
            raise SystemExit(f"Class names differ for {variant}: positive={pos_classes}, clean={clean_classes}")
        if tuple(pos_classes) != VSB_CLASSES:
            class_names = tuple(pos_classes)
        clean_records = [record for record in clean_records if record.is_knot_free]
        if not clean_records:
            raise SystemExit(f"No empty-label clean records for {variant}: {clean_eval_yamls[variant]}")
        records = pos_records + clean_records
        for seed in args.seeds:
            run_dir = args.checkpoint_root.expanduser().resolve() / f"{variant}_seed{seed}"
            checkpoint = run_dir / "ultralytics" / "train" / "weights" / "best.pt"
            output_path = predictions_dir / f"{variant}_seed{seed}_predictions.json"
            if output_path.exists() and not args.overwrite_predictions:
                print(f"Existing prediction JSON; skipping: {output_path}")
                continue
            if not checkpoint.exists():
                message = f"Missing checkpoint: {checkpoint}"
                if args.allow_missing:
                    print(f"WARNING: {message}")
                    continue
                raise SystemExit(message)
            print(f"Predicting {variant}_seed{seed}: positive={len(pos_records)} clean={len(clean_records)}")
            predictions_by_key = infer.predict_records(
                checkpoint=checkpoint,
                records=records,
                class_names=tuple(class_names),
                imgsz=int(args.imgsz),
                batch=int(args.batch),
                max_det=int(args.max_det),
                conf=float(args.conf),
                iou=float(args.nms_iou),
                device=str(args.device),
            )
            payload = build_prediction_payload(
                variant=variant,
                seed=int(seed),
                checkpoint=checkpoint,
                positive_yaml=positive_eval_yamls[variant],
                clean_yaml=clean_eval_yamls[variant],
                records=records,
                predictions_by_key=predictions_by_key,
                class_names=tuple(class_names),
                conf=float(args.conf),
            )
            write_json(payload, output_path)
            print(f"Wrote: {output_path}")


def build_prediction_payload(
    *,
    variant: str,
    seed: int,
    checkpoint: Path,
    positive_yaml: Path,
    clean_yaml: Path,
    records: list[infer.TestRecord],
    predictions_by_key: dict[str, list[dict[str, Any]]],
    class_names: tuple[str, ...],
    conf: float,
) -> dict[str, Any]:
    images = []
    for record in records:
        images.append(
            {
                "image": record.image_name,
                "canonical_id": record.canonical_id,
                "image_path": str(record.image_path),
                "width": record.width,
                "height": record.height,
                "is_knot_free": record.is_knot_free,
                "gt_boxes": record.gt_boxes_json,
                "predictions": predictions_by_key.get(record.canonical_id, []),
            }
        )
    return {
        "checkpoint": f"{variant}_seed{seed}",
        "variant": variant,
        "seed": seed,
        "checkpoint_path": str(checkpoint),
        "dataset_yaml": str(positive_yaml),
        "clean_dataset_yaml": str(clean_yaml),
        "split": "test",
        "base_confidence_threshold": float(conf),
        "class_names": list(class_names),
        "num_images": len(images),
        "num_positive_images": sum(1 for image in images if not image["is_knot_free"]),
        "num_knot_free_images": sum(1 for image in images if image["is_knot_free"]),
        "images": images,
    }


def analyze_thresholds(
    prediction_sets: list[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    thresholds = th.make_thresholds(args.threshold_start, args.threshold_end, args.threshold_step)
    threshold_dir = output_dir / "threshold_sweep"
    bootstrap_dir = output_dir / "bootstrap"
    threshold_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(int(args.bootstrap_seed))
    raw_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    pooled_bootstrap: dict[tuple[str, float], list[np.ndarray]] = defaultdict(list)
    for prediction_set in sorted(prediction_sets, key=prediction_sort_key):
        print(f"Analyzing thresholds {prediction_set['variant']}_seed{prediction_set['seed']}")
        for threshold in thresholds:
            metrics, negative_vector = th.evaluate_prediction_set(
                prediction_set,
                threshold=threshold,
                iou_threshold=float(args.match_iou),
            )
            raw_rows.append(metrics)
            distribution = th.bootstrap_fp_rates(negative_vector, rng=rng, samples=int(args.bootstrap_samples))
            pooled_bootstrap[(prediction_set["variant"], threshold)].append(distribution)
            bootstrap_rows.append(
                {
                    "variant": prediction_set["variant"],
                    "seed": prediction_set["seed"],
                    "threshold": th.fmt_threshold(threshold),
                    "fp_rate": th.format_float(th.parse_float(metrics["fp_image_rate"])),
                    "ci_lower": th.format_float(th.percentile(distribution, 2.5)),
                    "ci_upper": th.format_float(th.percentile(distribution, 97.5)),
                }
            )
    summary_rows = th.aggregate_raw_rows(raw_rows, pooled_bootstrap)
    bootstrap_summary_rows = th.build_bootstrap_summary(raw_rows, pooled_bootstrap, key_thresholds=(0.25, 0.50, 0.75))
    sweet_spots = th.find_operational_sweet_spots(raw_rows)
    th.write_csv(raw_rows, threshold_dir / "raw_data.csv")
    th.write_csv(summary_rows, threshold_dir / "summary_aggregated.csv")
    th.write_csv(bootstrap_rows, bootstrap_dir / "bootstrap_ci_results.csv")
    th.write_csv(bootstrap_summary_rows, bootstrap_dir / "bootstrap_summary_table.csv")
    th.write_csv(sweet_spots, output_dir / "operational_sweet_spots_threshold_analysis.csv")
    return {
        "thresholds": thresholds,
        "raw_rows": raw_rows,
        "summary_rows": summary_rows,
        "bootstrap_rows": bootstrap_rows,
        "bootstrap_summary_rows": bootstrap_summary_rows,
        "sweet_spots": sweet_spots,
    }


def analyze_retained_operating_points(
    prediction_sets: list[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    thresholds = th.make_thresholds(args.threshold_start, args.threshold_end, args.threshold_step)
    per_variant: dict[str, Any] = {}
    for variant in args.variants:
        variant_sets = sorted([item for item in prediction_sets if item["variant"] == variant], key=lambda item: int(item["seed"]))
        if not variant_sets:
            continue
        threshold_metrics = {
            threshold: [evaluate_retained(item, threshold=threshold, iou_threshold=float(args.match_iou)) for item in variant_sets]
            for threshold in thresholds
        }
        tau_star = find_zero_fp_tau(thresholds, threshold_metrics)
        per_variant[variant] = {
            "variant": variant,
            "variant_label": LABELS.get(variant, variant),
            "tau_star": tau_star,
            "per_seed_at_tau_star": threshold_metrics[tau_star],
            "summary_at_tau_star": summarize_threshold(tau_star, threshold_metrics),
            "epsilon_operating_points": {
                f"{epsilon:.2f}": summarize_threshold(select_epsilon_tau(thresholds, threshold_metrics, epsilon), threshold_metrics)
                for epsilon in EPSILONS
            },
        }
    payload = {
        "threshold_grid": thresholds,
        "iou_threshold": float(args.match_iou),
        "variants": per_variant,
    }
    write_json(payload, output_dir / "retained_metrics.json")
    return payload


def analyze_calibration(
    prediction_sets: list[dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> dict[str, Any]:
    calibration_dir = output_dir / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    records = cal.build_detection_records(prediction_sets, min_conf=float(args.conf), iou_threshold=float(args.match_iou))
    per_seed_rows, reliability_rows = cal.compute_calibration(records, bins=10)
    summary_rows = cal.summarize_per_seed(per_seed_rows)
    clean_max_rows, clean_max_per_seed_rows, clean_max_summary_rows, clean_max_cdf_rows = cal.compute_clean_max_confidence(
        prediction_sets,
        min_conf=float(args.conf),
    )
    summary_rows = cal.merge_clean_max_into_summary(summary_rows, clean_max_summary_rows)
    cal.write_csv(records, calibration_dir / "calibration_records.csv", cal.REQUESTED_RECORD_FIELDS)
    cal.write_csv(records, calibration_dir / "per_detection_records.csv", cal.REQUESTED_RECORD_FIELDS)
    cal.write_csv(per_seed_rows, calibration_dir / "calibration_per_seed.csv")
    cal.write_csv(summary_rows, calibration_dir / "calibration_summary.csv")
    cal.write_csv(reliability_rows, calibration_dir / "reliability_bins.csv")
    cal.write_csv(clean_max_rows, calibration_dir / "clean_max_confidence_per_image.csv")
    cal.write_csv(clean_max_per_seed_rows, calibration_dir / "clean_max_confidence_per_seed.csv")
    cal.write_csv(clean_max_summary_rows, calibration_dir / "clean_max_confidence_summary.csv")
    cal.write_csv(clean_max_cdf_rows, calibration_dir / "clean_max_confidence_cdf.csv")
    cal.plot_reliability(reliability_rows, calibration_dir)
    cal.plot_clean_max_confidence_cdf(clean_max_cdf_rows, calibration_dir)
    return {
        "records": records,
        "per_seed_rows": per_seed_rows,
        "summary_rows": summary_rows,
        "reliability_rows": reliability_rows,
        "clean_max_rows": clean_max_rows,
        "clean_max_summary_rows": clean_max_summary_rows,
        "clean_max_cdf_rows": clean_max_cdf_rows,
    }


def load_vsb_prediction_sets(predictions_dir: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    prediction_sets = th.load_prediction_sets(predictions_dir)
    wanted = {(variant, int(seed)) for variant in args.variants for seed in args.seeds}
    output = []
    for item in prediction_sets:
        key = (str(item["variant"]), int(item["seed"]))
        if key in wanted:
            output.append(item)
    found = {(str(item["variant"]), int(item["seed"])) for item in output}
    missing = sorted(wanted - found, key=lambda item: (variant_order(item[0], args.variants), item[1]))
    if missing and not args.allow_missing:
        raise SystemExit(f"Missing VSB prediction JSONs: {missing}")
    for item in output:
        positive = [image for image in item["images"] if not bool(image.get("is_knot_free", False))]
        negative = [image for image in item["images"] if bool(image.get("is_knot_free", False))]
        gt = sum(len(image.get("gt_boxes", [])) for image in positive)
        print(f"Prediction set {item['variant']}_seed{item['seed']}: positive={len(positive)} clean={len(negative)} gt={gt}")
    return sorted(output, key=prediction_sort_key)


def evaluate_retained(prediction_set: dict[str, Any], *, threshold: float, iou_threshold: float) -> dict[str, Any]:
    row, _ = th.evaluate_prediction_set(prediction_set, threshold=threshold, iou_threshold=iou_threshold)
    positive_images = [image for image in prediction_set["images"] if not bool(image.get("is_knot_free", False))]
    negative_images = [image for image in prediction_set["images"] if bool(image.get("is_knot_free", False))]
    retained_positive = count_retained(positive_images, threshold)
    retained_negative = count_retained(negative_images, threshold)
    tp = int(row["tp50"])
    fp_positive = int(row["fp50_positive"])
    fn = int(row["fn50"])
    return {
        "variant": prediction_set["variant"],
        "seed": int(prediction_set["seed"]),
        "threshold": float(threshold),
        "n_retained": int(retained_positive + retained_negative),
        "n_retained_positive": int(retained_positive),
        "n_retained_clean": int(retained_negative),
        "n_TP": tp,
        "n_FP": fp_positive + int(retained_negative),
        "n_FP_positive": fp_positive,
        "n_FP_clean": int(retained_negative),
        "n_FN": fn,
        "precision_at_tau": float(th.parse_float(row["precision"])),
        "retained_recall": float(th.parse_float(row["recall"])),
        "retained_AP50": float(th.parse_float(row["ap50"])),
        "fp_image_rate_clean": float(th.parse_float(row["fp_image_rate"])),
        "clean_fp_images": int(row["knotfree_fp_images"]),
        "num_positive_images": int(row["num_positive_images"]),
        "num_clean_images": int(row["num_knotfree_images"]),
        "num_targets": tp + fn,
    }


def count_retained(images: list[dict[str, Any]], threshold: float) -> int:
    return sum(
        1
        for image in images
        for prediction in image.get("predictions", [])
        if float(prediction.get("conf", 0.0)) >= threshold
    )


def find_zero_fp_tau(thresholds: list[float], threshold_metrics: dict[float, list[dict[str, Any]]]) -> float:
    for threshold in thresholds:
        if all(row["fp_image_rate_clean"] == 0.0 for row in threshold_metrics[threshold]):
            return float(threshold)
    return float("nan")


def select_epsilon_tau(thresholds: list[float], threshold_metrics: dict[float, list[dict[str, Any]]], epsilon: float) -> float:
    for threshold in thresholds:
        fp_mean = mean(row["fp_image_rate_clean"] for row in threshold_metrics[threshold])
        if fp_mean <= epsilon + 1e-12:
            return float(threshold)
    return float("nan")


def summarize_threshold(threshold: float, threshold_metrics: dict[float, list[dict[str, Any]]]) -> dict[str, Any]:
    if math.isnan(threshold):
        return {"threshold": float("nan"), "n_seeds": 0, "per_seed": []}
    rows = threshold_metrics[threshold]
    keys = (
        "n_retained",
        "n_TP",
        "n_FP",
        "precision_at_tau",
        "retained_recall",
        "retained_AP50",
        "fp_image_rate_clean",
        "clean_fp_images",
        "num_clean_images",
        "num_targets",
    )
    summary: dict[str, Any] = {
        "threshold": float(threshold),
        "n_seeds": len(rows),
        "seeds": [int(row["seed"]) for row in rows],
        "per_seed": rows,
    }
    for key in keys:
        summary[f"{key}_mean"] = mean(row[key] for row in rows)
        summary[f"{key}_std"] = std(row[key] for row in rows)
    return summary


def build_numeric_summary(clean_report: dict[str, Any], retained: dict[str, Any], calibration: dict[str, Any]) -> dict[str, Any]:
    zero_fp_rows = []
    for variant, item in retained["variants"].items():
        summary = item["summary_at_tau_star"]
        zero_fp_rows.append(
            {
                "variant": variant,
                "variant_label": LABELS.get(variant, variant),
                "tau_star": item["tau_star"],
                "retained_recall_mean": summary.get("retained_recall_mean"),
                "retained_recall_std": summary.get("retained_recall_std"),
                "retained_AP50_mean": summary.get("retained_AP50_mean"),
                "retained_AP50_std": summary.get("retained_AP50_std"),
                "precision_at_tau_mean": summary.get("precision_at_tau_mean"),
                "fp_image_rate_clean_mean": summary.get("fp_image_rate_clean_mean"),
            }
        )
    clean_max = {
        row["variant"]: {
            "mean_max_confidence": row.get("mean_max_confidence_mean"),
            "p90_max_confidence": row.get("p90_max_confidence_mean"),
            "p95_max_confidence": row.get("p95_max_confidence_mean"),
        }
        for row in calibration["clean_max_summary_rows"]
    }
    return {
        "clean_set": {
            "num_clean_source_images": clean_report["num_clean_source_images"],
            "num_clean_tiles": clean_report["num_clean_tiles"],
            "leakage": clean_report["leakage"],
        },
        "zero_fp_operating_points": zero_fp_rows,
        "clean_max_confidence": clean_max,
        "interpretation_placeholder": "Compare tau_star, retained_AP50, and clean max-confidence rankings against VNWoodKnot.",
    }


def write_latex_tables(
    output_dir: Path,
    threshold_outputs: dict[str, Any],
    retained_outputs: dict[str, Any],
    calibration_outputs: dict[str, Any],
) -> None:
    table_dir = output_dir / "latex"
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / "opsel_rows.tex").write_text("\n".join(latex_opsel_rows(retained_outputs)) + "\n", encoding="utf-8")
    (table_dir / "sensitivity_rows.tex").write_text("\n".join(latex_sensitivity_rows(retained_outputs)) + "\n", encoding="utf-8")
    (table_dir / "fp_rows.tex").write_text("\n".join(latex_fp_rows(threshold_outputs)) + "\n", encoding="utf-8")
    (table_dir / "calibration_rows.tex").write_text("\n".join(latex_calibration_rows(calibration_outputs)) + "\n", encoding="utf-8")


def print_latex_tables(retained: dict[str, Any], threshold_outputs: dict[str, Any], calibration: dict[str, Any]) -> None:
    print("\nLATEX OPSel ROWS")
    for row in latex_opsel_rows(retained):
        print(row)
    print("\nLATEX SENSITIVITY ROWS")
    for row in latex_sensitivity_rows(retained):
        print(row)
    print("\nLATEX FP ROWS")
    for row in latex_fp_rows(threshold_outputs):
        print(row)
    print("\nLATEX CALIBRATION ROWS")
    for row in latex_calibration_rows(calibration):
        print(row)


def latex_opsel_rows(retained: dict[str, Any]) -> list[str]:
    rows = []
    for variant in sorted(retained["variants"], key=lambda item: variant_order(item, VARIANTS)):
        item = retained["variants"][variant]
        summary = item["summary_at_tau_star"]
        rows.append(
            f"{LABELS.get(variant, variant)} & {fmt_num(item['tau_star'], 2)} & "
            f"{fmt_pm(summary.get('precision_at_tau_mean'), summary.get('precision_at_tau_std'))} & "
            f"{fmt_pm(summary.get('retained_recall_mean'), summary.get('retained_recall_std'))} & "
            f"{fmt_pm(summary.get('retained_AP50_mean'), summary.get('retained_AP50_std'))} \\\\"
        )
    return rows


def latex_sensitivity_rows(retained: dict[str, Any]) -> list[str]:
    rows = []
    for variant in sorted(retained["variants"], key=lambda item: variant_order(item, VARIANTS)):
        item = retained["variants"][variant]
        parts = [LABELS.get(variant, variant)]
        for epsilon in EPSILONS:
            summary = item["epsilon_operating_points"][f"{epsilon:.2f}"]
            parts.extend([fmt_num(summary.get("threshold"), 2), fmt_num(summary.get("retained_AP50_mean"), 3)])
        rows.append(" & ".join(parts) + r" \\")
    return rows


def latex_fp_rows(threshold_outputs: dict[str, Any]) -> list[str]:
    rows = []
    for row in threshold_outputs["bootstrap_summary_rows"]:
        variant = str(row["variant"])
        rows.append(
            f"{LABELS.get(variant, variant)} & {float(row['threshold']):.2f} & "
            f"{float(row['fp_rate_mean']):.3f}$\\pm${float(row['fp_rate_std']):.3f} & "
            f"[{float(row['ci_lower']):.3f}, {float(row['ci_upper']):.3f}] \\\\"
        )
    return rows


def latex_calibration_rows(calibration: dict[str, Any]) -> list[str]:
    rows = []
    for row in calibration["summary_rows"]:
        variant = str(row["variant"])
        rows.append(
            f"{LABELS.get(variant, variant)} & "
            f"{float(row['num_defective_detections_mean']):.0f}$\\pm${float(row['num_defective_detections_std']):.0f} & "
            f"{float(row['d_ece_mean']):.3f}$\\pm${float(row['d_ece_std']):.3f} & "
            f"{float(row['signed_gap_mean']):+.3f}$\\pm${float(row['signed_gap_std']):.3f} & "
            f"{float(row.get('clean_p95_max_confidence_mean', 'nan')):.3f}$\\pm${float(row.get('clean_p95_max_confidence_std', 'nan')):.3f} \\\\"
        )
    return rows


def print_clean_report(report: dict[str, Any]) -> None:
    leakage = report["leakage"]
    print("\nCLEAN SET REPORT")
    print(f"- clean source images: {report['num_clean_source_images']}")
    print(f"- clean tiles: {report['num_clean_tiles']}")
    print(f"- source split tile counts: {report['source_split_tile_counts']}")
    print(
        "- leakage train/val: "
        f"source_overlap={leakage['train_val_source_overlap']}, tile_overlap={leakage['train_val_tile_overlap']}"
    )
    if leakage["train_val_source_overlap"] == 0 and leakage["train_val_tile_overlap"] == 0:
        print("LEAKAGE CHECK: PASS (zero source/tile overlap with VSB rare-first train/val)")
    else:
        print("LEAKAGE CHECK: FAIL")


def print_numeric_summary(summary: dict[str, Any]) -> None:
    print("\nNUMERIC SUMMARY")
    clean = summary["clean_set"]
    print(f"Clean VSB sources={clean['num_clean_source_images']}, clean tiles={clean['num_clean_tiles']}")
    for row in summary["zero_fp_operating_points"]:
        print(
            f"- {row['variant_label']}: tau={fmt_num(row['tau_star'], 2)}, "
            f"recall={fmt_num(row['retained_recall_mean'], 3)}, AP50={fmt_num(row['retained_AP50_mean'], 3)}, "
            f"precision={fmt_num(row['precision_at_tau_mean'], 3)}"
        )
    print("Conclusion check: compare these VSB rankings with VNWoodKnot; do not assume they match.")


def source_key(row: dict[str, Any]) -> str:
    for key in ("source_image_id", "source_id", "original_image_id", "parent_image_id"):
        value = str(row.get(key) or "").strip()
        if value:
            return normalize_identifier(strip_tile_suffix(value))
    return source_key_from_tile_text(tile_key(row))


def source_key_from_tile_text(value: str) -> str:
    text = Path(str(value).replace("\\", "/")).with_suffix("").as_posix()
    stem = Path(text).name
    stripped = strip_tile_suffix(stem)
    if stripped != stem:
        return normalize_identifier(stripped)
    parts = [part for part in Path(text).parts if part not in {"images", "train", "val", "test"}]
    return normalize_identifier("/".join(parts[:-1] + [stripped]) if len(parts) > 1 else stripped)


def tile_key(row: dict[str, Any]) -> str:
    value = str(row.get("image_id") or row.get("image_path") or "").strip()
    return normalize_identifier(Path(value.replace("\\", "/")).with_suffix("").as_posix())


def tile_key_from_path(path: Path) -> str:
    return normalize_identifier(path.with_suffix("").as_posix())


def strip_tile_suffix(value: str) -> str:
    text = Path(str(value).replace("\\", "/")).with_suffix("").as_posix()
    directory = Path(text).parent.as_posix()
    stem = Path(text).name
    match = SOURCE_TILE_RE.match(stem)
    if match:
        stem = match.group("source")
    return f"{directory}/{stem}" if directory and directory != "." else stem


def normalize_identifier(value: str) -> str:
    return str(value).replace("\\", "/").strip().lower().strip("/")


def clean_relative_image_path(row: dict[str, Any], source_image: Path, images_root: Path) -> Path:
    try:
        return source_image.relative_to(images_root)
    except ValueError:
        split = normalize_split(row.get("split"))
        category = safe_part(str(row.get("source_category") or "clean"))
        return Path(split) / category / Path(str(row.get("image_path") or source_image.name)).name


def write_clean_dataset_yaml(root: Path) -> Path:
    for split in ("train", "val", "test"):
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    path = root / "dataset.yaml"
    payload = {
        "path": str(root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(VSB_CLASSES)},
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def yolo_split_images(data_yaml: Path, splits: Iterable[str]) -> Iterable[tuple[str, Path]]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    root = infer.resolve_dataset_root(data_yaml, data)
    for split in splits:
        for split_dir in infer.resolve_split_dirs(data_yaml, data, split):
            if not split_dir.exists():
                continue
            for image_path in sorted(path for path in split_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS):
                try:
                    rel = image_path.relative_to(root)
                except ValueError:
                    rel = image_path
                yield split, rel


def read_vsb_eval_map(path: Path) -> dict[str, Path]:
    mapping: dict[str, Path] = {}
    if not path.exists():
        return mapping
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("dataset") != "vsb_rarefirst":
                continue
            variant = str(row.get("variant") or "").strip()
            data_yaml = str(row.get("data_yaml") or "").strip()
            if variant and data_yaml:
                mapping[variant] = Path(data_yaml).expanduser()
    return mapping


def write_eval_map(path: Path, positive_yamls: dict[str, Path], clean_yamls: dict[str, Path]) -> None:
    rows = []
    for variant in sorted(positive_yamls, key=lambda item: variant_order(item, VARIANTS)):
        rows.append(
            {
                "dataset": "vsb_rarefirst",
                "variant": variant,
                "positive_test_yaml": str(positive_yamls[variant]),
                "clean_test_yaml": str(clean_yamls[variant]),
            }
        )
    write_csv(rows, path)


def prepare_output(path: Path, *, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise SystemExit(f"Output root exists and is not empty: {path}. Use --overwrite-clean-set intentionally.")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def prediction_sort_key(item: dict[str, Any]) -> tuple[int, int]:
    return variant_order(str(item["variant"]), VARIANTS), int(item["seed"])


def variant_order(variant: str, variants: Iterable[str]) -> int:
    order = {name: index for index, name in enumerate(variants)}
    return order.get(str(variant), 999)


def safe_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return cleaned or "clean"


def source_location(func: Any) -> str:
    import inspect

    path = Path(inspect.getsourcefile(func) or "")
    try:
        rel = path.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = path
    _, start_line = inspect.getsourcelines(func)
    return f"{rel}:{start_line}"


def mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if float(value) == float(value)]
    return sum(vals) / len(vals) if vals else float("nan")


def std(values: Iterable[float]) -> float:
    vals = [float(value) for value in values if float(value) == float(value)]
    if len(vals) < 2:
        return 0.0
    return statistics.stdev(vals)


def fmt_num(value: Any, digits: int) -> str:
    try:
        number = float(value)
    except Exception:
        return "n/a"
    if math.isnan(number):
        return "n/a"
    return f"{number:.{digits}f}"


def fmt_pm(mean_value: Any, std_value: Any, digits: int = 3) -> str:
    return f"{fmt_num(mean_value, digits)}$\\pm${fmt_num(std_value, digits)}"


if __name__ == "__main__":
    main()
