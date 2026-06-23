#!/usr/bin/env python3
"""Detection-level calibration analysis for VNWoodKnot.

The analysis uses the corrected common-evaluation prediction JSON files emitted by
``scripts/threshold_sweep_inference.py``. If those files are not present, pass
``--run-inference`` to export them first from the seed checkpoints.

Outputs:
- calibration_records.csv
- per_detection_records.csv
- calibration_per_seed.csv
- calibration_summary.csv
- reliability_bins.csv
- clean_max_confidence_per_image.csv
- clean_max_confidence_summary.csv
- clean_max_confidence_cdf.{pdf,png}
- reliability_curve.{pdf,png}
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
VARIANT_LABELS = {
    "baseline": "Baseline",
    "p2_illumination": "P2 illumination",
    "a1_crop": "A1 crop",
    "a2_colorjitter": "A2 colour jitter",
    "p4_a4_combined": "P4+A4 combined",
}
VARIANT_ORDER = {variant: index for index, variant in enumerate(VARIANT_LABELS)}
EXPECTED_VARIANTS = tuple(VARIANT_LABELS)
EXPECTED_SEEDS = (42, 43, 44)
PREDICTION_RE = re.compile(r"^(?P<variant>.+)_seed(?P<seed>\d+)_predictions\.json$")
REQUESTED_RECORD_FIELDS = ["variant", "seed", "image_id", "is_knotfree", "conf", "x", "y", "w", "h", "matched_TP"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/calibration"))
    parser.add_argument(
        "--eval-map",
        type=Path,
        default=Path("results/corrected_common_eval_fixed/corrected_eval_dataset_map.csv"),
        help="Corrected per-variant dataset YAML map, used only with --run-inference.",
    )
    parser.add_argument(
        "--checkpoint-root",
        type=Path,
        default=Path("results/multiseed/vnwoodknot/per_seed/runs"),
        help="VNWoodKnot multiseed checkpoint root, used only with --run-inference.",
    )
    parser.add_argument("--variants", nargs="+", default=list(EXPECTED_VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(EXPECTED_SEEDS))
    parser.add_argument("--gpus", default="0", help="GPU list passed to threshold_sweep_inference.py when --run-inference is set.")
    parser.add_argument("--device", default="0", help="Reserved for single-GPU consistency; inference subprocess uses device 0 inside CUDA_VISIBLE_DEVICES.")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--conf", type=float, default=0.001, help="Minimum exported confidence.")
    parser.add_argument("--nms-iou", type=float, default=0.7, help="NMS IoU used during prediction export.")
    parser.add_argument("--match-iou", type=float, default=0.50, help="IoU threshold for detection-to-GT calibration matching.")
    parser.add_argument("--bins", type=int, default=10, help="Number of equal-width confidence bins.")
    parser.add_argument("--run-inference", action="store_true", help="Export corrected prediction JSONs before calibration analysis.")
    parser.add_argument("--overwrite-predictions", action="store_true")
    parser.add_argument(
        "--allow-higher-base-conf",
        action="store_true",
        help="Do not fail if existing prediction JSONs were exported with base_confidence_threshold above --conf.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    predictions_dir = resolve_predictions_dir(args, output_dir)

    if args.run_inference:
        run_inference(args, predictions_dir)

    prediction_sets = load_prediction_sets(
        predictions_dir=predictions_dir,
        variants=tuple(args.variants),
        seeds=tuple(args.seeds),
        min_conf=float(args.conf),
        allow_higher_base_conf=bool(args.allow_higher_base_conf),
    )
    if not prediction_sets:
        raise SystemExit(f"No prediction JSONs found in: {predictions_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    records = build_detection_records(prediction_sets, min_conf=float(args.conf), iou_threshold=float(args.match_iou))
    per_seed_rows, reliability_rows = compute_calibration(records, bins=int(args.bins))
    summary_rows = summarize_per_seed(per_seed_rows)
    knotfree_rows, knotfree_summary_rows, knotfree_hist_rows = compute_knotfree_confidence(
        prediction_sets,
        min_conf=float(args.conf),
        bins=int(args.bins),
    )
    clean_max_rows, clean_max_per_seed_rows, clean_max_summary_rows, clean_max_cdf_rows = compute_clean_max_confidence(
        prediction_sets,
        min_conf=float(args.conf),
    )
    summary_rows = merge_clean_max_into_summary(summary_rows, clean_max_summary_rows)

    write_csv(records, output_dir / "calibration_records.csv", REQUESTED_RECORD_FIELDS)
    write_csv(records, output_dir / "per_detection_records.csv", REQUESTED_RECORD_FIELDS)
    write_csv(per_seed_rows, output_dir / "calibration_per_seed.csv")
    write_csv(summary_rows, output_dir / "calibration_summary.csv")
    write_csv(reliability_rows, output_dir / "reliability_bins.csv")
    write_csv(knotfree_rows, output_dir / "knotfree_confidence_per_seed.csv")
    write_csv(knotfree_summary_rows, output_dir / "knotfree_confidence_summary.csv")
    write_csv(knotfree_hist_rows, output_dir / "knotfree_confidence_bins.csv")
    write_csv(clean_max_rows, output_dir / "clean_max_confidence_per_image.csv")
    write_csv(clean_max_per_seed_rows, output_dir / "clean_max_confidence_per_seed.csv")
    write_csv(clean_max_summary_rows, output_dir / "clean_max_confidence_summary.csv")
    write_csv(clean_max_cdf_rows, output_dir / "clean_max_confidence_cdf.csv")
    plot_reliability(reliability_rows, output_dir)
    plot_knotfree_confidence(knotfree_hist_rows, output_dir)
    plot_clean_max_confidence_cdf(clean_max_cdf_rows, output_dir)

    print(f"Wrote: {output_dir / 'calibration_records.csv'}")
    print(f"Wrote: {output_dir / 'per_detection_records.csv'}")
    print(f"Wrote: {output_dir / 'calibration_per_seed.csv'}")
    print(f"Wrote: {output_dir / 'calibration_summary.csv'}")
    print(f"Wrote: {output_dir / 'reliability_bins.csv'}")
    print(f"Wrote: {output_dir / 'knotfree_confidence_per_seed.csv'}")
    print(f"Wrote: {output_dir / 'knotfree_confidence_summary.csv'}")
    print(f"Wrote: {output_dir / 'knotfree_confidence_bins.csv'}")
    print(f"Wrote: {output_dir / 'clean_max_confidence_per_image.csv'}")
    print(f"Wrote: {output_dir / 'clean_max_confidence_per_seed.csv'}")
    print(f"Wrote: {output_dir / 'clean_max_confidence_summary.csv'}")
    print(f"Wrote: {output_dir / 'clean_max_confidence_cdf.csv'}")
    print(f"Wrote: {output_dir / 'reliability_curve.pdf'}")
    print(f"Wrote: {output_dir / 'reliability_curve.png'}")
    print(f"Wrote: {output_dir / 'knotfree_confidence_histogram.pdf'}")
    print(f"Wrote: {output_dir / 'knotfree_confidence_histogram.png'}")
    print(f"Wrote: {output_dir / 'clean_max_confidence_cdf.pdf'}")
    print(f"Wrote: {output_dir / 'clean_max_confidence_cdf.png'}")
    print_summary(summary_rows)
    print_knotfree_summary(knotfree_summary_rows)
    print_clean_max_summary(clean_max_summary_rows)


def resolve_predictions_dir(args: argparse.Namespace, output_dir: Path) -> Path:
    if args.predictions_dir is not None:
        return args.predictions_dir.expanduser().resolve()
    corrected = PROJECT_ROOT / "results" / "negative_aware_corrected_fixed" / "predictions"
    if corrected.exists() and not args.run_inference:
        return corrected.resolve()
    return (output_dir / "predictions").resolve()


def run_inference(args: argparse.Namespace, predictions_dir: Path) -> None:
    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "threshold_sweep_inference.py"),
        "--checkpoint-root",
        str(args.checkpoint_root.expanduser()),
        "--output-dir",
        str(predictions_dir),
        "--gpus",
        str(args.gpus),
        "--variants",
        *[str(item) for item in args.variants],
        "--seeds",
        *[str(item) for item in args.seeds],
        "--conf",
        str(float(args.conf)),
        "--iou",
        str(float(args.nms_iou)),
        "--imgsz",
        str(int(args.imgsz)),
        "--batch",
        str(int(args.batch)),
        "--max-det",
        str(int(args.max_det)),
    ]
    for variant, yaml_path in read_eval_map(args.eval_map.expanduser()).items():
        if variant in set(args.variants):
            command.extend(["--variant-data-yaml", f"{variant}={yaml_path}"])
    if args.overwrite_predictions:
        command.append("--overwrite")
    print("Running corrected prediction export:")
    print(" ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"Prediction export failed with exit code {completed.returncode}")


def read_eval_map(path: Path) -> dict[str, Path]:
    if not path.exists():
        print(f"WARNING: eval map not found; inference will fall back to run configs/defaults: {path}")
        return {}
    mapping: dict[str, Path] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("dataset") != "vnwoodknot":
                continue
            variant = str(row.get("variant", "")).strip()
            data_yaml = str(row.get("data_yaml", "")).strip()
            if variant and data_yaml:
                mapping[variant] = Path(data_yaml)
    return mapping


def load_prediction_sets(
    *,
    predictions_dir: Path,
    variants: tuple[str, ...],
    seeds: tuple[int, ...],
    min_conf: float,
    allow_higher_base_conf: bool,
) -> list[dict[str, Any]]:
    wanted = {(variant, int(seed)) for variant in variants for seed in seeds}
    prediction_sets: list[dict[str, Any]] = []
    for path in sorted(predictions_dir.glob("*_predictions.json")):
        match = PREDICTION_RE.match(path.name)
        if not match:
            continue
        variant = match.group("variant")
        seed = int(match.group("seed"))
        if (variant, seed) not in wanted:
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        base_conf = float(payload.get("base_confidence_threshold", 1.0))
        if base_conf > min_conf and not allow_higher_base_conf:
            raise SystemExit(
                f"{path} was exported at base_confidence_threshold={base_conf}, above requested --conf={min_conf}. "
                "Re-run with --run-inference --overwrite-predictions, or pass --allow-higher-base-conf intentionally."
            )
        images = payload.get("images") or []
        if not images:
            print(f"WARNING: no images in {path}; skipping")
            continue
        prediction_sets.append(
            {
                "variant": str(payload.get("variant") or variant),
                "seed": int(payload.get("seed") or seed),
                "path": path,
                "images": images,
            }
        )

    found = {(item["variant"], int(item["seed"])) for item in prediction_sets}
    missing = sorted(wanted - found, key=lambda item: (variant_sort_key(item[0]), item[1]))
    for variant, seed in missing:
        print(f"WARNING: missing prediction JSON for {variant}_seed{seed}")
    return sorted(prediction_sets, key=lambda item: (variant_sort_key(item["variant"]), int(item["seed"])))


def build_detection_records(prediction_sets: list[dict[str, Any]], *, min_conf: float, iou_threshold: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prediction_set in prediction_sets:
        variant = str(prediction_set["variant"])
        seed = int(prediction_set["seed"])
        for image in prediction_set["images"]:
            rows.extend(match_image_predictions(variant, seed, image, min_conf=min_conf, iou_threshold=iou_threshold))
    return rows


def match_image_predictions(
    variant: str,
    seed: int,
    image: dict[str, Any],
    *,
    min_conf: float,
    iou_threshold: float,
) -> list[dict[str, Any]]:
    image_id = str(image.get("canonical_id") or image.get("image") or image.get("image_path"))
    is_knotfree = bool(image.get("is_knot_free", False))
    predictions = []
    for prediction in image.get("predictions", []):
        score = float(prediction.get("conf", 0.0))
        if score < min_conf:
            continue
        predictions.append(
            {
                "conf": score,
                "bbox": np.asarray(xywh_to_xyxy(prediction.get("bbox", [0, 0, 0, 0])), dtype=np.float32),
                "xywh": [float(value) for value in prediction.get("bbox", [0, 0, 0, 0])],
                "class": str(prediction.get("class", "")),
            }
        )
    predictions.sort(key=lambda item: item["conf"], reverse=True)

    gt_rows = []
    if not is_knotfree:
        for gt in image.get("gt_boxes", []):
            if len(gt) < 5:
                continue
            gt_rows.append(
                {
                    "bbox": np.asarray(xywh_to_xyxy(gt[:4]), dtype=np.float32),
                    "class": str(gt[4]),
                    "matched": False,
                }
            )

    output: list[dict[str, Any]] = []
    for prediction in predictions:
        matched_tp = 0
        if not is_knotfree and gt_rows:
            candidates = [index for index, gt in enumerate(gt_rows) if not gt["matched"] and gt["class"] == prediction["class"]]
            if candidates:
                gt_boxes = np.asarray([gt_rows[index]["bbox"] for index in candidates], dtype=np.float32).reshape(-1, 4)
                ious = box_iou(prediction["bbox"][None, :], gt_boxes)[0]
                best_local = int(np.argmax(ious))
                if float(ious[best_local]) >= iou_threshold:
                    gt_rows[candidates[best_local]]["matched"] = True
                    matched_tp = 1
        x, y, w, h = prediction["xywh"]
        output.append(
            {
                "variant": variant,
                "seed": seed,
                "image_id": image_id,
                "is_knotfree": int(is_knotfree),
                "conf": format_float(prediction["conf"]),
                "x": format_float(x),
                "y": format_float(y),
                "w": format_float(w),
                "h": format_float(h),
                "matched_TP": matched_tp,
            }
        )
    return output


def compute_calibration(records: list[dict[str, Any]], *, bins: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    pooled: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if int(record["is_knotfree"]) == 1:
            continue
        key = (str(record["variant"]), int(record["seed"]))
        grouped[key].append(record)
        pooled[str(record["variant"])].append(record)

    per_seed_rows = []
    for (variant, seed), rows in sorted(grouped.items(), key=lambda item: (variant_sort_key(item[0][0]), item[0][1])):
        metrics, _ = calibration_for_rows(rows, bins=bins)
        metrics.update({"variant": variant, "variant_label": VARIANT_LABELS.get(variant, variant), "seed": seed})
        per_seed_rows.append(metrics)

    reliability_rows = []
    for variant, rows in sorted(pooled.items(), key=lambda item: variant_sort_key(item[0])):
        _, bin_rows = calibration_for_rows(rows, bins=bins)
        for row in bin_rows:
            row.update({"variant": variant, "variant_label": VARIANT_LABELS.get(variant, variant)})
            reliability_rows.append(row)
    return per_seed_rows, reliability_rows


def calibration_for_rows(rows: list[dict[str, Any]], *, bins: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not rows:
        metrics = {
            "num_defective_detections": 0,
            "d_ece": "nan",
            "signed_gap": "nan",
            "mean_confidence": "nan",
            "empirical_precision": "nan",
        }
        return metrics, []

    confs = np.asarray([float(row["conf"]) for row in rows], dtype=np.float64)
    matched = np.asarray([int(row["matched_TP"]) for row in rows], dtype=np.float64)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)
    bin_ids = np.minimum(np.searchsorted(bin_edges, confs, side="right") - 1, bins - 1)
    bin_ids = np.maximum(bin_ids, 0)

    total = len(rows)
    d_ece = 0.0
    signed_gap = 0.0
    bin_rows: list[dict[str, Any]] = []
    for bin_id in range(bins):
        mask = bin_ids == bin_id
        count = int(np.sum(mask))
        left, right = float(bin_edges[bin_id]), float(bin_edges[bin_id + 1])
        if count:
            mean_conf = float(np.mean(confs[mask]))
            precision = float(np.mean(matched[mask]))
            abs_gap = abs(mean_conf - precision)
            gap = mean_conf - precision
            d_ece += (count / total) * abs_gap
            signed_gap += (count / total) * gap
            bin_rows.append(
                {
                    "bin": bin_id,
                    "bin_left": format_float(left),
                    "bin_right": format_float(right),
                    "n": count,
                    "mean_confidence": format_float(mean_conf),
                    "empirical_precision": format_float(precision),
                    "gap": format_float(gap),
                    "abs_gap": format_float(abs_gap),
                }
            )
        else:
            bin_rows.append(
                {
                    "bin": bin_id,
                    "bin_left": format_float(left),
                    "bin_right": format_float(right),
                    "n": 0,
                    "mean_confidence": "nan",
                    "empirical_precision": "nan",
                    "gap": "nan",
                    "abs_gap": "nan",
                }
            )

    metrics = {
        "num_defective_detections": total,
        "d_ece": format_float(d_ece),
        "signed_gap": format_float(signed_gap),
        "mean_confidence": format_float(float(np.mean(confs))),
        "empirical_precision": format_float(float(np.mean(matched))),
    }
    return metrics, bin_rows


def summarize_per_seed(per_seed_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed_rows:
        grouped[str(row["variant"])].append(row)

    output = []
    for variant, rows in sorted(grouped.items(), key=lambda item: variant_sort_key(item[0])):
        output_row: dict[str, Any] = {
            "variant": variant,
            "variant_label": VARIANT_LABELS.get(variant, variant),
            "n_seeds": len(rows),
            "seeds": " ".join(str(row["seed"]) for row in sorted(rows, key=lambda value: int(value["seed"]))),
        }
        for metric in ("num_defective_detections", "d_ece", "signed_gap", "mean_confidence", "empirical_precision"):
            values = [parse_float(row[metric]) for row in rows]
            output_row[f"{metric}_mean"] = format_float(mean(values))
            output_row[f"{metric}_std"] = format_float(std(values))
        output.append(output_row)
    return output


def compute_clean_max_confidence(
    prediction_sets: list[dict[str, Any]],
    *,
    min_conf: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Summarize the maximum prediction confidence per knot-free image.

    This is the confidence statistic a deployment threshold must exceed to keep a
    clean image quiet. Images with no exported detections receive max_confidence=0.
    """

    per_image_rows: list[dict[str, Any]] = []
    for prediction_set in prediction_sets:
        variant = str(prediction_set["variant"])
        seed = int(prediction_set["seed"])
        for image in prediction_set["images"]:
            if not bool(image.get("is_knot_free", False)):
                continue
            confidences = [
                float(prediction.get("conf", 0.0))
                for prediction in image.get("predictions", [])
                if float(prediction.get("conf", 0.0)) >= min_conf
            ]
            max_confidence = max(confidences) if confidences else 0.0
            per_image_rows.append(
                {
                    "variant": variant,
                    "variant_label": VARIANT_LABELS.get(variant, variant),
                    "seed": seed,
                    "image_id": str(image.get("canonical_id") or image.get("image") or image.get("image_path")),
                    "num_predictions": len(confidences),
                    "max_confidence": format_float(max_confidence),
                }
            )

    grouped_seed: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in per_image_rows:
        grouped_seed[(str(row["variant"]), int(row["seed"]))].append(row)

    per_seed_rows: list[dict[str, Any]] = []
    for (variant, seed), rows in sorted(grouped_seed.items(), key=lambda item: (variant_sort_key(item[0][0]), item[0][1])):
        max_values = np.asarray([parse_float(row["max_confidence"]) for row in rows], dtype=np.float64)
        pred_counts = np.asarray([parse_float(row["num_predictions"]) for row in rows], dtype=np.float64)
        per_seed_rows.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS.get(variant, variant),
                "seed": seed,
                "num_knotfree_images": len(rows),
                "images_with_predictions": int(np.sum(pred_counts > 0)) if len(pred_counts) else 0,
                "mean_max_confidence": format_float(float(np.mean(max_values)) if len(max_values) else float("nan")),
                "p90_max_confidence": format_float(float(np.percentile(max_values, 90)) if len(max_values) else float("nan")),
                "p95_max_confidence": format_float(float(np.percentile(max_values, 95)) if len(max_values) else float("nan")),
                "max_of_max_confidence": format_float(float(np.max(max_values)) if len(max_values) else float("nan")),
            }
        )

    grouped_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed_rows:
        grouped_variant[str(row["variant"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for variant, rows in sorted(grouped_variant.items(), key=lambda item: variant_sort_key(item[0])):
        output_row: dict[str, Any] = {
            "variant": variant,
            "variant_label": VARIANT_LABELS.get(variant, variant),
            "n_seeds": len(rows),
            "seeds": " ".join(str(row["seed"]) for row in sorted(rows, key=lambda value: int(value["seed"]))),
        }
        for metric in (
            "num_knotfree_images",
            "images_with_predictions",
            "mean_max_confidence",
            "p90_max_confidence",
            "p95_max_confidence",
            "max_of_max_confidence",
        ):
            values = [parse_float(row[metric]) for row in rows]
            output_row[f"{metric}_mean"] = format_float(mean(values))
            output_row[f"{metric}_std"] = format_float(std(values))
        summary_rows.append(output_row)

    cdf_rows: list[dict[str, Any]] = []
    grouped_images: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_image_rows:
        grouped_images[str(row["variant"])].append(row)
    for variant, rows in sorted(grouped_images.items(), key=lambda item: variant_sort_key(item[0])):
        max_values = np.asarray(sorted(parse_float(row["max_confidence"]) for row in rows), dtype=np.float64)
        total = len(max_values)
        for index, value in enumerate(max_values, start=1):
            cdf_rows.append(
                {
                    "variant": variant,
                    "variant_label": VARIANT_LABELS.get(variant, variant),
                    "max_confidence": format_float(float(value)),
                    "cdf": format_float(index / total if total else float("nan")),
                    "n_images": total,
                }
            )
    return per_image_rows, per_seed_rows, summary_rows, cdf_rows


def merge_clean_max_into_summary(
    calibration_rows: list[dict[str, Any]],
    clean_max_summary_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    clean_by_variant = {str(row["variant"]): row for row in clean_max_summary_rows}
    merged_rows: list[dict[str, Any]] = []
    for row in calibration_rows:
        merged = dict(row)
        clean = clean_by_variant.get(str(row["variant"]), {})
        for key, value in clean.items():
            if key in {"variant", "variant_label", "n_seeds", "seeds"}:
                continue
            merged[f"clean_{key}"] = value
        merged_rows.append(merged)
    return merged_rows


def compute_knotfree_confidence(
    prediction_sets: list[dict[str, Any]],
    *,
    min_conf: float,
    bins: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    per_seed_rows: list[dict[str, Any]] = []
    pooled_confidences: dict[str, list[float]] = defaultdict(list)
    bin_edges = np.linspace(0.0, 1.0, bins + 1)

    for prediction_set in prediction_sets:
        variant = str(prediction_set["variant"])
        seed = int(prediction_set["seed"])
        negative_images = [image for image in prediction_set["images"] if bool(image.get("is_knot_free", False))]
        per_image_counts = []
        confidences: list[float] = []
        for image in negative_images:
            image_confidences = [
                float(prediction.get("conf", 0.0))
                for prediction in image.get("predictions", [])
                if float(prediction.get("conf", 0.0)) >= min_conf
            ]
            per_image_counts.append(len(image_confidences))
            confidences.extend(image_confidences)
            pooled_confidences[variant].extend(image_confidences)

        conf_array = np.asarray(confidences, dtype=np.float64)
        count_array = np.asarray(per_image_counts, dtype=np.float64)
        per_seed_rows.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS.get(variant, variant),
                "seed": seed,
                "num_knotfree_images": len(negative_images),
                "knotfree_images_with_predictions": int(np.sum(count_array > 0)) if len(count_array) else 0,
                "knotfree_image_rate_with_predictions": format_float(float(np.mean(count_array > 0)) if len(count_array) else 0.0),
                "num_knotfree_predictions": len(confidences),
                "mean_predictions_per_knotfree_image": format_float(float(np.mean(count_array)) if len(count_array) else 0.0),
                "mean_confidence": format_float(float(np.mean(conf_array)) if len(conf_array) else float("nan")),
                "median_confidence": format_float(float(np.median(conf_array)) if len(conf_array) else float("nan")),
                "p90_confidence": format_float(float(np.percentile(conf_array, 90)) if len(conf_array) else float("nan")),
                "max_confidence": format_float(float(np.max(conf_array)) if len(conf_array) else float("nan")),
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed_rows:
        grouped[str(row["variant"])].append(row)

    summary_rows: list[dict[str, Any]] = []
    for variant, rows in sorted(grouped.items(), key=lambda item: variant_sort_key(item[0])):
        output_row: dict[str, Any] = {
            "variant": variant,
            "variant_label": VARIANT_LABELS.get(variant, variant),
            "n_seeds": len(rows),
            "seeds": " ".join(str(row["seed"]) for row in sorted(rows, key=lambda value: int(value["seed"]))),
        }
        for metric in (
            "num_knotfree_images",
            "knotfree_images_with_predictions",
            "knotfree_image_rate_with_predictions",
            "num_knotfree_predictions",
            "mean_predictions_per_knotfree_image",
            "mean_confidence",
            "median_confidence",
            "p90_confidence",
            "max_confidence",
        ):
            values = [parse_float(row[metric]) for row in rows]
            output_row[f"{metric}_mean"] = format_float(mean(values))
            output_row[f"{metric}_std"] = format_float(std(values))
        summary_rows.append(output_row)

    hist_rows: list[dict[str, Any]] = []
    for variant, confidences in sorted(pooled_confidences.items(), key=lambda item: variant_sort_key(item[0])):
        conf_array = np.asarray(confidences, dtype=np.float64)
        if len(conf_array):
            counts, _ = np.histogram(conf_array, bins=bin_edges)
        else:
            counts = np.zeros((bins,), dtype=int)
        total = int(np.sum(counts))
        for bin_id, count in enumerate(counts):
            hist_rows.append(
                {
                    "variant": variant,
                    "variant_label": VARIANT_LABELS.get(variant, variant),
                    "bin": bin_id,
                    "bin_left": format_float(float(bin_edges[bin_id])),
                    "bin_right": format_float(float(bin_edges[bin_id + 1])),
                    "n": int(count),
                    "fraction": format_float(float(count / total) if total else 0.0),
                }
            )
    return per_seed_rows, summary_rows, hist_rows


def plot_reliability(reliability_rows: list[dict[str, Any]], output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed; skipping reliability curve.")
        return

    colors = {
        "baseline": "#4C78A8",
        "p2_illumination": "#54A24B",
        "a1_crop": "#E45756",
        "a2_colorjitter": "#B279A2",
        "p4_a4_combined": "#F58518",
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in reliability_rows:
        grouped[str(row["variant"])].append(row)

    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot([0, 1], [0, 1], color="0.45", linestyle="--", linewidth=1.0, label="Perfect calibration")
    for variant in sorted(grouped, key=variant_sort_key):
        rows = [row for row in sorted(grouped[variant], key=lambda value: int(value["bin"])) if int(row["n"]) > 0]
        if not rows:
            continue
        x = [parse_float(row["mean_confidence"]) for row in rows]
        y = [parse_float(row["empirical_precision"]) for row in rows]
        sizes = [max(18.0, math.sqrt(int(row["n"])) * 7.0) for row in rows]
        ax.plot(x, y, color=colors.get(variant), linewidth=1.5, label=VARIANT_LABELS.get(variant, variant))
        ax.scatter(x, y, s=sizes, color=colors.get(variant), edgecolor="white", linewidth=0.6, zorder=3)
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical precision")
    ax.grid(True, color="0.85", linewidth=0.7)
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "reliability_curve.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "reliability_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_knotfree_confidence(hist_rows: list[dict[str, Any]], output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed; skipping knot-free confidence histogram.")
        return

    colors = {
        "baseline": "#4C78A8",
        "p2_illumination": "#54A24B",
        "a1_crop": "#E45756",
        "a2_colorjitter": "#B279A2",
        "p4_a4_combined": "#F58518",
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in hist_rows:
        grouped[str(row["variant"])].append(row)
    if not grouped:
        return

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    width = 0.014
    variants = [variant for variant in sorted(grouped, key=variant_sort_key)]
    offsets = np.linspace(-width * 2, width * 2, max(len(variants), 1))
    for offset, variant in zip(offsets, variants):
        rows = sorted(grouped[variant], key=lambda value: int(value["bin"]))
        centers = [(parse_float(row["bin_left"]) + parse_float(row["bin_right"])) / 2.0 + offset for row in rows]
        values = [int(row["n"]) for row in rows]
        ax.bar(
            centers,
            values,
            width=width,
            color=colors.get(variant),
            alpha=0.78,
            label=VARIANT_LABELS.get(variant, variant),
        )
    ax.set_xlabel("Confidence on knot-free detections")
    ax.set_ylabel("Detection count")
    ax.set_xlim(0.0, 1.0)
    ax.grid(True, axis="y", color="0.86", linewidth=0.7)
    ax.legend(loc="upper right", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "knotfree_confidence_histogram.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "knotfree_confidence_histogram.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_clean_max_confidence_cdf(cdf_rows: list[dict[str, Any]], output_dir: Path) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("WARNING: matplotlib not installed; skipping clean max-confidence CDF.")
        return

    colors = {
        "baseline": "#4C78A8",
        "p2_illumination": "#54A24B",
        "a1_crop": "#E45756",
        "a2_colorjitter": "#B279A2",
        "p4_a4_combined": "#F58518",
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in cdf_rows:
        grouped[str(row["variant"])].append(row)
    if not grouped:
        return

    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for variant in sorted(grouped, key=variant_sort_key):
        rows = sorted(grouped[variant], key=lambda value: parse_float(value["max_confidence"]))
        x = [0.0] + [parse_float(row["max_confidence"]) for row in rows]
        y = [0.0] + [parse_float(row["cdf"]) for row in rows]
        ax.plot(
            x,
            y,
            color=colors.get(variant),
            linewidth=1.8,
            drawstyle="steps-post",
            label=VARIANT_LABELS.get(variant, variant),
        )
    ax.set_xlim(0.0, 0.55)
    ax.set_ylim(0.0, 1.01)
    ax.set_xlabel("Per-clean-image maximum confidence")
    ax.set_ylabel("CDF")
    ax.grid(True, color="0.86", linewidth=0.7)
    ax.legend(loc="lower right", frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "clean_max_confidence_cdf.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "clean_max_confidence_cdf.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(summary_rows: list[dict[str, Any]]) -> None:
    print("\nDetection calibration on defective VNWoodKnot test images")
    print("Variant              D-ECE mean±std     signed gap mean±std   precision")
    for row in summary_rows:
        print(
            f"{row['variant_label']:<20s} "
            f"{parse_float(row['d_ece_mean']):.3f}±{parse_float(row['d_ece_std']):.3f}        "
            f"{parse_float(row['signed_gap_mean']):+.3f}±{parse_float(row['signed_gap_std']):.3f}         "
            f"{parse_float(row['empirical_precision_mean']):.3f}"
        )
    if not summary_rows:
        return
    worst_ece = max(summary_rows, key=lambda row: parse_float(row["d_ece_mean"]))
    worst_gap = max(summary_rows, key=lambda row: parse_float(row["signed_gap_mean"]))
    if worst_ece["variant"] != "a1_crop" or worst_gap["variant"] != "a1_crop":
        print(
            "\nNOTE: A1 crop is not worst on both calibration criteria. "
            f"Highest D-ECE: {worst_ece['variant_label']}; highest signed over-confidence gap: {worst_gap['variant_label']}."
        )


def print_knotfree_summary(summary_rows: list[dict[str, Any]]) -> None:
    print("\nKnot-free confidence profile")
    print("Variant              FP images mean     preds/image mean   mean conf")
    for row in summary_rows:
        print(
            f"{row['variant_label']:<20s} "
            f"{parse_float(row['knotfree_images_with_predictions_mean']):5.1f}             "
            f"{parse_float(row['mean_predictions_per_knotfree_image_mean']):.3f}              "
            f"{parse_float(row['mean_confidence_mean']):.3f}"
        )


def print_clean_max_summary(summary_rows: list[dict[str, Any]]) -> None:
    print("\nClean-wood per-image maximum confidence")
    print("Variant              mean max conf      p90 max conf      p95 max conf")
    for row in summary_rows:
        print(
            f"{row['variant_label']:<20s} "
            f"{parse_float(row['mean_max_confidence_mean']):.3f}±{parse_float(row['mean_max_confidence_std']):.3f}        "
            f"{parse_float(row['p90_max_confidence_mean']):.3f}±{parse_float(row['p90_max_confidence_std']):.3f}        "
            f"{parse_float(row['p95_max_confidence_mean']):.3f}±{parse_float(row['p95_max_confidence_std']):.3f}"
        )


def xywh_to_xyxy(box: Iterable[float]) -> list[float]:
    x, y, w, h = [float(value) for value in box]
    return [x, y, x + w, y + h]


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    top_left = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], 0.0, None) * np.clip(boxes1[:, 3] - boxes1[:, 1], 0.0, None)
    area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], 0.0, None) * np.clip(boxes2[:, 3] - boxes2[:, 1], 0.0, None)
    union = np.clip(area1[:, None] + area2[None, :] - inter, 1e-8, None)
    return inter / union


def variant_sort_key(variant: str) -> int:
    return VARIANT_ORDER.get(str(variant), 99)


def mean(values: Iterable[float]) -> float:
    clean = [value for value in values if value == value]
    return sum(clean) / len(clean) if clean else float("nan")


def std(values: Iterable[float]) -> float:
    clean = [value for value in values if value == value]
    if len(clean) < 2:
        return 0.0
    return statistics.stdev(clean)


def parse_float(value: Any) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return float("nan")


def format_float(value: float) -> str:
    if value != value:
        return "nan"
    return f"{float(value):.6f}"


if __name__ == "__main__":
    main()
