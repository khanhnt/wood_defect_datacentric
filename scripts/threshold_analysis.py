#!/usr/bin/env python3
"""Analyze VNWoodKnot threshold-sweep predictions and bootstrap FP-rate CIs."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable

import numpy as np


VARIANT_LABELS = {
    "baseline": "Baseline",
    "p2_illumination": "P2 illumination",
    "a1_crop": "A1 crop",
    "a2_colorjitter": "A2 color jitter",
    "p4_a4_combined": "P4+A4 combined",
}
VARIANT_ORDER = {variant: index for index, variant in enumerate(VARIANT_LABELS)}
EXPECTED_SEEDS = {42, 43, 44}
PREDICTION_RE = re.compile(r"^(?P<variant>.+)_seed(?P<seed>\d+)_predictions\.json$")
METRICS = ("ap50", "precision", "recall", "fp_image_rate", "mean_preds_per_knotfree", "mean_conf_fp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=Path("results/negative_aware/predictions"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/negative_aware"))
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260606)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_dir = args.predictions_dir.expanduser().resolve()
    if not predictions_dir.exists():
        raise SystemExit(f"Predictions directory does not exist: {predictions_dir}")

    output_dir = args.output_dir.expanduser().resolve()
    threshold_dir = output_dir / "threshold_sweep"
    bootstrap_dir = output_dir / "bootstrap"
    threshold_dir.mkdir(parents=True, exist_ok=True)
    bootstrap_dir.mkdir(parents=True, exist_ok=True)

    thresholds = make_thresholds(args.threshold_start, args.threshold_end, args.threshold_step)
    prediction_sets = load_prediction_sets(predictions_dir)
    if not prediction_sets:
        raise SystemExit(f"No prediction JSON files found in: {predictions_dir}")

    rng = np.random.default_rng(args.bootstrap_seed)
    raw_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []
    pooled_bootstrap: dict[tuple[str, float], list[np.ndarray]] = defaultdict(list)

    for prediction_set in sorted(prediction_sets, key=prediction_sort_key):
        print(f"Analyzing {prediction_set['variant']}_seed{prediction_set['seed']}")
        for threshold in thresholds:
            metrics, negative_vector = evaluate_prediction_set(
                prediction_set,
                threshold=threshold,
                iou_threshold=args.iou_threshold,
            )
            raw_rows.append(metrics)
            bootstrap_distribution = bootstrap_fp_rates(negative_vector, rng=rng, samples=args.bootstrap_samples)
            pooled_bootstrap[(prediction_set["variant"], threshold)].append(bootstrap_distribution)
            bootstrap_rows.append(
                {
                    "variant": prediction_set["variant"],
                    "seed": prediction_set["seed"],
                    "threshold": fmt_threshold(threshold),
                    "fp_rate": format_float(metrics["fp_image_rate"]),
                    "ci_lower": format_float(percentile(bootstrap_distribution, 2.5)),
                    "ci_upper": format_float(percentile(bootstrap_distribution, 97.5)),
                }
            )

    summary_rows = aggregate_raw_rows(raw_rows, pooled_bootstrap)
    bootstrap_summary_rows = build_bootstrap_summary(raw_rows, pooled_bootstrap, key_thresholds=(0.25, 0.50, 0.75))
    sweet_spots = find_operational_sweet_spots(raw_rows)

    write_csv(raw_rows, threshold_dir / "raw_data.csv")
    write_csv(summary_rows, threshold_dir / "summary_aggregated.csv")
    write_csv(bootstrap_rows, bootstrap_dir / "bootstrap_ci_results.csv")
    write_csv(bootstrap_summary_rows, bootstrap_dir / "bootstrap_summary_table.csv")
    write_csv(sweet_spots, output_dir / "operational_sweet_spots.csv")

    print(f"Wrote: {threshold_dir / 'raw_data.csv'}")
    print(f"Wrote: {threshold_dir / 'summary_aggregated.csv'}")
    print(f"Wrote: {bootstrap_dir / 'bootstrap_ci_results.csv'}")
    print(f"Wrote: {bootstrap_dir / 'bootstrap_summary_table.csv'}")
    print(f"Wrote: {output_dir / 'operational_sweet_spots.csv'}")
    print_completion_warnings(raw_rows)
    print_sweet_spots(sweet_spots)


def load_prediction_sets(predictions_dir: Path) -> list[dict[str, Any]]:
    prediction_sets = []
    for path in sorted(predictions_dir.glob("*_predictions.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        match = PREDICTION_RE.match(path.name)
        variant = payload.get("variant") or (match.group("variant") if match else None)
        seed = payload.get("seed") or (int(match.group("seed")) if match else None)
        if variant is None or seed is None:
            print(f"WARNING: could not infer variant/seed from {path}; skipping")
            continue
        images = payload.get("images") or []
        if not images:
            print(f"WARNING: no images in {path}; skipping")
            continue
        prediction_sets.append(
            {
                "variant": str(variant),
                "seed": int(seed),
                "path": path,
                "class_names": tuple(payload.get("class_names") or infer_class_names(images)),
                "images": images,
            }
        )
    return prediction_sets


def evaluate_prediction_set(prediction_set: dict[str, Any], *, threshold: float, iou_threshold: float) -> tuple[dict[str, Any], np.ndarray]:
    images = prediction_set["images"]
    class_names = tuple(prediction_set["class_names"])
    positive_images = [image for image in images if not bool(image.get("is_knot_free", False))]
    negative_images = [image for image in images if bool(image.get("is_knot_free", False))]

    class_results = [
        evaluate_class(positive_images, class_name=class_name, threshold=threshold, iou_threshold=iou_threshold)
        for class_name in class_names
    ]
    ap_values = [result["ap"] for result in class_results if not math.isnan(result["ap"])]
    tp_total = sum(int(result["tp"]) for result in class_results)
    fp_total = sum(int(result["fp"]) for result in class_results)
    fn_total = sum(int(result["fn"]) for result in class_results)
    precision = tp_total / max(tp_total + fp_total, 1)
    recall = tp_total / max(tp_total + fn_total, 1)

    negative_counts = []
    negative_confidences: list[float] = []
    binary = []
    for image in negative_images:
        predictions = [pred for pred in image.get("predictions", []) if float(pred.get("conf", 0.0)) >= threshold]
        negative_counts.append(len(predictions))
        binary.append(1 if predictions else 0)
        negative_confidences.extend(float(pred.get("conf", 0.0)) for pred in predictions)
    negative_vector = np.asarray(binary, dtype=np.float32)

    row = {
        "variant": prediction_set["variant"],
        "seed": prediction_set["seed"],
        "threshold": fmt_threshold(threshold),
        "ap50": format_float(float(np.mean(ap_values)) if ap_values else 0.0),
        "precision": format_float(precision),
        "recall": format_float(recall),
        "fp_image_rate": format_float(float(np.mean(negative_vector)) if len(negative_vector) else 0.0),
        "mean_preds_per_knotfree": format_float(float(np.mean(negative_counts)) if negative_counts else 0.0),
        "mean_conf_fp": format_float(float(np.mean(negative_confidences)) if negative_confidences else float("nan")),
        "num_positive_images": len(positive_images),
        "num_knotfree_images": len(negative_images),
        "knotfree_fp_images": int(np.sum(negative_vector)) if len(negative_vector) else 0,
        "tp50": tp_total,
        "fp50_positive": fp_total,
        "fn50": fn_total,
    }
    return row, negative_vector


def evaluate_class(images: list[dict[str, Any]], *, class_name: str, threshold: float, iou_threshold: float) -> dict[str, float]:
    gt_by_image: dict[int, np.ndarray] = {}
    matched_by_image: dict[int, np.ndarray] = {}
    num_gt = 0
    pred_rows: list[tuple[int, float, np.ndarray]] = []

    for image_index, image in enumerate(images):
        gt_boxes = np.asarray(
            [xywh_to_xyxy(row[:4]) for row in image.get("gt_boxes", []) if str(row[4]) == class_name],
            dtype=np.float32,
        ).reshape(-1, 4)
        gt_by_image[image_index] = gt_boxes
        matched_by_image[image_index] = np.zeros((len(gt_boxes),), dtype=bool)
        num_gt += len(gt_boxes)
        for prediction in image.get("predictions", []):
            if str(prediction.get("class")) != class_name:
                continue
            score = float(prediction.get("conf", 0.0))
            if score < threshold:
                continue
            pred_rows.append((image_index, score, np.asarray(xywh_to_xyxy(prediction["bbox"]), dtype=np.float32)))

    pred_rows.sort(key=lambda item: item[1], reverse=True)
    if not pred_rows:
        return {"ap": 0.0 if num_gt else float("nan"), "tp": 0, "fp": 0, "fn": num_gt}

    tp = np.zeros((len(pred_rows),), dtype=np.float32)
    fp = np.zeros((len(pred_rows),), dtype=np.float32)
    for index, (image_index, _, pred_box) in enumerate(pred_rows):
        gt_boxes = gt_by_image[image_index]
        if len(gt_boxes) == 0:
            fp[index] = 1.0
            continue
        ious = box_iou(pred_box[None, :], gt_boxes)[0]
        best_idx = int(np.argmax(ious))
        best_iou = float(ious[best_idx])
        if best_iou >= iou_threshold and not matched_by_image[image_index][best_idx]:
            matched_by_image[image_index][best_idx] = True
            tp[index] = 1.0
        else:
            fp[index] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precision_curve = tp_cum / np.clip(tp_cum + fp_cum, 1e-8, None)
    recall_curve = tp_cum / max(float(num_gt), 1e-8)
    ap = average_precision(recall_curve, precision_curve) if num_gt else float("nan")
    tp_total = int(tp.sum())
    fp_total = int(fp.sum())
    fn_total = int(max(num_gt - tp_total, 0))
    return {"ap": float(ap), "tp": tp_total, "fp": fp_total, "fn": fn_total}


def aggregate_raw_rows(raw_rows: list[dict[str, Any]], pooled_bootstrap: dict[tuple[str, float], list[np.ndarray]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw_rows:
        grouped[(row["variant"], row["threshold"])].append(row)

    summary_rows = []
    for (variant, threshold_text), rows in sorted(grouped.items(), key=lambda item: (variant_sort_key(item[0][0]), float(item[0][1]))):
        threshold = float(threshold_text)
        row: dict[str, Any] = {
            "variant": variant,
            "variant_label": VARIANT_LABELS.get(variant, variant),
            "threshold": threshold_text,
            "n_seeds": len(rows),
            "seeds": " ".join(str(item["seed"]) for item in sorted(rows, key=lambda value: int(value["seed"]))),
        }
        for metric in METRICS:
            values = [parse_float(item[metric]) for item in rows]
            row[f"{metric}_mean"] = format_float(mean(values))
            row[f"{metric}_std"] = format_float(std(values))
        pooled = concatenate_bootstrap(pooled_bootstrap.get((variant, threshold), []))
        row["fp_image_rate_ci_lower"] = format_float(percentile(pooled, 2.5))
        row["fp_image_rate_ci_upper"] = format_float(percentile(pooled, 97.5))
        row["num_knotfree_images_mean"] = format_float(mean(parse_float(item["num_knotfree_images"]) for item in rows))
        row["knotfree_fp_images_mean"] = format_float(mean(parse_float(item["knotfree_fp_images"]) for item in rows))
        summary_rows.append(row)
    return summary_rows


def build_bootstrap_summary(
    raw_rows: list[dict[str, Any]],
    pooled_bootstrap: dict[tuple[str, float], list[np.ndarray]],
    *,
    key_thresholds: Iterable[float],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    wanted = {fmt_threshold(threshold) for threshold in key_thresholds}
    for row in raw_rows:
        if row["threshold"] in wanted:
            grouped[(row["variant"], row["threshold"])].append(row)

    output = []
    for (variant, threshold_text), rows in sorted(grouped.items(), key=lambda item: (variant_sort_key(item[0][0]), float(item[0][1]))):
        threshold = float(threshold_text)
        fp_rates = [parse_float(row["fp_image_rate"]) for row in rows]
        pooled = concatenate_bootstrap(pooled_bootstrap.get((variant, threshold), []))
        output.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS.get(variant, variant),
                "threshold": threshold_text,
                "fp_rate_mean": format_float(mean(fp_rates)),
                "fp_rate_std": format_float(std(fp_rates)),
                "ci_lower": format_float(percentile(pooled, 2.5)),
                "ci_upper": format_float(percentile(pooled, 97.5)),
                "knotfree_fp_images_mean": format_float(mean(parse_float(row["knotfree_fp_images"]) for row in rows)),
                "num_knotfree_images": format_float(mean(parse_float(row["num_knotfree_images"]) for row in rows)),
            }
        )
    return output


def find_operational_sweet_spots(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for row in raw_rows:
        grouped[row["variant"]][row["threshold"]].append(row)

    output = []
    for variant in sorted(grouped, key=variant_sort_key):
        selected_rows: list[dict[str, Any]] | None = None
        selected_threshold = ""
        for threshold_text in sorted(grouped[variant], key=float):
            rows = grouped[variant][threshold_text]
            seeds = {int(row["seed"]) for row in rows}
            if seeds != EXPECTED_SEEDS:
                continue
            if all(parse_float(row["fp_image_rate"]) == 0.0 for row in rows):
                selected_rows = rows
                selected_threshold = threshold_text
                break
        if selected_rows is None:
            output.append(
                {
                    "variant": variant,
                    "variant_label": VARIANT_LABELS.get(variant, variant),
                    "zero_fp_threshold": "",
                    "recall_at_that_threshold": "",
                    "ap50_at_that_threshold": "",
                    "n_seeds": 0,
                }
            )
            continue
        output.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS.get(variant, variant),
                "zero_fp_threshold": selected_threshold,
                "recall_at_that_threshold": format_float(mean(parse_float(row["recall"]) for row in selected_rows)),
                "ap50_at_that_threshold": format_float(mean(parse_float(row["ap50"]) for row in selected_rows)),
                "n_seeds": len(selected_rows),
            }
        )
    return output


def bootstrap_fp_rates(binary: np.ndarray, *, rng: np.random.Generator, samples: int) -> np.ndarray:
    if len(binary) == 0:
        return np.asarray([float("nan")])
    indices = rng.integers(0, len(binary), size=(int(samples), len(binary)))
    return binary[indices].mean(axis=1)


def infer_class_names(images: list[dict[str, Any]]) -> list[str]:
    names = set()
    for image in images:
        for row in image.get("gt_boxes", []):
            if len(row) >= 5:
                names.add(str(row[4]))
        for prediction in image.get("predictions", []):
            if "class" in prediction:
                names.add(str(prediction["class"]))
    return sorted(names)


def make_thresholds(start: float, end: float, step: float) -> list[float]:
    count = int(round((end - start) / step)) + 1
    return [round(start + index * step, 2) for index in range(count)]


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


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(mpre) - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    change = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1]))


def prediction_sort_key(item: dict[str, Any]) -> tuple[int, int]:
    return variant_sort_key(item["variant"]), int(item["seed"])


def variant_sort_key(variant: str) -> int:
    return VARIANT_ORDER.get(str(variant), 99)


def concatenate_bootstrap(parts: list[np.ndarray]) -> np.ndarray:
    valid = [part for part in parts if len(part)]
    return np.concatenate(valid) if valid else np.asarray([float("nan")])


def mean(values: Iterable[float]) -> float:
    values = [value for value in values if value == value]
    return sum(values) / len(values) if values else float("nan")


def std(values: Iterable[float]) -> float:
    values = [value for value in values if value == value]
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def percentile(values: np.ndarray, q: float) -> float:
    if values.size == 0 or np.all(np.isnan(values)):
        return float("nan")
    return float(np.nanpercentile(values, q))


def parse_float(value: Any) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return float("nan")


def format_float(value: float) -> str:
    if value != value:
        return "nan"
    return f"{float(value):.6f}"


def fmt_threshold(value: float) -> str:
    return f"{float(value):.2f}"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_completion_warnings(raw_rows: list[dict[str, Any]]) -> None:
    seen: dict[str, set[int]] = defaultdict(set)
    for row in raw_rows:
        seen[row["variant"]].add(int(row["seed"]))
    for variant in sorted(VARIANT_LABELS, key=variant_sort_key):
        seeds = seen.get(variant, set())
        if seeds != EXPECTED_SEEDS:
            print(f"WARNING: {variant} has seeds {sorted(seeds)}; expected {sorted(EXPECTED_SEEDS)}")


def print_sweet_spots(rows: list[dict[str, Any]]) -> None:
    print("\nOperational sweet spots:")
    for row in rows:
        threshold = row["zero_fp_threshold"] or "n/a"
        recall = row["recall_at_that_threshold"] or "n/a"
        ap50 = row["ap50_at_that_threshold"] or "n/a"
        print(f"- {row['variant_label']}: zero-FP threshold={threshold}, recall={recall}, AP50={ap50}")


if __name__ == "__main__":
    main()
