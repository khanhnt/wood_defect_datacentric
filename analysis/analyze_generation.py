#!/usr/bin/env python3
"""Analyze one frozen experiment generation without training or inference.

Operating thresholds are selected on validation clean images and then applied
unchanged to the held-out test split. Detection AP uses the Ultralytics 8.4.60
101-point interpolation convention and its IoU-based one-to-one matcher.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import sys
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.verify_prediction_map_reproduction import box_iou  # noqa: E402


DATASETS = ("vnwoodknot", "vsb_rarefirst")
SPLITS = ("val", "test")
SEEDS = (42, 43, 44)
EPSILONS = (0.0, 0.01, 0.02, 0.05)
THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.05, 1.0, 0.05))
VARIANTS = (
    "baseline",
    "p1_clahe",
    "p2_illumination",
    "p3_unsharp",
    "a1_crop",
    "a2_colorjitter",
    "p4_a4_combined",
)
VARIANT_LABELS = {
    "baseline": "Baseline",
    "p1_clahe": "P1 CLAHE",
    "p2_illumination": "P2 illumination",
    "p3_unsharp": "P3 unsharp",
    "a1_crop": "A1 crop",
    "a2_colorjitter": "A2 colour jitter",
    "p4_a4_combined": "P4+A4 combined",
}
EXPECTED_COUNTS = {
    ("vnwoodknot", "val"): (226, 151, 75),
    ("vnwoodknot", "test"): (229, 155, 75),
    ("vsb_rarefirst", "val"): (977, 1146, 276),
    ("vsb_rarefirst", "test"): (972, 1173, 276),
    ("vsb_strict_clean", "val"): (2988, 0, 2988),
    ("vsb_strict_clean", "test"): (2988, 0, 2988),
}
PREDICTION_RE = re.compile(r"^(?P<variant>.+)_seed(?P<seed>\d+)_predictions\.json$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation-root",
        type=Path,
        default=PROJECT_ROOT / "revised" / "generations" / "access_r1_g2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "revised" / "analysis" / "access_r1_g2",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260817)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--base-conf", type=float, default=0.001)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_root = args.generation_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not generation_root.exists():
        raise SystemExit(f"Generation root does not exist: {generation_root}")
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists: {output_dir}. Use --overwrite intentionally.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    print("GENERATION ANALYSIS")
    print(f"- frozen input: {generation_root}")
    print(f"- output: {output_dir}")
    print("- threshold rule: select on validation clean images; apply unchanged to test")
    print("- TP assignment: saved DetectionValidator TP mask at the 0.001 export floor")
    print("- AP: Ultralytics 8.4.60 compute_ap, 101-point interpolation")

    fair_rows = read_csv(generation_root / "fair_eval" / "fair_metrics.csv")
    fair_summary = aggregate_fair_metrics(fair_rows)
    write_csv(output_dir / "fair_metrics_per_seed.csv", fair_rows)
    write_csv(output_dir / "fair_metrics_summary.csv", fair_summary)

    payloads = load_primary_payloads(generation_root)
    verify_payload_inventory(payloads)
    reproduction_rows = verify_base_ap_reproduction(payloads, tolerance=2e-6)
    write_csv(output_dir / "base_ap_reproduction.csv", reproduction_rows)

    clean_sweep_rows, clean_vectors = build_clean_sweeps(payloads)
    write_csv(output_dir / "clean_fp_sweep_per_seed.csv", clean_sweep_rows)
    clean_sweep_summary = summarize_clean_sweeps(
        clean_sweep_rows,
        clean_vectors,
        samples=int(args.bootstrap_samples),
        seed=int(args.bootstrap_seed),
    )
    write_csv(output_dir / "clean_fp_sweep_summary.csv", clean_sweep_summary)

    selections = select_validation_thresholds(clean_sweep_rows)
    write_csv(output_dir / "validation_threshold_selection.csv", selections)
    operational_per_seed = evaluate_locked_test(payloads, selections, iou=float(args.match_iou))
    write_csv(output_dir / "locked_test_operating_points_per_seed.csv", operational_per_seed)
    operational_summary = summarize_operating_points(operational_per_seed, clean_sweep_summary)
    write_csv(output_dir / "locked_test_operating_points_summary.csv", operational_summary)
    sensitivity_summary = [row for row in operational_summary if float(row["epsilon"]) in EPSILONS]
    write_csv(output_dir / "locked_test_sensitivity_summary.csv", sensitivity_summary)

    calibration_per_seed, reliability_rows = analyze_calibration(payloads, bins=10, iou=float(args.match_iou))
    calibration_summary = summarize_calibration(calibration_per_seed)
    write_csv(output_dir / "calibration_per_seed.csv", calibration_per_seed)
    write_csv(output_dir / "calibration_summary.csv", calibration_summary)
    write_csv(output_dir / "reliability_bins.csv", reliability_rows)

    clean_max_rows, clean_max_seed, clean_max_summary, clean_cdf = analyze_clean_max(payloads)
    write_csv(output_dir / "clean_max_confidence_per_image.csv", clean_max_rows)
    write_csv(output_dir / "clean_max_confidence_per_seed.csv", clean_max_seed)
    write_csv(output_dir / "clean_max_confidence_summary.csv", clean_max_summary)
    write_csv(output_dir / "clean_max_confidence_cdf.csv", clean_cdf)

    deprecated_source = generation_root / "deprecated_audit" / "comparison"
    deprecated_dir = output_dir / "deprecated_impact"
    deprecated_dir.mkdir()
    for name in ("deprecated_vs_corrected_per_seed.csv", "deprecated_vs_corrected_summary.csv"):
        source = deprecated_source / name
        if source.exists():
            shutil.copy2(source, deprecated_dir / name)

    write_latex_tables(output_dir, fair_summary, operational_summary, calibration_summary, clean_max_summary)
    write_plots(output_dir, clean_sweep_summary, operational_summary, clean_cdf)
    report = build_report(
        generation_root,
        fair_summary,
        selections,
        operational_summary,
        calibration_summary,
        clean_max_summary,
        reproduction_rows,
        generation_root / "deprecated_audit" / "comparison" / "deprecated_vs_corrected_summary.csv",
        int(args.bootstrap_samples),
    )
    (output_dir / "ANALYSIS_REPORT.md").write_text(report, encoding="utf-8")
    payload = {
        "generation_root": str(generation_root),
        "output_dir": str(output_dir),
        "threshold_grid": THRESHOLDS,
        "epsilons": EPSILONS,
        "selection_rule": "lowest validation threshold whose seed-mean clean-image FP rate is <= epsilon; epsilon=0 additionally requires every seed to be zero",
        "base_ap_reproduction_max_abs_delta": max(abs(float(row["delta_mAP50"])) for row in reproduction_rows),
        "validation_thresholds": selections,
        "locked_test_summary": operational_summary,
    }
    (output_dir / "generation_analysis.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print_console_summary(operational_summary, reproduction_rows)
    print(f"Wrote: {output_dir / 'ANALYSIS_REPORT.md'}")


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Missing required CSV: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values if not math.isnan(float(value))]
    return statistics.mean(values) if values else float("nan")


def std(values: Iterable[float]) -> float:
    values = [float(value) for value in values if not math.isnan(float(value))]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def aggregate_fair_metrics(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["variant"], row["split"])].append(row)
    output = []
    for key, items in sorted(grouped.items(), key=lambda item: group_sort(item[0])):
        row: dict[str, Any] = {
            "dataset": key[0],
            "variant": key[1],
            "variant_label": VARIANT_LABELS[key[1]],
            "split": key[2],
            "n_seeds": len(items),
            "n_images": int(items[0]["n_images"]),
            "n_instances": int(items[0]["n_instances"]),
        }
        for metric in ("precision", "recall", "mAP50", "mAP50_95"):
            values = [float(item[metric]) for item in items]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = std(values)
        output.append(row)
    return output


def group_sort(key: tuple[str, str, str]) -> tuple[int, int, int]:
    return DATASETS.index(key[0]), VARIANTS.index(key[1]), SPLITS.index(key[2])


def load_primary_payloads(root: Path) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    output: dict[tuple[str, str, int, str], dict[str, Any]] = {}
    for path in sorted((root / "predictions").rglob("*_predictions.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = str(payload["dataset"])
        key = (dataset, str(payload["variant"]), int(payload["seed"]), str(payload["split"]))
        payload["_path"] = str(path)
        output[key] = payload
    return output


def verify_payload_inventory(payloads: dict[tuple[str, str, int, str], dict[str, Any]]) -> None:
    expected = {
        (dataset, variant, seed, split)
        for dataset in ("vnwoodknot", "vsb_rarefirst", "vsb_strict_clean")
        for variant in VARIANTS
        for seed in SEEDS
        for split in SPLITS
    }
    missing = sorted(expected - set(payloads))
    extra = sorted(set(payloads) - expected)
    if missing or extra:
        raise SystemExit(f"Prediction inventory mismatch: missing={missing}, extra={extra}")
    for key, payload in payloads.items():
        if payload.get("inference_path") != "ultralytics_detection_validator":
            raise SystemExit(f"Wrong inference path for {key}: {payload.get('inference_path')}")
        expected_images, expected_targets, expected_clean = EXPECTED_COUNTS[(key[0], key[3])]
        images = payload["images"]
        targets = sum(len(image.get("gt_boxes", [])) for image in images)
        clean = sum(bool(image.get("is_knot_free", False)) for image in images)
        observed = (len(images), targets, clean)
        if observed != (expected_images, expected_targets, expected_clean):
            raise SystemExit(f"Count mismatch for {key}: observed={observed}, expected={(expected_images, expected_targets, expected_clean)}")


def compute_ap_ultralytics_8460(recall: np.ndarray, precision: np.ndarray) -> float:
    """Exact compute_ap body from ultralytics 8.4.60 utils/metrics.py."""
    mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(np.interp(x, mrec, mpre), x))


def evaluate_detection_payload(payload: dict[str, Any], threshold: float, iou: float) -> dict[str, Any]:
    del iou  # TP assignment is fixed by the validator export, as in a truncated PR curve.
    class_names = tuple(payload["class_names"])
    class_ids = {name: index for index, name in enumerate(class_names)}
    tp_rows: list[np.ndarray] = []
    confidences: list[np.ndarray] = []
    pred_classes: list[np.ndarray] = []
    target_classes: list[np.ndarray] = []
    n_retained = 0
    for image in payload["images"]:
        gt_cls = np.asarray([class_ids[str(row[4])] for row in image.get("gt_boxes", [])], dtype=np.float32)
        predictions = image.get("predictions", [])
        pred_cls = np.asarray([float(row["class_id"]) for row in predictions], dtype=np.float32)
        pred_conf = np.asarray([float(row["conf"]) for row in predictions], dtype=np.float32)
        saved_tp = np.asarray(
            [[bool(int(row["validator_tp_mask"]) & (1 << bit)) for bit in range(10)] for row in predictions],
            dtype=bool,
        ).reshape(-1, 10)
        keep = pred_conf >= float(threshold)
        pred_cls, pred_conf, saved_tp = pred_cls[keep], pred_conf[keep], saved_tp[keep]
        tp_rows.append(saved_tp[:, :1])
        confidences.append(pred_conf)
        pred_classes.append(pred_cls)
        target_classes.append(gt_cls)
        n_retained += len(pred_conf)
    tp = concatenate(tp_rows, shape=(0, 1), dtype=bool)
    conf = concatenate(confidences, shape=(0,), dtype=np.float32)
    pred_cls = concatenate(pred_classes, shape=(0,), dtype=np.float32)
    target_cls = concatenate(target_classes, shape=(0,), dtype=np.float32)
    tp50 = int(tp[:, 0].sum()) if len(tp) else 0
    fp50 = int(len(tp) - tp50)
    fn50 = int(len(target_cls) - tp50)
    precision = tp50 / max(tp50 + fp50, 1)
    recall = tp50 / max(tp50 + fn50, 1)
    ap_values = []
    for class_id in np.unique(target_cls).astype(int):
        pred_mask = pred_cls == class_id
        num_labels = int((target_cls == class_id).sum())
        if not pred_mask.any():
            ap_values.append(0.0)
            continue
        order = np.argsort(-conf[pred_mask], kind="stable")
        class_tp = tp[pred_mask, 0][order].astype(float)
        tpc = np.cumsum(class_tp)
        fpc = np.cumsum(1.0 - class_tp)
        class_recall = tpc / max(num_labels, 1)
        class_precision = tpc / np.maximum(tpc + fpc, 1e-16)
        ap_values.append(compute_ap_ultralytics_8460(class_recall, class_precision))
    return {
        "n_retained": n_retained,
        "n_targets": len(target_cls),
        "tp50": tp50,
        "fp50": fp50,
        "fn50": fn50,
        "precision": precision,
        "recall": recall,
        "mAP50": mean(ap_values),
    }


def concatenate(parts: list[np.ndarray], *, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
    valid = [part for part in parts if len(part)]
    return np.concatenate(valid, axis=0) if valid else np.empty(shape, dtype=dtype)


def verify_base_ap_reproduction(
    payloads: dict[tuple[str, str, int, str], dict[str, Any]], tolerance: float
) -> list[dict[str, Any]]:
    rows = []
    for key in sorted((key for key in payloads if key[0] in DATASETS), key=prediction_sort):
        payload = payloads[key]
        metrics = evaluate_detection_payload(payload, threshold=float(payload["base_confidence_threshold"]), iou=0.50)
        expected = float(payload["validator_metrics"]["mAP50"])
        delta = metrics["mAP50"] - expected
        status = "PASS" if abs(delta) <= tolerance else "FAIL"
        rows.append(
            {
                "dataset": key[0], "variant": key[1], "seed": key[2], "split": key[3],
                "export_validator_mAP50": expected, "offline_mAP50": metrics["mAP50"],
                "delta_mAP50": delta, "status": status,
            }
        )
    failures = [row for row in rows if row["status"] != "PASS"]
    if failures:
        worst = max(failures, key=lambda row: abs(float(row["delta_mAP50"])))
        raise SystemExit(f"Base AP reproduction failed: {worst}")
    return rows


def prediction_sort(key: tuple[str, str, int, str]) -> tuple[int, int, int, int]:
    dataset_order = ("vnwoodknot", "vsb_rarefirst", "vsb_strict_clean")
    return dataset_order.index(key[0]), VARIANTS.index(key[1]), key[2], SPLITS.index(key[3])


def clean_payload_key(dataset: str, variant: str, seed: int, split: str) -> tuple[str, str, int, str]:
    clean_dataset = "vnwoodknot" if dataset == "vnwoodknot" else "vsb_strict_clean"
    return clean_dataset, variant, seed, split


def clean_images(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [image for image in payload["images"] if bool(image.get("is_knot_free", False))]


def build_clean_sweeps(
    payloads: dict[tuple[str, str, int, str], dict[str, Any]]
) -> tuple[list[dict[str, Any]], dict[tuple[str, str, int, str, float], np.ndarray]]:
    rows: list[dict[str, Any]] = []
    vectors: dict[tuple[str, str, int, str, float], np.ndarray] = {}
    for dataset in DATASETS:
        for variant in VARIANTS:
            for seed in SEEDS:
                for split in SPLITS:
                    payload = payloads[clean_payload_key(dataset, variant, seed, split)]
                    images = clean_images(payload)
                    for threshold in THRESHOLDS:
                        counts = np.asarray(
                            [sum(float(pred["conf"]) >= threshold for pred in image["predictions"]) for image in images],
                            dtype=np.int32,
                        )
                        vector = (counts > 0).astype(np.float32)
                        key = (dataset, variant, seed, split, threshold)
                        vectors[key] = vector
                        confidences = [
                            float(pred["conf"])
                            for image in images for pred in image["predictions"]
                            if float(pred["conf"]) >= threshold
                        ]
                        rows.append(
                            {
                                "dataset": dataset, "variant": variant, "variant_label": VARIANT_LABELS[variant],
                                "seed": seed, "split": split, "threshold": threshold, "n_clean_images": len(images),
                                "fp_images": int(vector.sum()), "fp_image_rate": float(vector.mean()),
                                "n_predictions": int(counts.sum()), "mean_predictions_per_clean_image": float(counts.mean()),
                                "mean_false_confidence": mean(confidences),
                            }
                        )
    return rows, vectors


def summarize_clean_sweeps(
    rows: list[dict[str, Any]],
    vectors: dict[tuple[str, str, int, str, float], np.ndarray],
    *,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["variant"], row["split"], float(row["threshold"]))].append(row)
    output = []
    rng = np.random.default_rng(seed)
    for key, items in sorted(grouped.items(), key=lambda item: sweep_sort(item[0])):
        stacked = np.stack([vectors[(key[0], key[1], int(item["seed"]), key[2], key[3])] for item in items])
        per_image_seed_mean = stacked.mean(axis=0)
        indices = rng.integers(0, stacked.shape[1], size=(samples, stacked.shape[1]))
        bootstrap = per_image_seed_mean[indices].mean(axis=1)
        output.append(
            {
                "dataset": key[0], "variant": key[1], "variant_label": VARIANT_LABELS[key[1]],
                "split": key[2], "threshold": key[3], "n_seeds": len(items),
                "n_clean_images": stacked.shape[1],
                "fp_image_rate_mean": mean(float(item["fp_image_rate"]) for item in items),
                "fp_image_rate_std": std(float(item["fp_image_rate"]) for item in items),
                "fp_image_rate_ci_lower": float(np.percentile(bootstrap, 2.5)),
                "fp_image_rate_ci_upper": float(np.percentile(bootstrap, 97.5)),
                "mean_predictions_per_clean_image_mean": mean(float(item["mean_predictions_per_clean_image"]) for item in items),
                "mean_predictions_per_clean_image_std": std(float(item["mean_predictions_per_clean_image"]) for item in items),
            }
        )
    return output


def sweep_sort(key: tuple[str, str, str, float]) -> tuple[int, int, int, float]:
    return DATASETS.index(key[0]), VARIANTS.index(key[1]), SPLITS.index(key[2]), key[3]


def select_validation_thresholds(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "val":
            lookup[(row["dataset"], row["variant"], float(row["threshold"]))].append(row)
    output = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            for epsilon in EPSILONS:
                selected = None
                for threshold in THRESHOLDS:
                    items = lookup[(dataset, variant, threshold)]
                    rates = [float(item["fp_image_rate"]) for item in items]
                    qualifies = all(rate == 0.0 for rate in rates) if epsilon == 0.0 else mean(rates) <= epsilon
                    if qualifies:
                        selected = (threshold, rates)
                        break
                if selected is None:
                    raise SystemExit(f"No validation threshold found for {dataset}/{variant}/epsilon={epsilon}")
                output.append(
                    {
                        "dataset": dataset, "variant": variant, "variant_label": VARIANT_LABELS[variant],
                        "epsilon": epsilon, "selected_threshold": selected[0],
                        "validation_fp_rate_mean": mean(selected[1]), "validation_fp_rate_std": std(selected[1]),
                        "selection_rule": "all_seeds_zero" if epsilon == 0.0 else "seed_mean_le_epsilon",
                    }
                )
    return output


def evaluate_locked_test(
    payloads: dict[tuple[str, str, int, str], dict[str, Any]],
    selections: list[dict[str, Any]],
    *,
    iou: float,
) -> list[dict[str, Any]]:
    output = []
    for selection in selections:
        dataset, variant = selection["dataset"], selection["variant"]
        threshold = float(selection["selected_threshold"])
        for seed in SEEDS:
            detection = payloads[(dataset, variant, seed, "test")]
            clean = payloads[clean_payload_key(dataset, variant, seed, "test")]
            metrics = evaluate_detection_payload(detection, threshold=threshold, iou=iou)
            images = clean_images(clean)
            vector = np.asarray(
                [any(float(pred["conf"]) >= threshold for pred in image["predictions"]) for image in images],
                dtype=float,
            )
            output.append(
                {
                    "dataset": dataset, "variant": variant, "variant_label": VARIANT_LABELS[variant],
                    "seed": seed, "epsilon": selection["epsilon"], "selected_threshold": threshold,
                    "validation_fp_rate_mean": selection["validation_fp_rate_mean"],
                    "test_n_clean_images": len(images), "test_fp_images": int(vector.sum()),
                    "test_fp_image_rate": float(vector.mean()), **metrics,
                }
            )
    return output


def summarize_operating_points(
    rows: list[dict[str, Any]], clean_sweep_summary: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    ci_lookup = {
        (row["dataset"], row["variant"], float(row["threshold"])): row
        for row in clean_sweep_summary if row["split"] == "test"
    }
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["variant"], float(row["epsilon"]))].append(row)
    output = []
    for key, items in sorted(grouped.items(), key=lambda item: sensitivity_sort(item[0])):
        threshold = float(items[0]["selected_threshold"])
        ci = ci_lookup[(key[0], key[1], threshold)]
        row: dict[str, Any] = {
            "dataset": key[0], "variant": key[1], "variant_label": VARIANT_LABELS[key[1]],
            "epsilon": key[2], "selected_threshold": threshold,
            "validation_fp_rate_mean": float(items[0]["validation_fp_rate_mean"]),
            "test_n_clean_images": int(items[0]["test_n_clean_images"]),
            "test_fp_image_rate_mean": mean(float(item["test_fp_image_rate"]) for item in items),
            "test_fp_image_rate_std": std(float(item["test_fp_image_rate"]) for item in items),
            "test_fp_image_rate_ci_lower": ci["fp_image_rate_ci_lower"],
            "test_fp_image_rate_ci_upper": ci["fp_image_rate_ci_upper"],
        }
        for metric in ("precision", "recall", "mAP50", "tp50", "fp50", "fn50", "n_retained"):
            values = [float(item[metric]) for item in items]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = std(values)
        output.append(row)
    return output


def sensitivity_sort(key: tuple[str, str, float]) -> tuple[int, int, float]:
    return DATASETS.index(key[0]), VARIANTS.index(key[1]), key[2]


def analyze_calibration(
    payloads: dict[tuple[str, str, int, str], dict[str, Any]], *, bins: int, iou: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    per_seed = []
    pooled: dict[tuple[str, str], list[tuple[float, int]]] = defaultdict(list)
    for dataset in DATASETS:
        for variant in VARIANTS:
            for seed in SEEDS:
                payload = payloads[(dataset, variant, seed, "test")]
                rows: list[tuple[float, int]] = []
                for image in payload["images"]:
                    if bool(image.get("is_knot_free", False)) or not image.get("gt_boxes"):
                        continue
                    rows.extend(confidence_greedy_records(image, float(payload["base_confidence_threshold"]), iou))
                metrics, _ = calibration_metrics(rows, bins=bins)
                per_seed.append({"dataset": dataset, "variant": variant, "variant_label": VARIANT_LABELS[variant], "seed": seed, **metrics})
                pooled[(dataset, variant)].extend(rows)
    reliability = []
    for key, rows in sorted(pooled.items(), key=lambda item: (DATASETS.index(item[0][0]), VARIANTS.index(item[0][1]))):
        _, bin_rows = calibration_metrics(rows, bins=bins)
        for row in bin_rows:
            reliability.append({"dataset": key[0], "variant": key[1], "variant_label": VARIANT_LABELS[key[1]], **row})
    return per_seed, reliability


def confidence_greedy_records(image: dict[str, Any], min_conf: float, iou: float) -> list[tuple[float, int]]:
    classes = sorted({str(row[4]) for row in image["gt_boxes"]} | {str(pred["class"]) for pred in image["predictions"]})
    class_ids = {name: index for index, name in enumerate(classes)}
    gt_boxes = np.asarray([xywh_to_xyxy(row[:4]) for row in image["gt_boxes"]], dtype=np.float32).reshape(-1, 4)
    gt_cls = np.asarray([class_ids[str(row[4])] for row in image["gt_boxes"]], dtype=int)
    predictions = sorted(
        [pred for pred in image["predictions"] if float(pred["conf"]) >= min_conf],
        key=lambda pred: float(pred["conf"]), reverse=True,
    )
    used: set[int] = set()
    output = []
    for pred in predictions:
        box = np.asarray(xywh_to_xyxy(pred["bbox"]), dtype=np.float32)[None, :]
        pred_class = class_ids[str(pred["class"])]
        ious = box_iou(gt_boxes, box)[:, 0] if len(gt_boxes) else np.asarray([])
        candidates = [idx for idx in range(len(gt_boxes)) if idx not in used and gt_cls[idx] == pred_class and ious[idx] >= iou]
        matched = 0
        if candidates:
            best = max(candidates, key=lambda idx: float(ious[idx]))
            used.add(best)
            matched = 1
        output.append((float(pred["conf"]), matched))
    return output


def xywh_to_xyxy(values: Iterable[float]) -> list[float]:
    x, y, w, h = [float(value) for value in values]
    return [x, y, x + w, y + h]


def calibration_metrics(rows: list[tuple[float, int]], bins: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    conf = np.asarray([row[0] for row in rows], dtype=float)
    matched = np.asarray([row[1] for row in rows], dtype=float)
    d_ece = 0.0
    signed = 0.0
    bin_rows = []
    for bin_index in range(bins):
        left, right = bin_index / bins, (bin_index + 1) / bins
        mask = (conf >= left) & ((conf < right) if bin_index < bins - 1 else (conf <= right))
        count = int(mask.sum())
        mean_conf = float(conf[mask].mean()) if count else float("nan")
        precision = float(matched[mask].mean()) if count else float("nan")
        gap = mean_conf - precision if count else float("nan")
        if count:
            weight = count / max(len(conf), 1)
            d_ece += weight * abs(gap)
            signed += weight * gap
        bin_rows.append({"bin": bin_index, "bin_left": left, "bin_right": right, "n": count, "mean_confidence": mean_conf, "empirical_precision": precision, "gap": gap})
    return {
        "num_defective_detections": len(rows), "d_ece": d_ece, "signed_gap": signed,
        "mean_confidence": float(conf.mean()) if len(conf) else float("nan"),
        "empirical_precision": float(matched.mean()) if len(matched) else float("nan"),
    }, bin_rows


def summarize_calibration(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["variant"])].append(row)
    output = []
    for key, items in sorted(grouped.items(), key=lambda item: (DATASETS.index(item[0][0]), VARIANTS.index(item[0][1]))):
        row = {"dataset": key[0], "variant": key[1], "variant_label": VARIANT_LABELS[key[1]], "n_seeds": len(items)}
        for metric in ("num_defective_detections", "d_ece", "signed_gap", "mean_confidence", "empirical_precision"):
            values = [float(item[metric]) for item in items]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = std(values)
        output.append(row)
    return output


def analyze_clean_max(
    payloads: dict[tuple[str, str, int, str], dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    image_rows = []
    per_seed = []
    cdf_rows = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            pooled = []
            for seed in SEEDS:
                payload = payloads[clean_payload_key(dataset, variant, seed, "test")]
                values = []
                for image in clean_images(payload):
                    value = max((float(pred["conf"]) for pred in image["predictions"]), default=0.0)
                    values.append(value)
                    pooled.append(value)
                    image_rows.append({"dataset": dataset, "variant": variant, "seed": seed, "image_id": image.get("canonical_id", image.get("image")), "max_confidence": value})
                array = np.asarray(values)
                per_seed.append(
                    {"dataset": dataset, "variant": variant, "variant_label": VARIANT_LABELS[variant], "seed": seed,
                     "n_clean_images": len(values), "mean_max_confidence": float(array.mean()),
                     "p90_max_confidence": float(np.percentile(array, 90)), "p95_max_confidence": float(np.percentile(array, 95))}
                )
            pooled_array = np.asarray(pooled)
            for threshold in np.linspace(0, 1, 201):
                cdf_rows.append({"dataset": dataset, "variant": variant, "variant_label": VARIANT_LABELS[variant], "max_confidence": threshold, "cdf": float((pooled_array <= threshold).mean())})
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed:
        grouped[(row["dataset"], row["variant"])].append(row)
    summary = []
    for key, items in sorted(grouped.items(), key=lambda item: (DATASETS.index(item[0][0]), VARIANTS.index(item[0][1]))):
        row = {"dataset": key[0], "variant": key[1], "variant_label": VARIANT_LABELS[key[1]], "n_seeds": len(items), "n_clean_images_per_seed": items[0]["n_clean_images"]}
        for metric in ("mean_max_confidence", "p90_max_confidence", "p95_max_confidence"):
            values = [float(item[metric]) for item in items]
            row[f"{metric}_mean"] = mean(values)
            row[f"{metric}_std"] = std(values)
        summary.append(row)
    return image_rows, per_seed, summary, cdf_rows


def write_latex_tables(
    output_dir: Path,
    fair: list[dict[str, Any]],
    operational: list[dict[str, Any]],
    calibration: list[dict[str, Any]],
    clean_max: list[dict[str, Any]],
) -> None:
    latex = output_dir / "latex"
    latex.mkdir()
    for dataset in DATASETS:
        fair_rows = [row for row in fair if row["dataset"] == dataset and row["split"] == "test"]
        text = [f"{row['variant_label']} & {pm(row['precision_mean'], row['precision_std'])} & {pm(row['recall_mean'], row['recall_std'])} & {pm(row['mAP50_mean'], row['mAP50_std'])} & {pm(row['mAP50_95_mean'], row['mAP50_95_std'])} \\\\" for row in fair_rows]
        (latex / f"{dataset}_fair_test_rows.tex").write_text("\n".join(text) + "\n", encoding="utf-8")

        op_rows = [row for row in operational if row["dataset"] == dataset and float(row["epsilon"]) == 0.0]
        text = [f"{row['variant_label']} & {row['selected_threshold']:.2f} & {pm(row['test_fp_image_rate_mean'], row['test_fp_image_rate_std'])} & {pm(row['precision_mean'], row['precision_std'])} & {pm(row['recall_mean'], row['recall_std'])} & {pm(row['mAP50_mean'], row['mAP50_std'])} \\\\" for row in op_rows]
        (latex / f"{dataset}_locked_opsel_rows.tex").write_text("\n".join(text) + "\n", encoding="utf-8")

        sens_rows = [row for row in operational if row["dataset"] == dataset]
        text = [f"{row['variant_label']} & {row['epsilon']:.2f} & {row['selected_threshold']:.2f} & {row['validation_fp_rate_mean']:.3f} & {row['test_fp_image_rate_mean']:.3f} & {row['recall_mean']:.3f} & {row['mAP50_mean']:.3f} \\\\" for row in sens_rows]
        (latex / f"{dataset}_locked_sensitivity_rows.tex").write_text("\n".join(text) + "\n", encoding="utf-8")

        cal_lookup = {(row["dataset"], row["variant"]): row for row in calibration}
        clean_lookup = {(row["dataset"], row["variant"]): row for row in clean_max}
        text = []
        for variant in VARIANTS:
            cal_row, clean_row = cal_lookup[(dataset, variant)], clean_lookup[(dataset, variant)]
            text.append(f"{VARIANT_LABELS[variant]} & {cal_row['num_defective_detections_mean']:.0f}$\\pm${cal_row['num_defective_detections_std']:.0f} & {pm(cal_row['d_ece_mean'], cal_row['d_ece_std'])} & {pm(cal_row['signed_gap_mean'], cal_row['signed_gap_std'])} & {pm(clean_row['p95_max_confidence_mean'], clean_row['p95_max_confidence_std'])} \\\\")
        (latex / f"{dataset}_calibration_rows.tex").write_text("\n".join(text) + "\n", encoding="utf-8")


def pm(value: float, deviation: float) -> str:
    return f"{value:.3f}$\\pm${deviation:.3f}"


def write_plots(
    output_dir: Path,
    sweep: list[dict[str, Any]],
    operational: list[dict[str, Any]],
    cdf: list[dict[str, Any]],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir = output_dir / "figures"
    figure_dir.mkdir()
    colors = plt.get_cmap("tab10").colors
    for dataset in DATASETS:
        fig, ax = plt.subplots(figsize=(6.7, 4.1))
        for index, variant in enumerate(VARIANTS):
            rows = [row for row in sweep if row["dataset"] == dataset and row["split"] == "test" and row["variant"] == variant]
            ax.plot([row["threshold"] for row in rows], [row["fp_image_rate_mean"] for row in rows], label=VARIANT_LABELS[variant], color=colors[index])
        max_rate = max((float(row["fp_image_rate_mean"]) for row in sweep if row["dataset"] == dataset and row["split"] == "test"), default=0.0)
        ax.set(xlabel="Confidence threshold", ylabel="Clean-image FP rate", xlim=(0.05, 0.95), ylim=(0, max(max_rate * 1.08, 0.01)))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        save_figure(fig, figure_dir / f"{dataset}_locked_clean_fp_curve")

        fig, ax = plt.subplots(figsize=(6.7, 4.1))
        for index, variant in enumerate(VARIANTS):
            rows = [row for row in cdf if row["dataset"] == dataset and row["variant"] == variant]
            ax.plot([row["max_confidence"] for row in rows], [row["cdf"] for row in rows], label=VARIANT_LABELS[variant], color=colors[index])
        ax.set(xlabel="Maximum confidence per clean image", ylabel="Empirical CDF", xlim=(0, 1), ylim=(0, 1.01))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2, loc="lower right")
        save_figure(fig, figure_dir / f"{dataset}_clean_max_confidence_cdf")

        zero_rows = [row for row in operational if row["dataset"] == dataset and float(row["epsilon"]) == 0.0]
        fig, ax = plt.subplots(figsize=(6.7, 4.1))
        for index, row in enumerate(zero_rows):
            ax.scatter(row["test_fp_image_rate_mean"], row["recall_mean"], s=48, color=colors[index], label=row["variant_label"])
        max_fp = max(float(row["test_fp_image_rate_mean"]) for row in zero_rows)
        ax.set(xlabel="Held-out clean-image FP rate", ylabel="Retained recall", xlim=(0, max(max_fp * 1.20, 0.001)), ylim=(0, 1))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, ncol=2)
        save_figure(fig, figure_dir / f"{dataset}_locked_operating_points")


def save_figure(fig: Any, base: Path) -> None:
    fig.tight_layout()
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    import matplotlib.pyplot as plt
    plt.close(fig)


def build_report(
    generation_root: Path,
    fair: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    operational: list[dict[str, Any]],
    calibration: list[dict[str, Any]],
    clean_max: list[dict[str, Any]],
    reproduction: list[dict[str, Any]],
    deprecated_path: Path,
    bootstrap_samples: int,
) -> str:
    lines = [
        "# Generation Analysis Report", "",
        f"- Frozen generation: `{generation_root}`",
        "- No training or inference was run.",
        "- Thresholds were selected on validation clean images and applied unchanged to test.",
        "- Retained AP truncates the validator-exported PR ranking at the locked threshold: saved DetectionValidator TP assignments plus Ultralytics 8.4.60 101-point interpolation.",
        f"- Clean-image FP-rate confidence intervals use {bootstrap_samples:,} deterministic bootstrap resamples (images for VNWoodKnot; tiles for VSB).",
        f"- Base-AP reproduction: **PASS ({len(reproduction)}/{len(reproduction)})**, maximum absolute delta `{max(abs(float(row['delta_mAP50'])) for row in reproduction):.9f}`.",
        "", "## Standard Fair-Test Metrics", "",
    ]
    for dataset in DATASETS:
        lines.extend([f"### {dataset}", "", "| Variant | P | R | mAP50 | mAP50-95 |", "|---|---:|---:|---:|---:|"])
        for row in fair:
            if row["dataset"] == dataset and row["split"] == "test":
                lines.append(f"| {row['variant_label']} | {fmt_pm(row['precision_mean'], row['precision_std'])} | {fmt_pm(row['recall_mean'], row['recall_std'])} | {fmt_pm(row['mAP50_mean'], row['mAP50_std'])} | {fmt_pm(row['mAP50_95_mean'], row['mAP50_95_std'])} |")
        lines.append("")
    lines.extend(["## Validation-Selected Zero-FP Operating Points", "", "The zero-FP threshold is the lowest 0.05-grid threshold with zero validation clean-image FP rate in every seed.", ""])
    for dataset in DATASETS:
        lines.extend([f"### {dataset}", "", "| Variant | tau(val) | test FP rate | test P | test R | retained AP50 |", "|---|---:|---:|---:|---:|---:|"])
        for row in operational:
            if row["dataset"] == dataset and float(row["epsilon"]) == 0.0:
                lines.append(f"| {row['variant_label']} | {row['selected_threshold']:.2f} | {fmt_pm(row['test_fp_image_rate_mean'], row['test_fp_image_rate_std'])} | {fmt_pm(row['precision_mean'], row['precision_std'])} | {fmt_pm(row['recall_mean'], row['recall_std'])} | {fmt_pm(row['mAP50_mean'], row['mAP50_std'])} |")
        lines.append("")
    lines.extend(["## Calibration", "", "D-ECE is computed on detections from defective test images only. Clean-image confidence is reported separately.", "", "| Dataset | Variant | D-ECE | signed gap | clean max-conf P95 |", "|---|---|---:|---:|---:|"])
    clean_lookup = {(row["dataset"], row["variant"]): row for row in clean_max}
    for row in calibration:
        clean = clean_lookup[(row["dataset"], row["variant"])]
        lines.append(f"| {row['dataset']} | {row['variant_label']} | {fmt_pm(row['d_ece_mean'], row['d_ece_std'])} | {fmt_pm(row['signed_gap_mean'], row['signed_gap_std'])} | {fmt_pm(clean['p95_max_confidence_mean'], clean['p95_max_confidence_std'])} |")
    if deprecated_path.exists():
        deprecated = read_csv(deprecated_path)
        lines.extend(["", "## Deprecated-Checkpoint Impact", "", "Values are corrected minus deprecated on the same non-augmented fair evaluation input.", "", "| Dataset | Variant | Split | delta mAP50 | delta mAP50-95 |", "|---|---|---|---:|---:|"])
        for row in deprecated:
            lines.append(f"| {row['dataset']} | {VARIANT_LABELS[row['variant']]} | {row['split']} | {float(row['delta_mAP50_mean']):+.4f} | {float(row['delta_mAP50_95_mean']):+.4f} |")
    lines.extend(["", "## Interpretation Guardrails", "", "- The test split was not used to choose any confidence threshold.", "- VSB threshold selection and final testing use source-disjoint strict-clean halves (2,988 tiles each).", "- VSB rare-first empty tiles remain part of its standard detection evaluation; strict-clean tiles are an additional operational test.", "- Deprecated checkpoints are audit evidence only and must not populate primary manuscript tables.", ""])
    return "\n".join(lines)


def fmt_pm(value: float, deviation: float) -> str:
    return f"{value:.3f} +/- {deviation:.3f}"


def print_console_summary(rows: list[dict[str, Any]], reproduction: list[dict[str, Any]]) -> None:
    print(f"BASE AP REPRODUCTION: PASS ({len(reproduction)}/{len(reproduction)}), max delta={max(abs(float(row['delta_mAP50'])) for row in reproduction):.9f}")
    print("\nLOCKED ZERO-FP TEST RESULTS")
    for dataset in DATASETS:
        print(f"[{dataset}]")
        for row in rows:
            if row["dataset"] == dataset and float(row["epsilon"]) == 0.0:
                print(f"- {row['variant_label']}: tau={row['selected_threshold']:.2f}, test_FP={row['test_fp_image_rate_mean']:.4f}, R={row['recall_mean']:.4f}, AP50={row['mAP50_mean']:.4f}")


if __name__ == "__main__":
    main()
