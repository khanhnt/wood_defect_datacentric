#!/usr/bin/env python3
"""Validation-selected negative-aware analysis for the Faster R-CNN check."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import threshold_analysis as ta  # noqa: E402


VARIANTS = ("baseline", "a1_crop", "a2_colorjitter")
VARIANT_LABELS = {
    "baseline": "Baseline",
    "a1_crop": "A1 crop",
    "a2_colorjitter": "A2 colour jitter",
}
SEEDS = (42, 43, 44)
EPSILONS = (0.0, 0.01, 0.02, 0.05)
EXPECTED_COUNTS = {
    "val": {"images": 226, "defective": 151, "clean": 75, "targets": 151},
    "test": {"images": 229, "defective": 154, "clean": 75, "targets": 155},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--val-predictions", type=Path, required=True)
    parser.add_argument("--test-predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260825)
    return parser.parse_args()


def mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return statistics.mean(values) if values else float("nan")


def std(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def count_dataset(images: list[dict[str, Any]]) -> dict[str, int]:
    defective = [image for image in images if not bool(image.get("is_knot_free", False))]
    clean = [image for image in images if bool(image.get("is_knot_free", False))]
    return {
        "images": len(images),
        "defective": len(defective),
        "clean": len(clean),
        "targets": sum(len(image.get("gt_boxes", [])) for image in defective),
    }


def load_split(path: Path, split: str) -> list[dict[str, Any]]:
    path = path.expanduser().resolve()
    prediction_sets = [
        item
        for item in ta.load_prediction_sets(path)
        if item["variant"] in VARIANTS and int(item["seed"]) in SEEDS
    ]
    found = {(item["variant"], int(item["seed"])) for item in prediction_sets}
    expected = {(variant, seed) for variant in VARIANTS for seed in SEEDS}
    if found != expected:
        raise SystemExit(f"{split}: prediction matrix mismatch; missing={sorted(expected - found)}, extra={sorted(found - expected)}")

    for item in prediction_sets:
        payload = json.loads(Path(item["path"]).read_text(encoding="utf-8"))
        if payload.get("split") != split:
            raise SystemExit(f"{item['path']}: expected split={split}, observed={payload.get('split')}")
        if payload.get("detector") != "fasterrcnn_mobilenet_v3_large_fpn":
            raise SystemExit(f"{item['path']}: unexpected detector={payload.get('detector')}")
        settings = payload.get("settings", {})
        expected_settings = {"imgsz": 1024, "conf": 0.001, "iou": 0.7, "max_det": 300, "augment": False}
        if settings != expected_settings:
            raise SystemExit(f"{item['path']}: settings mismatch; expected={expected_settings}, observed={settings}")
        observed_counts = count_dataset(item["images"])
        if observed_counts != EXPECTED_COUNTS[split]:
            raise SystemExit(
                f"{item['path']}: dataset counts mismatch; expected={EXPECTED_COUNTS[split]}, observed={observed_counts}"
            )
    return sorted(prediction_sets, key=lambda item: (VARIANTS.index(item["variant"]), int(item["seed"])))


def evaluate_grid(
    prediction_sets: list[dict[str, Any]],
    thresholds: list[float],
    split: str,
    iou_threshold: float,
) -> tuple[list[dict[str, Any]], dict[tuple[str, int, float], tuple[dict[str, Any], np.ndarray]]]:
    raw_rows = []
    indexed = {}
    for item in prediction_sets:
        for threshold in thresholds:
            row, clean_vector = ta.evaluate_prediction_set(
                item,
                threshold=threshold,
                iou_threshold=iou_threshold,
            )
            row = {"split": split, **row}
            raw_rows.append(row)
            indexed[(item["variant"], int(item["seed"]), float(threshold))] = (row, clean_vector)
    return raw_rows, indexed


def select_threshold(
    thresholds: list[float],
    indexed: dict[tuple[str, int, float], tuple[dict[str, Any], np.ndarray]],
    variant: str,
    epsilon: float,
) -> float:
    for threshold in thresholds:
        rates = [
            ta.parse_float(indexed[(variant, seed, threshold)][0]["fp_image_rate"])
            for seed in SEEDS
        ]
        if mean(rates) <= epsilon + 1e-12:
            return float(threshold)
    raise ValueError(f"No validation threshold satisfies variant={variant}, epsilon={epsilon:.2f}")


def retained_count(images: list[dict[str, Any]], threshold: float, *, clean: bool | None = None) -> int:
    selected = images
    if clean is not None:
        selected = [image for image in images if bool(image.get("is_knot_free", False)) is clean]
    return sum(
        1
        for image in selected
        for prediction in image.get("predictions", [])
        if float(prediction.get("conf", 0.0)) >= threshold
    )


def summarize_test_operating_point(
    *,
    variant: str,
    epsilon: float,
    threshold: float,
    prediction_sets: list[dict[str, Any]],
    indexed: dict[tuple[str, int, float], tuple[dict[str, Any], np.ndarray]],
    rng: np.random.Generator,
    bootstrap_samples: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    per_seed = []
    bootstrap_parts = []
    by_seed = {int(item["seed"]): item for item in prediction_sets if item["variant"] == variant}
    for seed in SEEDS:
        row, clean_vector = indexed[(variant, seed, threshold)]
        item = by_seed[seed]
        tp = int(row["tp50"])
        fp_positive = int(row["fp50_positive"])
        fp_clean = retained_count(item["images"], threshold, clean=True)
        total_fp = fp_positive + fp_clean
        per_seed.append(
            {
                "variant": variant,
                "variant_label": VARIANT_LABELS[variant],
                "seed": seed,
                "epsilon": f"{epsilon:.2f}",
                "validation_selected_threshold": f"{threshold:.2f}",
                "n_retained": retained_count(item["images"], threshold),
                "n_TP": tp,
                "n_FP_positive": fp_positive,
                "n_FP_clean": fp_clean,
                "n_FN": int(row["fn50"]),
                "precision_positive_only": ta.format_float(ta.parse_float(row["precision"])),
                "precision_including_clean_FP": ta.format_float(tp / max(tp + total_fp, 1)),
                "retained_recall": ta.format_float(ta.parse_float(row["recall"])),
                "retained_AP50": ta.format_float(ta.parse_float(row["ap50"])),
                "test_clean_FP_image_rate": ta.format_float(ta.parse_float(row["fp_image_rate"])),
                "test_clean_FP_images": int(row["knotfree_fp_images"]),
                "test_clean_images": int(row["num_knotfree_images"]),
            }
        )
        bootstrap_parts.append(
            ta.bootstrap_fp_rates(clean_vector, rng=rng, samples=bootstrap_samples)
        )

    fp_bootstrap_mean = np.mean(np.stack(bootstrap_parts, axis=0), axis=0)
    summary: dict[str, Any] = {
        "variant": variant,
        "variant_label": VARIANT_LABELS[variant],
        "epsilon": f"{epsilon:.2f}",
        "validation_selected_threshold": f"{threshold:.2f}",
        "n_seeds": len(per_seed),
    }
    metrics = (
        "precision_positive_only",
        "precision_including_clean_FP",
        "retained_recall",
        "retained_AP50",
        "test_clean_FP_image_rate",
        "test_clean_FP_images",
    )
    for metric in metrics:
        values = [float(row[metric]) for row in per_seed]
        summary[f"{metric}_mean"] = ta.format_float(mean(values))
        summary[f"{metric}_std"] = ta.format_float(std(values))
    summary["test_clean_FP_rate_ci_lower"] = ta.format_float(float(np.percentile(fp_bootstrap_mean, 2.5)))
    summary["test_clean_FP_rate_ci_upper"] = ta.format_float(float(np.percentile(fp_bootstrap_mean, 97.5)))
    summary["bootstrap_method"] = "resample_clean_images_within_seed_then_average_seed_rates"
    return per_seed, summary


def print_tables(summary_rows: list[dict[str, Any]]) -> None:
    zero_rows = [row for row in summary_rows if row["epsilon"] == "0.00"]
    print("\nVALIDATION-SELECTED ZERO-FP TEST RESULTS")
    print("Variant | tau_val | test FP rate | retained recall | retained AP50")
    for row in zero_rows:
        print(
            f"{row['variant_label']} | {row['validation_selected_threshold']} | "
            f"{row['test_clean_FP_image_rate_mean']}±{row['test_clean_FP_image_rate_std']} | "
            f"{row['retained_recall_mean']}±{row['retained_recall_std']} | "
            f"{row['retained_AP50_mean']}±{row['retained_AP50_std']}"
        )

    print("\nLATEX OPERATIONAL ROWS")
    for row in zero_rows:
        print(
            f"{row['variant_label']} & {row['validation_selected_threshold']} & "
            f"${float(row['test_clean_FP_image_rate_mean']):.3f}\\pm{float(row['test_clean_FP_image_rate_std']):.3f}$ & "
            f"${float(row['retained_recall_mean']):.3f}\\pm{float(row['retained_recall_std']):.3f}$ & "
            f"${float(row['retained_AP50_mean']):.3f}\\pm{float(row['retained_AP50_std']):.3f}$ \\\\"
        )

    print("\nLATEX SENSITIVITY ROWS (epsilon, validation tau, test AP50)")
    for variant in VARIANTS:
        rows = [row for row in summary_rows if row["variant"] == variant]
        parts = [VARIANT_LABELS[variant]]
        for epsilon in EPSILONS:
            row = next(item for item in rows if item["epsilon"] == f"{epsilon:.2f}")
            parts.extend(
                [
                    f"{epsilon:.2f}",
                    row["validation_selected_threshold"],
                    f"{float(row['retained_AP50_mean']):.3f}",
                ]
            )
        print(" & ".join(parts) + r" \\")


def main() -> None:
    args = parse_args()
    thresholds = ta.make_thresholds(args.threshold_start, args.threshold_end, args.threshold_step)
    val_sets = load_split(args.val_predictions, "val")
    test_sets = load_split(args.test_predictions, "test")

    val_ids = {str(image.get("canonical_id")) for item in val_sets for image in item["images"]}
    test_ids = {str(image.get("canonical_id")) for item in test_sets for image in item["images"]}
    overlap = val_ids & test_ids
    if overlap:
        raise SystemExit(f"Validation/test image overlap detected: {sorted(overlap)[:5]}")

    val_raw, val_indexed = evaluate_grid(val_sets, thresholds, "val", args.iou_threshold)
    test_raw, test_indexed = evaluate_grid(test_sets, thresholds, "test", args.iou_threshold)
    selected_rows = []
    per_seed_rows = []
    summary_rows = []
    rng = np.random.default_rng(args.bootstrap_seed)
    for variant in VARIANTS:
        for epsilon in EPSILONS:
            threshold = select_threshold(thresholds, val_indexed, variant, epsilon)
            val_rates = [
                ta.parse_float(val_indexed[(variant, seed, threshold)][0]["fp_image_rate"])
                for seed in SEEDS
            ]
            selected_rows.append(
                {
                    "variant": variant,
                    "variant_label": VARIANT_LABELS[variant],
                    "epsilon": f"{epsilon:.2f}",
                    "validation_selected_threshold": f"{threshold:.2f}",
                    "validation_FP_rate_mean": ta.format_float(mean(val_rates)),
                    "validation_FP_rate_std": ta.format_float(std(val_rates)),
                    "selection_rule": "lowest_grid_threshold_with_seed_mean_validation_FP_rate_at_or_below_epsilon",
                }
            )
            per_seed, summary = summarize_test_operating_point(
                variant=variant,
                epsilon=epsilon,
                threshold=threshold,
                prediction_sets=test_sets,
                indexed=test_indexed,
                rng=rng,
                bootstrap_samples=args.bootstrap_samples,
            )
            per_seed_rows.extend(per_seed)
            summary_rows.append(summary)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    ta.write_csv(val_raw, output_dir / "validation_threshold_raw.csv")
    ta.write_csv(test_raw, output_dir / "test_threshold_raw.csv")
    ta.write_csv(selected_rows, output_dir / "validation_selected_thresholds.csv")
    ta.write_csv(per_seed_rows, output_dir / "test_operating_metrics_per_seed.csv")
    ta.write_csv(summary_rows, output_dir / "test_operating_summary.csv")
    report = {
        "status": "PASS",
        "detector": "fasterrcnn_mobilenet_v3_large_fpn",
        "variants": list(VARIANTS),
        "seeds": list(SEEDS),
        "threshold_grid": thresholds,
        "epsilons": list(EPSILONS),
        "selection_split": "val",
        "evaluation_split": "test",
        "validation_test_image_overlap": 0,
        "validation_counts": EXPECTED_COUNTS["val"],
        "test_counts": EXPECTED_COUNTS["test"],
        "bootstrap_samples": int(args.bootstrap_samples),
        "bootstrap_seed": int(args.bootstrap_seed),
        "outputs": {
            "validation_threshold_raw": str(output_dir / "validation_threshold_raw.csv"),
            "test_threshold_raw": str(output_dir / "test_threshold_raw.csv"),
            "selected_thresholds": str(output_dir / "validation_selected_thresholds.csv"),
            "test_per_seed": str(output_dir / "test_operating_metrics_per_seed.csv"),
            "test_summary": str(output_dir / "test_operating_summary.csv"),
        },
    }
    (output_dir / "analysis_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print("REUSED EVALUATION CODE")
    print(f"- prediction loader: {Path(ta.__file__).resolve()}::load_prediction_sets")
    print(f"- retained AP50 evaluator: {Path(ta.__file__).resolve()}::evaluate_prediction_set/evaluate_class")
    print("- threshold selection: validation only; the selected threshold is held fixed on test")
    print(f"- validation/test image overlap: {len(overlap)}")
    print_tables(summary_rows)
    print(f"\nANALYSIS: PASS; outputs={output_dir}")


if __name__ == "__main__":
    main()
