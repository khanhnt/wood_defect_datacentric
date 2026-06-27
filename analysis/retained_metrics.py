#!/usr/bin/env python3
"""Audit retained recall and AP50 at VNWoodKnot operating thresholds.

The script reads the already-exported low-confidence prediction JSON files and
reuses the threshold-analysis AP estimator used by the paper's threshold tables.
It does not train models or run inference.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts import threshold_analysis as ta  # noqa: E402


VARIANTS = ("baseline", "p2_illumination", "a1_crop", "a2_colorjitter", "p4_a4_combined")
SEEDS = (42, 43, 44)
EPSILONS = (0.0, 0.01, 0.02, 0.05)
EXPECTED_OPSEL = {
    "baseline": {"tau": 0.60, "recall": 0.357, "ap50": 0.357},
    "p2_illumination": {"tau": 0.25, "recall": 0.865, "ap50": 0.780},
    "a1_crop": {"tau": 0.50, "recall": 0.551, "ap50": 0.490},
    "a2_colorjitter": {"tau": 0.20, "recall": 0.912, "ap50": 0.788},
    "p4_a4_combined": {"tau": 0.45, "recall": 0.742, "ap50": 0.680},
}
VARIANT_LABELS = {
    "baseline": "Baseline",
    "p2_illumination": "P2 illumination",
    "a1_crop": "A1 crop",
    "a2_colorjitter": "A2 colour jitter",
    "p4_a4_combined": "P4+A4 combined",
}
DEFAULT_PREDICTION_DIRS = (
    PROJECT_ROOT / "results" / "negative_aware_corrected_fixed" / "predictions",
    PROJECT_ROOT / "results" / "negative_aware_corrected" / "predictions",
    PROJECT_ROOT / "results" / "negative_aware" / "predictions",
    PROJECT_ROOT / "results 3" / "negative_aware_corrected_fixed" / "predictions",
    PROJECT_ROOT / "results 3" / "negative_aware_corrected" / "predictions",
    PROJECT_ROOT / "results 3" / "negative_aware" / "predictions",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-dir", type=Path, default=None)
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "results" / "retained_metrics_audit.json")
    parser.add_argument("--threshold-start", type=float, default=0.05)
    parser.add_argument("--threshold-end", type=float, default=0.95)
    parser.add_argument("--threshold-step", type=float, default=0.05)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--rounding-tolerance", type=float, default=0.0015)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_dir = resolve_predictions_dir(args.predictions_dir)
    thresholds = ta.make_thresholds(args.threshold_start, args.threshold_end, args.threshold_step)

    print_reuse_report(predictions_dir)
    prediction_sets = load_prediction_sets(predictions_dir)
    print_dataset_counts(prediction_sets)

    per_variant: dict[str, Any] = {}
    for variant in VARIANTS:
        variant_sets = sorted([item for item in prediction_sets if item["variant"] == variant], key=lambda item: int(item["seed"]))
        if {int(item["seed"]) for item in variant_sets} != set(SEEDS):
            found = sorted(int(item["seed"]) for item in variant_sets)
            raise SystemExit(f"Missing seeds for {variant}: found={found} expected={list(SEEDS)}")
        threshold_metrics = {
            threshold: [evaluate_retained(item, threshold=threshold, iou_threshold=args.iou_threshold) for item in variant_sets]
            for threshold in thresholds
        }
        tau_star = find_zero_fp_tau(thresholds, threshold_metrics)
        epsilon_rows = {
            f"{epsilon:.2f}": summarize_threshold(select_epsilon_tau(thresholds, threshold_metrics, epsilon), threshold_metrics)
            for epsilon in EPSILONS
        }
        tau_summary = summarize_threshold(tau_star, threshold_metrics)
        per_variant[variant] = {
            "variant": variant,
            "variant_label": VARIANT_LABELS[variant],
            "tau_star": tau_star,
            "per_seed_at_tau_star": threshold_metrics[tau_star],
            "summary_at_tau_star": tau_summary,
            "epsilon_operating_points": epsilon_rows,
        }

    output = {
        "predictions_dir": str(predictions_dir),
        "threshold_grid": thresholds,
        "iou_threshold": args.iou_threshold,
        "variants": per_variant,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(f"\nJSON_DUMP: {args.output_json}")
    print_latex_opsel(per_variant)
    print_latex_sensitivity(per_variant)
    print_audit_block(per_variant)
    print_acceptance_checks(per_variant, tolerance=args.rounding_tolerance)
    print_existing_table_definition()


def resolve_predictions_dir(cli_path: Path | None) -> Path:
    if cli_path is not None:
        path = cli_path.expanduser().resolve()
        if not path.exists():
            raise SystemExit(f"Predictions directory does not exist: {path}")
        return path
    for path in DEFAULT_PREDICTION_DIRS:
        if path.exists() and len(list(path.glob("*_predictions.json"))) >= len(VARIANTS) * len(SEEDS):
            return path.resolve()
    searched = "\n".join(f"- {path}" for path in DEFAULT_PREDICTION_DIRS)
    raise SystemExit(f"No complete prediction directory found. Searched:\n{searched}")


def load_prediction_sets(predictions_dir: Path) -> list[dict[str, Any]]:
    prediction_sets = ta.load_prediction_sets(predictions_dir)
    prediction_sets = [
        item for item in prediction_sets if item["variant"] in set(VARIANTS) and int(item["seed"]) in set(SEEDS)
    ]
    found = {(item["variant"], int(item["seed"])) for item in prediction_sets}
    missing = [(variant, seed) for variant in VARIANTS for seed in SEEDS if (variant, seed) not in found]
    if missing:
        raise SystemExit(f"Missing prediction JSONs: {missing}")
    return sorted(prediction_sets, key=lambda item: (VARIANTS.index(item["variant"]), int(item["seed"])))


def evaluate_retained(prediction_set: dict[str, Any], *, threshold: float, iou_threshold: float) -> dict[str, Any]:
    row, negative_vector = ta.evaluate_prediction_set(prediction_set, threshold=threshold, iou_threshold=iou_threshold)
    positive_images = [image for image in prediction_set["images"] if not bool(image.get("is_knot_free", False))]
    negative_images = [image for image in prediction_set["images"] if bool(image.get("is_knot_free", False))]
    retained_positive = count_retained(positive_images, threshold)
    retained_negative = count_retained(negative_images, threshold)
    tp = int(row["tp50"])
    fp_positive = int(row["fp50_positive"])
    fn = int(row["fn50"])
    precision = float(ta.parse_float(row["precision"]))
    recall = float(ta.parse_float(row["recall"]))
    ap50 = float(ta.parse_float(row["ap50"]))
    return {
        "variant": prediction_set["variant"],
        "seed": int(prediction_set["seed"]),
        "threshold": float(threshold),
        "n_retained": int(retained_positive + retained_negative),
        "n_retained_positive": int(retained_positive),
        "n_retained_knotfree": int(retained_negative),
        "n_TP": tp,
        "n_FP": fp_positive + int(retained_negative),
        "n_FP_positive": fp_positive,
        "n_FP_knotfree": int(retained_negative),
        "n_FN": fn,
        "precision_at_tau": precision,
        "retained_recall": recall,
        "retained_AP50": ap50,
        "fp_image_rate_knotfree": float(ta.parse_float(row["fp_image_rate"])),
        "knotfree_fp_images": int(row["knotfree_fp_images"]),
        "num_positive_images": int(row["num_positive_images"]),
        "num_knotfree_images": int(row["num_knotfree_images"]),
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
        rows = threshold_metrics[threshold]
        if all(row["fp_image_rate_knotfree"] == 0.0 for row in rows):
            return float(threshold)
    raise ValueError("No zero-FP threshold found on the configured grid.")


def select_epsilon_tau(
    thresholds: list[float],
    threshold_metrics: dict[float, list[dict[str, Any]]],
    epsilon: float,
) -> float:
    for threshold in thresholds:
        rows = threshold_metrics[threshold]
        fp_mean = mean(row["fp_image_rate_knotfree"] for row in rows)
        if fp_mean <= epsilon + 1e-12:
            return float(threshold)
    raise ValueError(f"No threshold satisfies epsilon={epsilon}")


def summarize_threshold(threshold: float, threshold_metrics: dict[float, list[dict[str, Any]]]) -> dict[str, Any]:
    rows = threshold_metrics[threshold]
    return {
        "threshold": float(threshold),
        "n_seeds": len(rows),
        "seeds": [int(row["seed"]) for row in rows],
        "n_retained_mean": mean(row["n_retained"] for row in rows),
        "n_retained_std": std(row["n_retained"] for row in rows),
        "n_TP_mean": mean(row["n_TP"] for row in rows),
        "n_TP_std": std(row["n_TP"] for row in rows),
        "n_FP_mean": mean(row["n_FP"] for row in rows),
        "n_FP_std": std(row["n_FP"] for row in rows),
        "precision_at_tau_mean": mean(row["precision_at_tau"] for row in rows),
        "precision_at_tau_std": std(row["precision_at_tau"] for row in rows),
        "retained_recall_mean": mean(row["retained_recall"] for row in rows),
        "retained_recall_std": std(row["retained_recall"] for row in rows),
        "retained_AP50_mean": mean(row["retained_AP50"] for row in rows),
        "retained_AP50_std": std(row["retained_AP50"] for row in rows),
        "fp_image_rate_knotfree_mean": mean(row["fp_image_rate_knotfree"] for row in rows),
        "fp_image_rate_knotfree_std": std(row["fp_image_rate_knotfree"] for row in rows),
        "per_seed": rows,
    }


def print_reuse_report(predictions_dir: Path) -> None:
    print("REUSED FUNCTIONS/PATHS")
    print(f"- predictions_dir: {predictions_dir}")
    print(f"- prediction loader: scripts.threshold_analysis.load_prediction_sets ({source_location(ta.load_prediction_sets)})")
    print(f"- threshold evaluator/AP wrapper: scripts.threshold_analysis.evaluate_prediction_set ({source_location(ta.evaluate_prediction_set)})")
    print(f"- class-level AP estimator: scripts.threshold_analysis.evaluate_class ({source_location(ta.evaluate_class)})")
    print(f"- AP interpolation: scripts.threshold_analysis.average_precision ({source_location(ta.average_precision)})")
    print(f"- IoU helper: scripts.threshold_analysis.box_iou ({source_location(ta.box_iou)})")
    print("- prediction JSON schema: exported by scripts.threshold_sweep_inference.py; uses image['is_knot_free'], image['gt_boxes'], and image['predictions']")


def print_dataset_counts(prediction_sets: list[dict[str, Any]]) -> None:
    print("\nDATASET COUNTS FROM PREDICTION JSONS")
    for item in prediction_sets:
        positive = [image for image in item["images"] if not bool(image.get("is_knot_free", False))]
        negative = [image for image in item["images"] if bool(image.get("is_knot_free", False))]
        gt = sum(len(image.get("gt_boxes", [])) for image in positive)
        base_conf = read_base_conf(item.get("path"))
        print(
            f"- {item['variant']}_seed{item['seed']}: images={len(item['images'])}, "
            f"defective={len(positive)}, knot_free={len(negative)}, gt_boxes={gt}, "
            f"base_conf={base_conf}"
        )
    unique_counts = {
        (
            len([image for image in item["images"] if not bool(image.get("is_knot_free", False))]),
            len([image for image in item["images"] if bool(image.get("is_knot_free", False))]),
            sum(len(image.get("gt_boxes", [])) for image in item["images"] if not bool(image.get("is_knot_free", False))),
        )
        for item in prediction_sets
    }
    print(f"- unique (defective_images, knot_free_images, gt_boxes): {sorted(unique_counts)}")


def read_base_conf(path: Any) -> str:
    if path is None:
        return "n/a"
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return str(payload.get("base_confidence_threshold", "n/a"))


def print_latex_opsel(per_variant: dict[str, Any]) -> None:
    print("\nLATEX tab:opsel rows")
    for variant in VARIANTS:
        summary = per_variant[variant]["summary_at_tau_star"]
        print(
            f"{VARIANT_LABELS[variant]:<16} & {summary['threshold']:.2f} & "
            f"${summary['retained_recall_mean']:.3f}\\pm{summary['retained_recall_std']:.3f}$ & "
            f"${summary['retained_AP50_mean']:.3f}\\pm{summary['retained_AP50_std']:.3f}$ \\\\"
        )


def print_latex_sensitivity(per_variant: dict[str, Any]) -> None:
    print("\nLATEX tab:sensitivity rows")
    for variant in VARIANTS:
        parts = [VARIANT_LABELS[variant]]
        for epsilon in EPSILONS:
            row = per_variant[variant]["epsilon_operating_points"][f"{epsilon:.2f}"]
            parts.extend([f"{row['threshold']:.2f}", f"{row['retained_AP50_mean']:.3f}"])
        print(" & ".join(parts) + r" \\")


def print_audit_block(per_variant: dict[str, Any]) -> None:
    print("\nAUDIT")
    for variant in VARIANTS:
        summary = per_variant[variant]["summary_at_tau_star"]
        identity = abs(summary["retained_recall_mean"] - summary["retained_AP50_mean"]) <= 1e-3
        print(
            f"- {VARIANT_LABELS[variant]} tau={summary['threshold']:.2f}: "
            f"precision_at_tau={summary['precision_at_tau_mean']:.6f}, "
            f"retained_recall={summary['retained_recall_mean']:.6f}, "
            f"retained_AP50={summary['retained_AP50_mean']:.6f}, "
            f"recall == AP50 within 1e-3: {identity}"
        )
        for row in summary["per_seed"]:
            print(
                f"    seed={row['seed']}: precision={row['precision_at_tau']:.6f}, "
                f"recall={row['retained_recall']:.6f}, AP50={row['retained_AP50']:.6f}, "
                f"TP={row['n_TP']}, FP_pos={row['n_FP_positive']}, FP_clean={row['n_FP_knotfree']}"
            )


def print_acceptance_checks(per_variant: dict[str, Any], *, tolerance: float) -> None:
    print("\nACCEPTANCE CHECKS")
    all_ok = True
    for variant in VARIANTS:
        summary = per_variant[variant]["summary_at_tau_star"]
        expected = EXPECTED_OPSEL[variant]
        checks = {
            "tau": summary["threshold"] - expected["tau"],
            "recall": round(summary["retained_recall_mean"], 3) - expected["recall"],
            "AP50": round(summary["retained_AP50_mean"], 3) - expected["ap50"],
        }
        ok = all(abs(delta) <= tolerance for delta in checks.values())
        all_ok = all_ok and ok
        status = "PASS" if ok else "FAIL"
        deltas = ", ".join(f"{name}_delta={delta:+.6f}" for name, delta in checks.items())
        print(f"- tab:opsel {VARIANT_LABELS[variant]}: {status} ({deltas})")

    baseline = per_variant["baseline"]["summary_at_tau_star"]
    baseline_precision_all_one = all(abs(row["precision_at_tau"] - 1.0) <= 1e-9 for row in baseline["per_seed"])
    baseline_identity = abs(baseline["retained_recall_mean"] - baseline["retained_AP50_mean"]) <= 1e-3
    if baseline_precision_all_one and baseline_identity:
        print("- BASELINE IDENTITY: expected (precision=1)")
    elif not baseline_precision_all_one and baseline_identity:
        all_ok = False
        print("- BASELINE IDENTITY: SUSPECT - investigate")
    else:
        print("- BASELINE IDENTITY: not an identity; AP50 and recall differ as expected")
    print(f"- OVERALL tab:opsel acceptance: {'PASS' if all_ok else 'FAIL'}")


def print_existing_table_definition() -> None:
    score_filter_lines = quote_source_lines(Path(ta.__file__), 195, 201)
    ap_lines = quote_source_lines(Path(ta.__file__), 223, 228)
    interp_lines = quote_source_lines(Path(ta.__file__), 380, 386)
    sweet_lines = quote_source_lines(Path(ta.__file__), 329, 331)
    print("\nEXISTING TABLE CODE DEFINITION")
    print(
        "EXISTING retained AP50 definition: sub-curve down to tau_star. "
        f"Evidence: threshold_analysis.py:195-201 filters out score < threshold: {compact(score_filter_lines)}; "
        f"threshold_analysis.py:223-228 computes AP from the retained ranked detections: {compact(ap_lines)}; "
        f"threshold_analysis.py:380-386 applies the precision-envelope AP interpolation: {compact(interp_lines)}; "
        f"threshold_analysis.py:329-331 writes mean(row['ap50']) at selected_threshold into operational_sweet_spots: {compact(sweet_lines)}"
    )


def source_location(func: Any) -> str:
    path = Path(inspect.getsourcefile(func) or "")
    try:
        rel = path.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = path
    _, start_line = inspect.getsourcelines(func)
    return f"{rel}:{start_line}"


def quote_source_lines(path: Path, start: int, end: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines()
    selected = []
    for line_no in range(start, end + 1):
        if 1 <= line_no <= len(lines):
            selected.append(f"L{line_no}: {lines[line_no - 1].strip()}")
    return " | ".join(selected)


def compact(text: str) -> str:
    return " ".join(text.split())


def mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else float("nan")


def std(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return statistics.stdev(values) if len(values) >= 2 else 0.0


if __name__ == "__main__":
    main()
