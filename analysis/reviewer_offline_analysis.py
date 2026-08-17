#!/usr/bin/env python3
"""Run reviewer-requested offline analyses on a frozen experiment generation."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import platform
import shutil
import statistics
import sys
from typing import Any, Iterable

import numpy as np
import scipy
from scipy.stats import beta, norm, spearmanr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from analysis import analyze_generation as core  # noqa: E402


THRESHOLDS = core.THRESHOLDS
EPSILONS = core.EPSILONS
VARIANTS = core.VARIANTS
VARIANT_LABELS = core.VARIANT_LABELS
SEEDS = core.SEEDS
DATASETS = core.DATASETS
CONFIDENCE_LEVEL = 0.95
SUBSAMPLE_SEED = 20260817
SUBSAMPLE_DRAWS = 1000
SUBSAMPLE_SIZE = 75


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
        default=PROJECT_ROOT / "revised" / "analysis" / "access_r1_g2" / "reviewer_offline",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--subsample-draws", type=int, default=SUBSAMPLE_DRAWS)
    parser.add_argument("--subsample-seed", type=int, default=SUBSAMPLE_SEED)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generation_root = args.generation_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not generation_root.is_dir():
        raise SystemExit(f"Frozen generation not found: {generation_root}")
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists: {output_dir}. Use --overwrite intentionally.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    print("REVIEWER OFFLINE ANALYSIS")
    print(f"- frozen read-only input: {generation_root}")
    print(f"- new output directory: {output_dir}")
    print("- no training or inference")
    print("- AP source: saved DetectionValidator TP masks and Ultralytics 8.4.60 101-point AP")

    payloads = core.load_primary_payloads(generation_root)
    core.verify_payload_inventory(payloads)
    reproduction = core.verify_base_ap_reproduction(payloads, tolerance=2e-6)
    fair_rows = core.read_csv(generation_root / "fair_eval" / "fair_metrics.csv")

    selections, locked_rows = compute_tile_level_locked_results(payloads)
    binding_rows = analyze_binding_constraints(payloads, selections)
    task0 = task0_report(generation_root, payloads, selections, locked_rows, binding_rows)
    write_task(output_dir, "T0_provenance_and_anomalies", task0)
    write_csv(output_dir / "T0_binding_constraints.csv", binding_rows)
    write_csv(output_dir / "T0_locked_operating_points_per_seed.csv", locked_rows)

    interval_rows = exact_interval_sweep(payloads)
    crosschecks = interval_crosschecks()
    task1 = task1_report(interval_rows, selections, crosschecks)
    write_task(output_dir, "T1_exact_binomial_intervals", task1)
    write_csv(output_dir / "T1_exact_binomial_intervals.csv", interval_rows)
    write_csv(output_dir / "T1_interval_crosschecks.csv", crosschecks)
    write_t1_latex(output_dir, interval_rows, selections)

    fppi_rows = fppi_sweep(payloads)
    task2 = task2_report(fppi_rows, selections)
    write_task(output_dir, "T2_fppi", task2)
    write_csv(output_dir / "T2_fp_rate_and_fppi_per_seed.csv", fppi_rows)
    write_t2_latex(output_dir, fppi_rows, selections)

    sensitivity_summary, ranking_rows = summarize_sensitivity(locked_rows)
    task3 = task3_report(sensitivity_summary, ranking_rows)
    write_task(output_dir, "T3_tolerance_sweep", task3)
    write_csv(output_dir / "T3_tolerance_per_seed.csv", locked_rows)
    write_csv(output_dir / "T3_tolerance_summary.csv", sensitivity_summary)
    write_csv(output_dir / "T3_ranking_stability.csv", ranking_rows)
    write_t3_latex(output_dir, sensitivity_summary)

    convergence_per_seed, convergence_summary, curves = analyze_convergence(generation_root)
    task4 = task4_report(convergence_per_seed, convergence_summary)
    write_task(output_dir, "T4_convergence", task4)
    write_csv(output_dir / "T4_convergence_per_seed.csv", convergence_per_seed)
    write_csv(output_dir / "T4_convergence_summary.csv", convergence_summary)
    plot_convergence(output_dir, curves)

    source_rows, source_locked, subsample_rows, subsample_summary = analyze_vsb_sources(
        payloads,
        draws=int(args.subsample_draws),
        rng_seed=int(args.subsample_seed),
    )
    p2_source_overlap = analyze_vsb_p2_source_overlap(payloads, source_locked)
    task5 = task5_report(
        source_rows,
        source_locked,
        subsample_summary,
        locked_rows,
        args,
        p2_source_overlap,
    )
    write_task(output_dir, "T5_vsb_source_level", task5)
    write_csv(output_dir / "T5_source_fp_sweep_per_seed.csv", source_rows)
    write_csv(output_dir / "T5_source_locked_results_per_seed.csv", source_locked)
    write_csv(output_dir / "T5_source_subsample_draws.csv", subsample_rows)
    write_csv(output_dir / "T5_source_subsample_summary.csv", subsample_summary)
    write_csv(output_dir / "T5_p2_repeated_fp_sources.csv", p2_source_overlap)
    write_t5_latex(output_dir, source_locked, locked_rows)

    calibration_per_seed, _ = core.analyze_calibration(payloads, bins=10, iou=0.50)
    calibration_summary = summarize_calibration_complete(calibration_per_seed)
    task6 = task6_report(calibration_per_seed, calibration_summary)
    write_task(output_dir, "T6_calibration_all_seeds", task6)
    write_csv(output_dir / "T6_calibration_per_seed.csv", calibration_per_seed)
    write_csv(output_dir / "T6_calibration_summary.csv", calibration_summary)
    write_t6_latex(output_dir, calibration_per_seed, calibration_summary)

    pr_rows, pr_summary, relation_summary = characterize_retained_pr(payloads, selections)
    task7 = task7_report(pr_summary, relation_summary)
    write_task(output_dir, "T7_retained_ap_characterization", task7)
    write_csv(output_dir / "T7_retained_pr_curve_points.csv", pr_rows)
    write_csv(output_dir / "T7_retained_pr_summary_per_seed.csv", pr_summary)
    write_csv(output_dir / "T7_relation_summary.csv", relation_summary)
    plot_retained_relation(output_dir, pr_summary)

    deprecated = load_deprecated_payloads(generation_root)
    dep_rows, dep_summary, validation_bias = analyze_deprecated_impact(
        generation_root, payloads, deprecated
    )
    task8 = task8_report(dep_rows, dep_summary, validation_bias)
    write_task(output_dir, "T8_deprecated_checkpoint_impact", task8)
    write_csv(output_dir / "T8_deprecated_clean_impact_per_seed.csv", dep_rows)
    write_csv(output_dir / "T8_deprecated_clean_impact_summary.csv", dep_summary)
    write_csv(output_dir / "T8_validation_selection_bias.csv", validation_bias)
    write_t8_latex(output_dir, dep_summary)

    master = build_master_report(output_dir, reproduction)
    (output_dir / "MASTER_REPORT.md").write_text(master, encoding="utf-8")
    metadata = {
        "frozen_generation": str(generation_root),
        "output_dir": str(output_dir),
        "base_ap_reproduction_rows": len(reproduction),
        "base_ap_max_abs_delta": max(abs(float(row["delta_mAP50"])) for row in reproduction),
        "confidence_level": CONFIDENCE_LEVEL,
        "threshold_grid": THRESHOLDS,
        "epsilons": EPSILONS,
        "subsample_seed": int(args.subsample_seed),
        "subsample_draws": int(args.subsample_draws),
        "subsample_size": SUBSAMPLE_SIZE,
        "fair_metric_rows": len(fair_rows),
        "analysis_environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
    }
    (output_dir / "analysis_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"BASE AP: PASS ({len(reproduction)}/{len(reproduction)})")
    print(f"Wrote: {output_dir / 'MASTER_REPORT.md'}")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    core.write_csv(path, rows)


def write_task(output_dir: Path, stem: str, text: str) -> None:
    (output_dir / f"{stem}.md").write_text(text.rstrip() + "\n", encoding="utf-8")


def avg(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return statistics.mean(values) if values else float("nan")


def sd(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return statistics.stdev(values) if len(values) > 1 else 0.0


def med(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return statistics.median(values) if values else float("nan")


def pm(value: float, deviation: float, digits: int = 3) -> str:
    return f"{value:.{digits}f} +/- {deviation:.{digits}f}"


def image_id(image: dict[str, Any]) -> str:
    return str(image.get("canonical_id", image.get("image", "")))


def max_confidence(image: dict[str, Any]) -> float:
    return max((float(pred["conf"]) for pred in image.get("predictions", [])), default=0.0)


def clean_units(payload: dict[str, Any], level: str = "tile") -> dict[str, list[dict[str, Any]]]:
    images = core.clean_images(payload)
    if level == "tile":
        return {image_id(image): [image] for image in images}
    if level != "source":
        raise ValueError(f"Unknown clean unit level: {level}")
    output: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for image in images:
        stem = Path(image_id(image)).stem
        source_id = stem.split("__", 1)[0]
        output[source_id].append(image)
    return dict(output)


def unit_maxima(payload: dict[str, Any], level: str = "tile") -> dict[str, float]:
    return {
        unit_id: max((max_confidence(image) for image in images), default=0.0)
        for unit_id, images in clean_units(payload, level).items()
    }


def unit_counts(payload: dict[str, Any], threshold: float, level: str = "tile") -> dict[str, int]:
    return {
        unit_id: sum(
            float(pred["conf"]) >= threshold
            for image in images
            for pred in image.get("predictions", [])
        )
        for unit_id, images in clean_units(payload, level).items()
    }


def clean_payload_key(dataset: str, variant: str, seed: int, split: str) -> tuple[str, str, int, str]:
    return core.clean_payload_key(dataset, variant, seed, split)


def select_threshold(
    payloads: dict[tuple[str, str, int, str], dict[str, Any]],
    dataset: str,
    variant: str,
    epsilon: float,
    *,
    level: str = "tile",
    subset_ids: set[str] | None = None,
) -> tuple[float, list[float]]:
    for threshold in THRESHOLDS:
        rates = []
        for seed in SEEDS:
            payload = payloads[clean_payload_key(dataset, variant, seed, "val")]
            counts = unit_counts(payload, threshold, level)
            if subset_ids is not None:
                counts = {key: value for key, value in counts.items() if key in subset_ids}
            if not counts:
                raise RuntimeError(f"No validation clean units for {dataset}/{variant}/seed{seed}")
            rates.append(sum(value > 0 for value in counts.values()) / len(counts))
        qualifies = all(rate == 0.0 for rate in rates) if epsilon == 0.0 else avg(rates) <= epsilon
        if qualifies:
            return threshold, rates
    raise RuntimeError(f"No qualifying threshold for {dataset}/{variant}/epsilon={epsilon}")


def compute_tile_level_locked_results(
    payloads: dict[tuple[str, str, int, str], dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selections = []
    locked = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            for epsilon in EPSILONS:
                threshold, validation_rates = select_threshold(payloads, dataset, variant, epsilon)
                selections.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "epsilon": epsilon,
                        "selected_threshold": threshold,
                        "validation_fp_rate_mean": avg(validation_rates),
                        "validation_fp_rate_std": sd(validation_rates),
                    }
                )
                for seed in SEEDS:
                    detection = payloads[(dataset, variant, seed, "test")]
                    clean = payloads[clean_payload_key(dataset, variant, seed, "test")]
                    metrics = core.evaluate_detection_payload(detection, threshold, 0.50)
                    counts = unit_counts(clean, threshold)
                    fp_images = sum(value > 0 for value in counts.values())
                    locked.append(
                        {
                            "dataset": dataset,
                            "variant": variant,
                            "seed": seed,
                            "epsilon": epsilon,
                            "selected_threshold": threshold,
                            "validation_fp_rate": validation_rates[SEEDS.index(seed)],
                            "test_n_clean": len(counts),
                            "test_fp_images": fp_images,
                            "test_fp_rate": fp_images / len(counts),
                            "test_false_detections": sum(counts.values()),
                            "test_fppi": avg(counts.values()),
                            **metrics,
                        }
                    )
    return selections, locked


def analyze_binding_constraints(
    payloads: dict[tuple[str, str, int, str], dict[str, Any]],
    selections: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    selected = {
        (row["dataset"], row["variant"]): float(row["selected_threshold"])
        for row in selections
        if float(row["epsilon"]) == 0.0
    }
    rows = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            threshold = selected[(dataset, variant)]
            previous = round(threshold - 0.05, 2)
            has_lower_grid_point = previous >= THRESHOLDS[0]
            all_entries = []
            for seed in SEEDS:
                payload = payloads[clean_payload_key(dataset, variant, seed, "val")]
                maxima = unit_maxima(payload)
                top = max(maxima.values())
                top_ids = sorted(key for key, value in maxima.items() if math.isclose(value, top, abs_tol=1e-12))
                all_entries.extend((seed, key, value) for key, value in maxima.items())
                rows.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "scope": "per_seed_max",
                        "seed": seed,
                        "selected_threshold": threshold,
                        "previous_grid_threshold": previous,
                        "image_or_tile_id": ";".join(top_ids),
                        "max_confidence": top,
                        "margin_to_selected_threshold": threshold - top,
                        "blocks_previous_grid_threshold": top >= previous if has_lower_grid_point else "not_applicable_grid_floor",
                        "binding_pairs_across_seeds": "",
                        "single_binding_pair": "",
                    }
                )
            global_max = max(entry[2] for entry in all_entries)
            global_top = [(seed, key) for seed, key, value in all_entries if math.isclose(value, global_max, abs_tol=1e-12)]
            blockers = (
                [(seed, key, value) for seed, key, value in all_entries if value >= previous]
                if has_lower_grid_point else []
            )
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "scope": "cross_seed_binding",
                    "seed": "all",
                    "selected_threshold": threshold,
                    "previous_grid_threshold": previous,
                    "image_or_tile_id": ";".join(f"seed{seed}:{key}" for seed, key in global_top),
                    "max_confidence": global_max,
                    "margin_to_selected_threshold": threshold - global_max,
                    "blocks_previous_grid_threshold": global_max >= previous if has_lower_grid_point else "not_applicable_grid_floor",
                    "binding_pairs_across_seeds": len(blockers),
                    "single_binding_pair": len(blockers) == 1,
                }
            )
    return rows


def task0_report(
    generation_root: Path,
    payloads: dict[tuple[str, str, int, str], dict[str, Any]],
    selections: list[dict[str, Any]],
    locked: list[dict[str, Any]],
    binding: list[dict[str, Any]],
) -> str:
    registry = core.read_csv(generation_root / "provenance" / "checkpoint_registry.csv")
    source_counts = Counter(row["source_type"] for row in registry)
    deprecated_registry = core.read_csv(
        generation_root / "deprecated_checkpoints" / "deprecated_checkpoint_registry.csv"
    )
    old_roots = sorted({row["source_snapshot_root"] for row in deprecated_registry})
    runtime = json.loads((generation_root / "provenance" / "runtime_preflight.json").read_text())
    lines = [
        "# T0 - Provenance and three preliminary questions",
        "",
        "## 0.1 Why this is `access_r1_g2`",
        "",
        "Yes. An earlier `access_r1_g1` server attempt existed and was superseded. The frozen registry preserves "
        f"the old source root `{old_roots[0] if old_roots else 'not recorded'}` for the 18 explicitly deprecated "
        "augmentation checkpoints. The operator log recorded that the first attempt used OpenCV 5.0.0 while the "
        "pinned environment required 4.10.0; the runtime checker was also corrected to the CUDA-12.4-compatible "
        "minimum NVIDIA driver. The replacement `g2` preflight passed and records "
        f"OpenCV {runtime['packages']['opencv']}, Ultralytics {runtime['packages']['ultralytics']}, "
        f"Torch {runtime['packages']['torch']}, driver {runtime['nvidia_driver']}, and commit `{runtime['git_commit']}`.",
        "",
        f"The final g2 registry contains {source_counts['archived_survivor']} archived survivor runs and "
        f"{source_counts['corrected_retrain']} corrected retrains, all marked PASS. The g1 runtime preflight JSON "
        "is not present in the downloaded frozen generation, so the exact g1 runtime is operator-log evidence, "
        "not independently hash-verifiable from g2. No g1 result is used as a primary corrected retrain.",
        "",
        "## 0.2 VNWoodKnot P4+A4 collapse",
        "",
        "| Seed | tau selected on validation | Test precision | Test recall | Retained AP50 | Base-floor detections (val/test) | Clean max confidence (val) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    selection = next(
        row for row in selections
        if row["dataset"] == "vnwoodknot" and row["variant"] == "p4_a4_combined" and row["epsilon"] == 0.0
    )
    for seed in SEEDS:
        row = next(
            item for item in locked
            if item["dataset"] == "vnwoodknot" and item["variant"] == "p4_a4_combined"
            and item["seed"] == seed and item["epsilon"] == 0.0
        )
        val_payload = payloads[("vnwoodknot", "p4_a4_combined", seed, "val")]
        test_payload = payloads[("vnwoodknot", "p4_a4_combined", seed, "test")]
        val_n = sum(len(image["predictions"]) for image in val_payload["images"])
        test_n = sum(len(image["predictions"]) for image in test_payload["images"])
        clean_max = max(unit_maxima(val_payload).values())
        lines.append(
            f"| {seed} | {selection['selected_threshold']:.2f} | {row['precision']:.6f} | "
            f"{row['recall']:.6f} | {row['mAP50']:.6f} | {val_n}/{test_n} | {clean_max:.6f} |"
        )
    counts = []
    for seed in SEEDS:
        for split in ("val", "test"):
            payload = payloads[("vnwoodknot", "p4_a4_combined", seed, split)]
            counts.append(sum(len(image["predictions"]) for image in payload["images"]))
    lines.extend(
        [
            "",
            "The collapse is caused by seed 42's validation clean prediction at confidence 0.669661, which forces "
            "the shared grid threshold to 0.70 for all seeds. Seeds 43 and 44 have validation clean maxima below "
            "0.051 and would not independently require this threshold. Thus the low retained recall is not a "
            "three-seed model failure; it is a cross-seed threshold-selection consequence driven by one seed/image.",
            "",
            f"No anomalous prediction flood recurs: total base-floor predictions across P4+A4 val/test payloads "
            f"range from {min(counts)} to {max(counts)}. These are comparable in order of magnitude, rather than "
            "the prior anomalous multi-thousand spread.",
            "",
            "## 0.3 Binding validation clean samples",
            "",
            "A pair is counted as grid-binding when it has max confidence at or above the immediately lower 0.05 "
            "grid point and therefore prevents selecting that lower threshold.",
            "",
            "| Dataset | Variant | tau | Highest seed:image/tile | max conf | Grid-binding pairs | Single pair? |",
            "|---|---|---:|---|---:|---:|---|",
        ]
    )
    for row in binding:
        if row["scope"] != "cross_seed_binding":
            continue
        lines.append(
            f"| {row['dataset']} | {VARIANT_LABELS[row['variant']]} | {row['selected_threshold']:.2f} | "
            f"`{row['image_or_tile_id']}` | {row['max_confidence']:.6f} | "
                f"{row['binding_pairs_across_seeds'] if row['selected_threshold'] > THRESHOLDS[0] else 'n/a'} | "
                f"{'yes' if row['single_binding_pair'] else ('n/a' if row['selected_threshold'] == THRESHOLDS[0] else 'no')} |"
        )
    lines.extend(
        [
            "",
            "**Conclusion:** exact-zero selection is brittle by construction. A single prediction can move the "
            "locked threshold by one or more grid steps; VN P4+A4 is the clearest observed example.",
        ]
    )
    return "\n".join(lines)


def binomial_intervals(k: int, n: int, confidence: float = CONFIDENCE_LEVEL) -> tuple[float, float, float, float]:
    if n <= 0 or not 0 <= k <= n:
        raise ValueError((k, n))
    alpha = 1.0 - confidence
    cp_low = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    cp_high = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    z = float(norm.ppf(1 - alpha / 2))
    p = k / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return cp_low, cp_high, max(0.0, center - half), min(1.0, center + half)


def exact_interval_sweep(
    payloads: dict[tuple[str, str, int, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            for seed in SEEDS:
                for split in ("val", "test"):
                    payload = payloads[clean_payload_key(dataset, variant, seed, split)]
                    for threshold in THRESHOLDS:
                        counts = unit_counts(payload, threshold)
                        n = len(counts)
                        k = sum(value > 0 for value in counts.values())
                        cp_low, cp_high, wilson_low, wilson_high = binomial_intervals(k, n)
                        rows.append(
                            {
                                "dataset": dataset,
                                "variant": variant,
                                "seed": seed,
                                "split": split,
                                "threshold": threshold,
                                "fp_images": k,
                                "n_clean_images_or_tiles": n,
                                "fp_rate": k / n,
                                "cp95_lower": cp_low,
                                "cp95_upper": cp_high,
                                "wilson95_lower": wilson_low,
                                "wilson95_upper": wilson_high,
                                "false_detections": sum(counts.values()),
                                "fppi": avg(counts.values()),
                            }
                        )
    return rows


def interval_crosschecks() -> list[dict[str, Any]]:
    expected = {
        (0, 75): (0.0, 0.0480),
        (1, 75): (0.0003, 0.0721),
        (5, 75): (0.0220, 0.1488),
        (0, 1992): (0.0, 0.00185),
        (0, 5976): (0.0, 0.00062),
    }
    rows = []
    for (k, n), (expected_low, expected_high) in expected.items():
        cp_low, cp_high, wilson_low, wilson_high = binomial_intervals(k, n)
        tolerance = 5e-5 if n <= 75 else 5e-6
        rows.append(
            {
                "k": k,
                "n": n,
                "cp95_lower": cp_low,
                "cp95_upper": cp_high,
                "wilson95_lower": wilson_low,
                "wilson95_upper": wilson_high,
                "expected_cp_lower": expected_low,
                "expected_cp_upper": expected_high,
                "status": "PASS"
                if abs(cp_low - expected_low) <= tolerance and abs(cp_high - expected_high) <= tolerance
                else "FAIL",
            }
        )
    if any(row["status"] != "PASS" for row in rows):
        raise RuntimeError(f"Binomial interval cross-check failed: {rows}")
    return rows


def task1_report(
    rows: list[dict[str, Any]],
    selections: list[dict[str, Any]],
    crosschecks: list[dict[str, Any]],
) -> str:
    selected_lookup = {
        (row["dataset"], row["variant"]): float(row["selected_threshold"])
        for row in selections if row["epsilon"] == 0.0
    }
    lines = [
        "# T1 - Exact binomial intervals",
        "",
        "All confidence intervals are two-sided 95% intervals. Clopper-Pearson is the primary interval; Wilson is "
        "reported for comparison. Each seed is kept separate because the three models are evaluated on the same "
        "clean samples. Pooling seeds would incorrectly treat repeated observations as independent and would be "
        "anti-conservative.",
        "",
        "## Independent cross-checks",
        "",
        "| k/N | Clopper-Pearson | Wilson | Status |",
        "|---|---:|---:|---|",
    ]
    for row in crosschecks:
        lines.append(
            f"| {row['k']}/{row['n']} | [{row['cp95_lower']:.6f}, {row['cp95_upper']:.6f}] | "
            f"[{row['wilson95_lower']:.6f}, {row['wilson95_upper']:.6f}] | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Replacement operating-point FP table (test split, per seed)",
            "",
            "| Dataset | Variant | tau from val | Seed | FP/N | FP rate | Exact CP 95% CI | FPPI |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        for variant in VARIANTS:
            threshold = selected_lookup[(dataset, variant)]
            for seed in SEEDS:
                row = next(
                    item for item in rows
                    if item["dataset"] == dataset and item["variant"] == variant
                    and item["seed"] == seed and item["split"] == "test"
                    and math.isclose(item["threshold"], threshold)
                )
                lines.append(
                    f"| {dataset} | {VARIANT_LABELS[variant]} | {threshold:.2f} | {seed} | "
                    f"{row['fp_images']}/{row['n_clean_images_or_tiles']} | {row['fp_rate']:.6f} | "
                    f"[{row['cp95_lower']:.6f}, {row['cp95_upper']:.6f}] | {row['fppi']:.6f} |"
                )
    lines.extend(
        [
            "",
            "The former bootstrap interval must be removed from the manuscript. In particular, observing zero false "
            "positive images does not imply a [0,0] population interval: for 0/75 the exact upper bound is 0.047995.",
        ]
    )
    return "\n".join(lines)


def write_t1_latex(
    output_dir: Path,
    interval_rows: list[dict[str, Any]],
    selections: list[dict[str, Any]],
) -> None:
    latex = output_dir / "latex"
    latex.mkdir(exist_ok=True)
    selected = {
        (row["dataset"], row["variant"]): float(row["selected_threshold"])
        for row in selections if row["epsilon"] == 0.0
    }
    lines = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            threshold = selected[(dataset, variant)]
            cells = []
            for seed in SEEDS:
                row = next(
                    item for item in interval_rows
                    if item["dataset"] == dataset and item["variant"] == variant and item["seed"] == seed
                    and item["split"] == "test" and math.isclose(item["threshold"], threshold)
                )
                cells.append(
                    f"{int(row['fp_images'])}/{int(row['n_clean_images_or_tiles'])} "
                    f"[{row['cp95_lower']:.3f},{row['cp95_upper']:.3f}]"
                )
            lines.append(
                f"{dataset} & {VARIANT_LABELS[variant]} & {threshold:.2f} & " + " & ".join(cells) + r" \\"
            )
    (latex / "T1_exact_fp_rows.tex").write_text("\n".join(lines) + "\n")


def fppi_sweep(payloads: dict[tuple[str, str, int, str], dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            for seed in SEEDS:
                for split in ("val", "test"):
                    payload = payloads[clean_payload_key(dataset, variant, seed, split)]
                    for threshold in THRESHOLDS:
                        counts = unit_counts(payload, threshold)
                        n = len(counts)
                        rows.append(
                            {
                                "dataset": dataset,
                                "variant": variant,
                                "seed": seed,
                                "split": split,
                                "threshold": threshold,
                                "n_clean_images_or_tiles": n,
                                "fp_images": sum(value > 0 for value in counts.values()),
                                "fp_image_rate": sum(value > 0 for value in counts.values()) / n,
                                "false_detections": sum(counts.values()),
                                "fppi": avg(counts.values()),
                                "max_detections_on_one_clean_image": max(counts.values()),
                            }
                        )
    return rows


def task2_report(rows: list[dict[str, Any]], selections: list[dict[str, Any]]) -> str:
    selected = {
        (row["dataset"], row["variant"]): float(row["selected_threshold"])
        for row in selections if row["epsilon"] == 0.0
    }
    lines = [
        "# T2 - FPPI alongside image-level FP rate",
        "",
        "`FP image rate` is the fraction of clean images/tiles with at least one retained prediction. `FPPI` is "
        "the number of retained false detections divided by the number of clean images/tiles. The complete 0.05 "
        "threshold sweep is in `T2_fp_rate_and_fppi_per_seed.csv`.",
        "",
        "| Dataset | Variant | tau | Seed | FP rate | FPPI | Total false detections | Maximum on one sample |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        for variant in VARIANTS:
            threshold = selected[(dataset, variant)]
            for seed in SEEDS:
                row = next(
                    item for item in rows
                    if item["dataset"] == dataset and item["variant"] == variant and item["seed"] == seed
                    and item["split"] == "test" and math.isclose(item["threshold"], threshold)
                )
                lines.append(
                    f"| {dataset} | {VARIANT_LABELS[variant]} | {threshold:.2f} | {seed} | "
                    f"{row['fp_image_rate']:.6f} | {row['fppi']:.6f} | {row['false_detections']} | "
                    f"{row['max_detections_on_one_clean_image']} |"
                )
    return "\n".join(lines)


def write_t2_latex(output_dir: Path, rows: list[dict[str, Any]], selections: list[dict[str, Any]]) -> None:
    selected = {
        (row["dataset"], row["variant"]): float(row["selected_threshold"])
        for row in selections if row["epsilon"] == 0.0
    }
    lines = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            threshold = selected[(dataset, variant)]
            items = [
                row for row in rows
                if row["dataset"] == dataset and row["variant"] == variant and row["split"] == "test"
                and math.isclose(row["threshold"], threshold)
            ]
            lines.append(
                f"{dataset} & {VARIANT_LABELS[variant]} & {threshold:.2f} & "
                f"{avg(row['fp_image_rate'] for row in items):.4f}$\\pm${sd(row['fp_image_rate'] for row in items):.4f} & "
                f"{avg(row['fppi'] for row in items):.4f}$\\pm${sd(row['fppi'] for row in items):.4f} \\\\"
            )
    latex = output_dir / "latex"
    latex.mkdir(exist_ok=True)
    (latex / "T2_fppi_rows.tex").write_text("\n".join(lines) + "\n")


def summarize_sensitivity(
    rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["variant"], float(row["epsilon"]))].append(row)
    summary = []
    for (dataset, variant, epsilon), items in sorted(grouped.items()):
        out = {
            "dataset": dataset,
            "variant": variant,
            "epsilon": epsilon,
            "selected_threshold": float(items[0]["selected_threshold"]),
            "validation_fp_rate_mean": avg(row["validation_fp_rate"] for row in items),
        }
        for metric in ("test_fp_rate", "test_fppi", "precision", "recall", "mAP50"):
            out[f"{metric}_mean"] = avg(row[metric] for row in items)
            out[f"{metric}_std"] = sd(row[metric] for row in items)
        summary.append(out)
    ranking = []
    for dataset in DATASETS:
        baseline_order = None
        for epsilon in EPSILONS:
            items = [row for row in summary if row["dataset"] == dataset and row["epsilon"] == epsilon]
            ap_order = sorted(items, key=lambda row: (-row["mAP50_mean"], VARIANTS.index(row["variant"])))
            recall_order = sorted(items, key=lambda row: (-row["recall_mean"], VARIANTS.index(row["variant"])))
            if baseline_order is None:
                baseline_order = [row["variant"] for row in ap_order]
            positions = {variant: index for index, variant in enumerate(baseline_order)}
            current = [positions[row["variant"]] for row in ap_order]
            rho = float(spearmanr(range(len(current)), current).statistic)
            for rank, row in enumerate(ap_order, start=1):
                ranking.append(
                    {
                        "dataset": dataset,
                        "epsilon": epsilon,
                        "metric": "retained_mAP50",
                        "rank": rank,
                        "variant": row["variant"],
                        "value": row["mAP50_mean"],
                        "rank_change_from_epsilon0": positions[row["variant"]] + 1 - rank,
                        "spearman_vs_epsilon0": rho,
                    }
                )
            for rank, row in enumerate(recall_order, start=1):
                ranking.append(
                    {
                        "dataset": dataset,
                        "epsilon": epsilon,
                        "metric": "retained_recall",
                        "rank": rank,
                        "variant": row["variant"],
                        "value": row["recall_mean"],
                        "rank_change_from_epsilon0": "",
                        "spearman_vs_epsilon0": "",
                    }
                )
    return summary, ranking


def task3_report(summary: list[dict[str, Any]], ranking: list[dict[str, Any]]) -> str:
    lines = [
        "# T3 - Validation-selected tolerance sweep",
        "",
        "Thresholds are selected only on validation clean data. The selected threshold is then locked and applied "
        "unchanged to the defective/common test set and held-out clean test set.",
        "",
        "| Dataset | Variant | epsilon | tau | Test FP rate | Retained recall | Retained AP50 | AP rank |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    rank_lookup = {
        (row["dataset"], row["variant"], row["epsilon"]): row["rank"]
        for row in ranking if row["metric"] == "retained_mAP50"
    }
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {VARIANT_LABELS[row['variant']]} | {row['epsilon']:.2f} | "
            f"{row['selected_threshold']:.2f} | {pm(row['test_fp_rate_mean'], row['test_fp_rate_std'], 4)} | "
            f"{pm(row['recall_mean'], row['recall_std'])} | {pm(row['mAP50_mean'], row['mAP50_std'])} | "
            f"{rank_lookup[(row['dataset'], row['variant'], row['epsilon'])]} |"
        )
    lines.extend(["", "## Ranking stability", ""])
    for dataset in DATASETS:
        lines.append(f"### {dataset}")
        for epsilon in EPSILONS:
            ordered = sorted(
                [row for row in ranking if row["dataset"] == dataset and row["epsilon"] == epsilon and row["metric"] == "retained_mAP50"],
                key=lambda row: row["rank"],
            )
            rho = ordered[0]["spearman_vs_epsilon0"]
            lines.append(
                f"- epsilon={epsilon:.2f}: " + " > ".join(VARIANT_LABELS[row["variant"]] for row in ordered)
                + f" (Spearman vs epsilon=0: {rho:.3f})"
            )
    lines.extend(
        [
            "",
            "**Interpretation:** the exact-zero result is not a stable global ranking. Several variants make large "
            "recall/AP jumps as soon as epsilon permits a small validation FP rate, especially VN P4+A4 and most VSB "
            "variants. Exact-zero should therefore be presented as one conservative operating regime, not as a "
            "universal ordering of pipelines.",
        ]
    )
    return "\n".join(lines)


def write_t3_latex(output_dir: Path, summary: list[dict[str, Any]]) -> None:
    lines = []
    for row in summary:
        lines.append(
            f"{row['dataset']} & {VARIANT_LABELS[row['variant']]} & {row['epsilon']:.2f} & "
            f"{row['selected_threshold']:.2f} & {row['test_fp_rate_mean']:.4f} & "
            f"{row['recall_mean']:.3f} & {row['mAP50_mean']:.3f} \\\\"
        )
    latex = output_dir / "latex"
    latex.mkdir(exist_ok=True)
    (latex / "T3_sensitivity_rows.tex").write_text("\n".join(lines) + "\n")


def read_epoch_csv(path: Path) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            converted = {key.strip(): float(value) for key, value in row.items()}
            rows.append(converted)
    return rows


def analyze_convergence(
    generation_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[tuple[str, str, int], list[dict[str, float]]]]:
    per_seed = []
    curves = {}
    for dataset in DATASETS:
        for variant in VARIANTS:
            for seed in SEEDS:
                path = (
                    generation_root / "multiseed" / dataset / "per_seed" / "runs"
                    / f"{variant}_seed{seed}" / "ultralytics" / "train" / "results.csv"
                )
                rows = read_epoch_csv(path)
                if len(rows) != 50:
                    raise RuntimeError(f"Expected 50 epochs: {path} has {len(rows)}")
                curves[(dataset, variant, seed)] = rows
                maps = np.asarray([row["metrics/mAP50(B)"] for row in rows])
                epochs = np.asarray([int(row["epoch"]) for row in rows])
                best_index = int(np.argmax(maps))
                pre_window_best = float(maps[:40].max())
                final_window_best = float(maps[40:].max())
                final_window_slope = float(np.polyfit(epochs[40:], maps[40:], 1)[0])
                late_gain = final_window_best - pre_window_best
                endpoint_change = float(maps[-1] - maps[39])
                plateau = late_gain <= 0.01
                per_seed.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "seed": seed,
                        "best_epoch": int(epochs[best_index]),
                        "best_val_mAP50": float(maps[best_index]),
                        "epoch40_val_mAP50": float(maps[39]),
                        "final_val_mAP50": float(maps[-1]),
                        "pre_final10_best_mAP50": pre_window_best,
                        "final10_best_mAP50": final_window_best,
                        "final10_new_best_gain": late_gain,
                        "epoch40_to_50_change": endpoint_change,
                        "final10_linear_slope_per_epoch": final_window_slope,
                        "positive_final10_trend": final_window_slope > 0.001,
                        "plateau_by_rule": plateau,
                    }
                )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in per_seed:
        grouped[(row["dataset"], row["variant"])].append(row)
    summary = []
    for (dataset, variant), items in sorted(grouped.items()):
        summary.append(
            {
                "dataset": dataset,
                "variant": variant,
                "best_epoch_mean": avg(row["best_epoch"] for row in items),
                "best_epoch_min": min(row["best_epoch"] for row in items),
                "best_epoch_max": max(row["best_epoch"] for row in items),
                "best_val_mAP50_mean": avg(row["best_val_mAP50"] for row in items),
                "best_val_mAP50_std": sd(row["best_val_mAP50"] for row in items),
                "final10_new_best_gain_mean": avg(row["final10_new_best_gain"] for row in items),
                "final10_new_best_gain_max": max(row["final10_new_best_gain"] for row in items),
                "final10_slope_mean": avg(row["final10_linear_slope_per_epoch"] for row in items),
                "positive_trend_seeds": sum(bool(row["positive_final10_trend"]) for row in items),
                "plateau_seeds": sum(bool(row["plateau_by_rule"]) for row in items),
                "all_seeds_plateau": all(bool(row["plateau_by_rule"]) for row in items),
            }
        )
    return per_seed, summary, curves


def task4_report(per_seed: list[dict[str, Any]], summary: list[dict[str, Any]]) -> str:
    lines = [
        "# T4 - Convergence at epoch 50",
        "",
        "Operational rule used for this audit: a seed is considered plateaued when the best mAP50 in epochs 41-50 "
        "improves on the best from epochs 1-40 by at most 0.01. A fitted slope above +0.001 per epoch is reported "
        "separately as a trend warning, but does not by itself imply non-convergence when the curve is merely "
        "recovering from a noisy dip. This is a declared diagnostic rule, not an Ultralytics stopping criterion.",
        "",
        "The two diagnostics are independent: `plateau-rule passes` concerns whether epochs 41-50 establish a "
        "new best more than 0.01 above epochs 1-40, whereas `slope warnings` only records a fitted final-10 slope "
        "above +0.001 per epoch. A noisy recovery can trigger the slope warning while still passing the plateau rule.",
        "",
        "| Dataset | Variant | Best epoch mean [range] | Best mAP50 | Mean/max late gain | Mean final-10 slope | Plateau-rule passes | Slope-warning seeds |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {VARIANT_LABELS[row['variant']]} | {row['best_epoch_mean']:.1f} "
            f"[{row['best_epoch_min']},{row['best_epoch_max']}] | "
            f"{pm(row['best_val_mAP50_mean'], row['best_val_mAP50_std'])} | "
            f"{row['final10_new_best_gain_mean']:+.4f}/{row['final10_new_best_gain_max']:+.4f} | "
            f"{row['final10_slope_mean']:+.5f} | {row['plateau_seeds']}/3 | {row['positive_trend_seeds']}/3 |"
        )
    failures = [row for row in per_seed if not row["plateau_by_rule"]]
    lines.extend(["", "## Direct answer", ""])
    if failures:
        by_dataset = Counter(row["dataset"] for row in failures)
        lines.append(
            f"Not every run had plateaued under the declared rule: {len(failures)}/42 seeds are flagged. "
            "They are listed below and should be treated as a convergence-related limitation rather than hidden."
        )
        lines.append(
            f"VNWoodKnot accounts for {by_dataset['vnwoodknot']}/21 flags, whereas VSB rare-first accounts for "
            f"{by_dataset['vsb_rarefirst']}/21. Thus the 50-epoch cap is a meaningful limitation for VNWoodKnot; "
            "VSB is substantially closer to a plateau."
        )
        vn_failures = Counter(
            row["variant"] for row in failures if row["dataset"] == "vnwoodknot"
        )
        lines.append(
            "The stronger augmentation variants are not uniquely responsible: VN baseline is flagged in "
            f"{vn_failures['baseline']}/3 seeds, A1 in {vn_failures['a1_crop']}/3, A2 in "
            f"{vn_failures['a2_colorjitter']}/3, and P4+A4 in {vn_failures['p4_a4_combined']}/3. "
            "This weakens absolute convergence confidence but does not selectively explain the augmentation ranking."
        )
        lines.append(
            "For convergence wording, cite the plateau rule: VN baseline fails it in 3/3 seeds (0/3 pass), whereas "
            "VN P4+A4 passes in 3/3. Their slope-warning counts are both 2/3 and should not be substituted for the "
            "plateau count."
        )
        for row in failures:
            lines.append(
                f"- {row['dataset']} / {VARIANT_LABELS[row['variant']]} / seed {row['seed']}: "
                f"best epoch {row['best_epoch']}, late gain {row['final10_new_best_gain']:+.4f}, "
                f"late slope {row['final10_linear_slope_per_epoch']:+.5f}."
            )
    else:
        lines.append("All 42 runs satisfy the declared plateau rule.")
    return "\n".join(lines)


def plot_convergence(
    output_dir: Path,
    curves: dict[tuple[str, str, int], list[dict[str, float]]],
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 9, "axes.titlesize": 10})
    colors = plt.get_cmap("tab10").colors
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    metrics = [
        ("metrics/mAP50(B)", "Validation mAP50", "T4_convergence_map50"),
        ("val_loss_sum", "Validation loss (box + cls + DFL)", "T4_convergence_validation_loss"),
    ]
    for metric, ylabel, stem in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), sharex=True)
        for axis, dataset in zip(axes, DATASETS):
            for index, variant in enumerate(VARIANTS):
                arrays = []
                for seed in SEEDS:
                    rows = curves[(dataset, variant, seed)]
                    if metric == "val_loss_sum":
                        values = np.asarray(
                            [row["val/box_loss"] + row["val/cls_loss"] + row["val/dfl_loss"] for row in rows],
                            dtype=float,
                        )
                        values[~np.isfinite(values)] = np.nan
                    else:
                        values = np.asarray([row[metric] for row in rows], dtype=float)
                    arrays.append(values)
                stacked = np.stack(arrays)
                center = np.asarray([
                    float(np.mean(column[np.isfinite(column)])) if np.isfinite(column).any() else np.nan
                    for column in stacked.T
                ])
                spread = np.asarray([
                    float(np.std(column[np.isfinite(column)], ddof=1)) if np.isfinite(column).sum() > 1 else 0.0
                    for column in stacked.T
                ])
                epochs = np.arange(1, len(center) + 1)
                axis.plot(epochs, center, label=VARIANT_LABELS[variant], color=colors[index], linewidth=1.4)
                axis.fill_between(epochs, center - spread, center + spread, color=colors[index], alpha=0.12)
            axis.set_title("VNWoodKnot" if dataset == "vnwoodknot" else "VSB rare-first")
            axis.set_xlabel("Epoch")
            if metric == "val_loss_sum":
                axis.set_yscale("log")
            axis.grid(alpha=0.22, linewidth=0.5)
        axes[0].set_ylabel(ylabel)
        axes[1].legend(fontsize=7, ncol=2, frameon=False)
        fig.tight_layout(w_pad=1.0)
        fig.savefig(figure_dir / f"{stem}.pdf", bbox_inches="tight")
        fig.savefig(figure_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def source_fp_rows(
    payloads: dict[tuple[str, str, int, str], dict[str, Any]]
) -> list[dict[str, Any]]:
    rows = []
    for variant in VARIANTS:
        for seed in SEEDS:
            for split in ("val", "test"):
                payload = payloads[("vsb_strict_clean", variant, seed, split)]
                for threshold in THRESHOLDS:
                    counts = unit_counts(payload, threshold, "source")
                    if len(counts) != 996:
                        raise RuntimeError(f"Expected 996 VSB sources, got {len(counts)}")
                    k = sum(value > 0 for value in counts.values())
                    cp_low, cp_high, wilson_low, wilson_high = binomial_intervals(k, len(counts))
                    rows.append(
                        {
                            "dataset": "vsb_rarefirst",
                            "variant": variant,
                            "seed": seed,
                            "split": split,
                            "threshold": threshold,
                            "n_clean_sources": len(counts),
                            "fp_sources": k,
                            "fp_source_rate": k / len(counts),
                            "false_detections": sum(counts.values()),
                            "fppi_per_source": avg(counts.values()),
                            "cp95_lower": cp_low,
                            "cp95_upper": cp_high,
                            "wilson95_lower": wilson_low,
                            "wilson95_upper": wilson_high,
                        }
                    )
    return rows


def analyze_vsb_sources(
    payloads: dict[tuple[str, str, int, str], dict[str, Any]],
    *,
    draws: int,
    rng_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    sweep = source_fp_rows(payloads)
    locked = []
    for variant in VARIANTS:
        threshold, val_rates = select_threshold(payloads, "vsb_rarefirst", variant, 0.0, level="source")
        for seed in SEEDS:
            clean = payloads[("vsb_strict_clean", variant, seed, "test")]
            counts = unit_counts(clean, threshold, "source")
            detection = payloads[("vsb_rarefirst", variant, seed, "test")]
            metrics = core.evaluate_detection_payload(detection, threshold, 0.50)
            k = sum(value > 0 for value in counts.values())
            cp_low, cp_high, wilson_low, wilson_high = binomial_intervals(k, len(counts))
            locked.append(
                {
                    "dataset": "vsb_rarefirst",
                    "variant": variant,
                    "seed": seed,
                    "selected_threshold": threshold,
                    "validation_fp_source_rate": val_rates[SEEDS.index(seed)],
                    "test_n_clean_sources": len(counts),
                    "test_fp_sources": k,
                    "test_fp_source_rate": k / len(counts),
                    "test_source_fppi": avg(counts.values()),
                    "cp95_lower": cp_low,
                    "cp95_upper": cp_high,
                    "wilson95_lower": wilson_low,
                    "wilson95_upper": wilson_high,
                    **metrics,
                }
            )

    reference_payload = payloads[("vsb_strict_clean", "baseline", 42, "val")]
    val_ids = sorted(unit_maxima(reference_payload, "source"))
    test_ids = sorted(unit_maxima(payloads[("vsb_strict_clean", "baseline", 42, "test")], "source"))
    if len(val_ids) != 996 or len(test_ids) != 996:
        raise RuntimeError("VSB source partition count mismatch")
    rng = np.random.default_rng(rng_seed)
    val_draws = [set(rng.choice(val_ids, size=SUBSAMPLE_SIZE, replace=False)) for _ in range(draws)]
    test_draws = [set(rng.choice(test_ids, size=SUBSAMPLE_SIZE, replace=False)) for _ in range(draws)]
    subsample = []
    for variant in VARIANTS:
        val_max = {
            seed: unit_maxima(payloads[("vsb_strict_clean", variant, seed, "val")], "source")
            for seed in SEEDS
        }
        test_max = {
            seed: unit_maxima(payloads[("vsb_strict_clean", variant, seed, "test")], "source")
            for seed in SEEDS
        }
        for draw_index, (val_subset, test_subset) in enumerate(zip(val_draws, test_draws)):
            threshold = None
            for candidate in THRESHOLDS:
                if all(not any(val_max[seed][key] >= candidate for key in val_subset) for seed in SEEDS):
                    threshold = candidate
                    break
            if threshold is None:
                raise RuntimeError(f"No threshold for VSB source subsample {draw_index}")
            seed_rates = [
                sum(test_max[seed][key] >= threshold for key in test_subset) / SUBSAMPLE_SIZE
                for seed in SEEDS
            ]
            subsample.append(
                {
                    "variant": variant,
                    "draw": draw_index,
                    "rng_seed": rng_seed,
                    "selection_sources": SUBSAMPLE_SIZE,
                    "test_sources": SUBSAMPLE_SIZE,
                    "selected_threshold": threshold,
                    "test_fp_source_rate_seed42": seed_rates[0],
                    "test_fp_source_rate_seed43": seed_rates[1],
                    "test_fp_source_rate_seed44": seed_rates[2],
                    "test_fp_source_rate_seed_mean": avg(seed_rates),
                }
            )
    summary = []
    for variant in VARIANTS:
        items = [row for row in subsample if row["variant"] == variant]
        thresholds = np.asarray([row["selected_threshold"] for row in items])
        rates = np.asarray([row["test_fp_source_rate_seed_mean"] for row in items])
        frequencies = Counter(thresholds)
        summary.append(
            {
                "variant": variant,
                "draws": draws,
                "rng_seed": rng_seed,
                "sample_size": SUBSAMPLE_SIZE,
                "tau_mean": float(thresholds.mean()),
                "tau_median": float(np.median(thresholds)),
                "tau_p025": float(np.percentile(thresholds, 2.5)),
                "tau_p975": float(np.percentile(thresholds, 97.5)),
                "tau_mode": float(frequencies.most_common(1)[0][0]),
                "test_fp_rate_mean": float(rates.mean()),
                "test_fp_rate_std": float(rates.std(ddof=1)),
                "test_fp_rate_p025": float(np.percentile(rates, 2.5)),
                "test_fp_rate_median": float(np.median(rates)),
                "test_fp_rate_p975": float(np.percentile(rates, 97.5)),
            }
        )
    return sweep, locked, subsample, summary


def analyze_vsb_p2_source_overlap(
    payloads: dict[tuple[str, str, int, str], dict[str, Any]],
    source_locked: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    thresholds = {
        float(row["selected_threshold"])
        for row in source_locked
        if row["variant"] == "p2_illumination"
    }
    if len(thresholds) != 1:
        raise RuntimeError(f"Expected one VSB P2 source threshold, got {sorted(thresholds)}")
    threshold = thresholds.pop()
    by_seed: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for seed in SEEDS:
        payload = payloads[("vsb_strict_clean", "p2_illumination", seed, "test")]
        hits: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for image in core.clean_images(payload):
            retained = [pred for pred in image.get("predictions", []) if float(pred["conf"]) >= threshold]
            if not retained:
                continue
            source_id = Path(image_id(image)).stem.split("__", 1)[0]
            hits[source_id].append(
                {
                    "image_id": image_id(image),
                    "confidences": [float(pred["conf"]) for pred in retained],
                }
            )
        by_seed[seed] = dict(hits)

    all_sources = sorted(set().union(*(set(items) for items in by_seed.values())))
    rows = []
    for source_id in all_sources:
        present = [seed for seed in SEEDS if source_id in by_seed[seed]]
        for seed in present:
            records = by_seed[seed][source_id]
            confidences = [conf for record in records for conf in record["confidences"]]
            rows.append(
                {
                    "variant": "p2_illumination",
                    "split": "test",
                    "selected_threshold": threshold,
                    "source_id": source_id,
                    "seed": seed,
                    "fp_tiles": len(records),
                    "false_detections": len(confidences),
                    "max_confidence": max(confidences),
                    "tile_ids": ";".join(record["image_id"] for record in records),
                    "present_seed_count": len(present),
                    "present_seeds": ";".join(str(value) for value in present),
                }
            )
    return rows


def task5_report(
    source_sweep: list[dict[str, Any]],
    source_locked: list[dict[str, Any]],
    subsample_summary: list[dict[str, Any]],
    tile_locked: list[dict[str, Any]],
    args: argparse.Namespace,
    p2_source_overlap: list[dict[str, Any]],
) -> str:
    del source_sweep
    lines = [
        "# T5 - VSB source-level and matched-sample analysis",
        "",
        "Each VSB clean source contributes exactly three tiles. A source is positive for a false alarm when any of "
        "its tiles has a retained prediction. Validation threshold selection uses 996 source IDs and testing uses a "
        "source-disjoint set of 996 IDs.",
        "",
        "## Tile-level versus source-level exact-zero selection",
        "",
        "| Variant | Tile tau | Tile test FP rate | Tile retained R/AP50 | Source tau | Source test FP rate | Source retained R/AP50 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        tiles = [
            row for row in tile_locked
            if row["dataset"] == "vsb_rarefirst" and row["variant"] == variant and row["epsilon"] == 0.0
        ]
        sources = [row for row in source_locked if row["variant"] == variant]
        lines.append(
            f"| {VARIANT_LABELS[variant]} | {tiles[0]['selected_threshold']:.2f} | "
            f"{pm(avg(row['test_fp_rate'] for row in tiles), sd(row['test_fp_rate'] for row in tiles), 4)} | "
            f"{avg(row['recall'] for row in tiles):.3f}/{avg(row['mAP50'] for row in tiles):.3f} | "
            f"{sources[0]['selected_threshold']:.2f} | "
            f"{pm(avg(row['test_fp_source_rate'] for row in sources), sd(row['test_fp_source_rate'] for row in sources), 4)} | "
            f"{avg(row['recall'] for row in sources):.3f}/{avg(row['mAP50'] for row in sources):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Source-level held-out false-alarm intervals (per seed)",
            "",
            "| Variant | tau | Seed | FP sources/N | Source FP rate | Exact CP 95% CI |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in VARIANTS:
        for row in [item for item in source_locked if item["variant"] == variant]:
            lines.append(
                f"| {VARIANT_LABELS[variant]} | {row['selected_threshold']:.2f} | {row['seed']} | "
                f"{row['test_fp_sources']}/{row['test_n_clean_sources']} | {row['test_fp_source_rate']:.6f} | "
                f"[{row['cp95_lower']:.6f}, {row['cp95_upper']:.6f}] |"
            )
    lines.extend(
        [
            "",
            f"## Matched N=75 source subsampling ({args.subsample_draws} draws; RNG seed {args.subsample_seed})",
            "",
            "The same source-ID draws are reused for every variant. Each repetition draws 75 validation sources for "
            "threshold selection and an independent 75 held-out sources for evaluation.",
            "",
            "| Variant | tau median [2.5%,97.5%] | Mean held-out FP rate [2.5%,97.5%] |",
            "|---|---:|---:|",
        ]
    )
    for row in subsample_summary:
        lines.append(
            f"| {VARIANT_LABELS[row['variant']]} | {row['tau_median']:.2f} "
            f"[{row['tau_p025']:.2f},{row['tau_p975']:.2f}] | {row['test_fp_rate_mean']:.4f} "
            f"[{row['test_fp_rate_p025']:.4f},{row['test_fp_rate_p975']:.4f}] |"
        )
    lines.extend(
        [
            "",
            "**Conclusion:** source aggregation does not remove the high-threshold behavior, but matched N=75 "
            "subsamples reveal substantial threshold variability. The VN-versus-VSB contrast is therefore partly "
            "sample-size sensitive and partly a domain/source-level effect; it should not be attributed solely to "
            "tile count.",
        ]
    )
    source_seed_map: dict[str, set[int]] = defaultdict(set)
    for row in p2_source_overlap:
        source_seed_map[row["source_id"]].add(int(row["seed"]))
    shared_43_44 = sorted(
        source_id for source_id, seeds in source_seed_map.items() if {43, 44}.issubset(seeds)
    )
    only_43 = sorted(
        source_id for source_id, seeds in source_seed_map.items() if 43 in seeds and 44 not in seeds
    )
    only_44 = sorted(
        source_id for source_id, seeds in source_seed_map.items() if 44 in seeds and 43 not in seeds
    )
    union_43_44 = set(shared_43_44 + only_43 + only_44)
    lines.extend(
        [
            "",
            "## P2 repeated false-positive sources",
            "",
            "At the validation-selected source-level threshold tau=0.75, seeds 43 and 44 each fire on five "
            "held-out sources (six tiles). Three sources recur in both seeds: "
            f"`{', '.join(shared_43_44)}`. Seed 43-only sources are `{', '.join(only_43)}`; seed 44-only sources "
            f"are `{', '.join(only_44)}`. Thus the excess is partly board-specific but not solely so "
            f"(source-set Jaccard={len(shared_43_44) / len(union_43_44):.3f}). Seed-level tile IDs and "
            "confidences are in `T5_p2_repeated_fp_sources.csv`.",
        ]
    )
    return "\n".join(lines)


def write_t5_latex(output_dir: Path, source: list[dict[str, Any]], tile: list[dict[str, Any]]) -> None:
    lines = []
    for variant in VARIANTS:
        sources = [row for row in source if row["variant"] == variant]
        tiles = [
            row for row in tile
            if row["dataset"] == "vsb_rarefirst" and row["variant"] == variant and row["epsilon"] == 0.0
        ]
        lines.append(
            f"{VARIANT_LABELS[variant]} & {tiles[0]['selected_threshold']:.2f} & "
            f"{avg(row['test_fp_rate'] for row in tiles):.4f} & {sources[0]['selected_threshold']:.2f} & "
            f"{avg(row['test_fp_source_rate'] for row in sources):.4f} & "
            f"{avg(row['recall'] for row in sources):.3f} & {avg(row['mAP50'] for row in sources):.3f} \\\\"
        )
    latex = output_dir / "latex"
    latex.mkdir(exist_ok=True)
    (latex / "T5_vsb_source_rows.tex").write_text("\n".join(lines) + "\n")


def summarize_calibration_complete(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["variant"])].append(row)
    output = []
    for (dataset, variant), items in sorted(grouped.items()):
        out = {"dataset": dataset, "variant": variant, "n_seeds": len(items)}
        for metric in ("num_defective_detections", "d_ece", "signed_gap", "mean_confidence", "empirical_precision"):
            values = [float(row[metric]) for row in items]
            out[f"{metric}_mean"] = avg(values)
            out[f"{metric}_std"] = sd(values)
            out[f"{metric}_median"] = med(values)
            out[f"{metric}_min"] = min(values)
            out[f"{metric}_max"] = max(values)
        counts = [float(row["num_defective_detections"]) for row in items]
        median_count = med(counts)
        flagged = [
            int(row["seed"]) for row in items
            if float(row["num_defective_detections"]) > 1.5 * median_count
            or float(row["num_defective_detections"]) < median_count / 1.5
        ]
        out["detection_count_outlier_seeds_ratio_1p5"] = ";".join(map(str, flagged))
        output.append(out)
    return output


def task6_report(per_seed: list[dict[str, Any]], summary: list[dict[str, Any]]) -> str:
    lines = [
        "# T6 - Calibration across all seeds",
        "",
        "D-ECE is computed only on detections from defective test images at the export floor (confidence >=0.001). "
        "All three seeds are shown; no run is suppressed.",
        "",
        "## Summary",
        "",
        "| Dataset | Variant | D-ECE mean +/- std (median) | Signed gap mean +/- std (median) | Detections mean +/- std (median) | Count outlier seed(s) |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {VARIANT_LABELS[row['variant']]} | "
            f"{pm(row['d_ece_mean'], row['d_ece_std'])} ({row['d_ece_median']:.3f}) | "
            f"{pm(row['signed_gap_mean'], row['signed_gap_std'])} ({row['signed_gap_median']:.3f}) | "
            f"{row['num_defective_detections_mean']:.0f} +/- {row['num_defective_detections_std']:.0f} "
            f"({row['num_defective_detections_median']:.0f}) | "
            f"{row['detection_count_outlier_seeds_ratio_1p5'] or 'none'} |"
        )
    lines.extend(
        [
            "",
            "## Per-seed appendix",
            "",
            "| Dataset | Variant | Seed | n detections | D-ECE | Signed gap | Empirical precision |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_seed:
        lines.append(
            f"| {row['dataset']} | {VARIANT_LABELS[row['variant']]} | {row['seed']} | "
            f"{row['num_defective_detections']} | {row['d_ece']:.6f} | {row['signed_gap']:+.6f} | "
            f"{row['empirical_precision']:.6f} |"
        )
    flagged = [row for row in summary if row["detection_count_outlier_seeds_ratio_1p5"]]
    lines.extend(["", "## Outlier characterization", ""])
    if flagged:
        for row in flagged:
            lines.append(
                f"- {row['dataset']} / {VARIANT_LABELS[row['variant']]}: seed(s) "
                f"{row['detection_count_outlier_seeds_ratio_1p5']} cross the declared 1.5x median count rule; "
                f"range {row['num_defective_detections_min']:.0f}-{row['num_defective_detections_max']:.0f}."
            )
    else:
        lines.append("No seed crosses the declared 1.5x median detection-count rule.")
    return "\n".join(lines)


def write_t6_latex(output_dir: Path, per_seed: list[dict[str, Any]], summary: list[dict[str, Any]]) -> None:
    latex = output_dir / "latex"
    latex.mkdir(exist_ok=True)
    summary_lines = [
        f"{row['dataset']} & {VARIANT_LABELS[row['variant']]} & "
        f"{row['d_ece_mean']:.3f}$\\pm${row['d_ece_std']:.3f} ({row['d_ece_median']:.3f}) & "
        f"{row['signed_gap_mean']:+.3f}$\\pm${row['signed_gap_std']:.3f} ({row['signed_gap_median']:+.3f}) & "
        f"{row['num_defective_detections_mean']:.0f}$\\pm${row['num_defective_detections_std']:.0f} "
        f"({row['num_defective_detections_median']:.0f}) \\\\"
        for row in summary
    ]
    seed_lines = [
        f"{row['dataset']} & {VARIANT_LABELS[row['variant']]} & {row['seed']} & "
        f"{row['num_defective_detections']} & {row['d_ece']:.4f} & {row['signed_gap']:+.4f} \\\\"
        for row in per_seed
    ]
    (latex / "T6_calibration_summary_rows.tex").write_text("\n".join(summary_lines) + "\n")
    (latex / "T6_calibration_seed_rows.tex").write_text("\n".join(seed_lines) + "\n")


def class_pr_curves(payload: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
    class_names = tuple(payload["class_names"])
    class_ids = {name: index for index, name in enumerate(class_names)}
    predictions: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    targets = Counter()
    for image in payload["images"]:
        for gt in image.get("gt_boxes", []):
            targets[class_ids[str(gt[4])]] += 1
        for pred in image.get("predictions", []):
            confidence = float(pred["conf"])
            if confidence < threshold:
                continue
            predictions[int(pred["class_id"])].append(
                (confidence, bool(int(pred["validator_tp_mask"]) & 1))
            )
    rows = []
    for class_id, class_name in enumerate(class_names):
        ordered = sorted(predictions[class_id], key=lambda item: -item[0])
        tp = 0
        fp = 0
        for rank, (confidence, matched) in enumerate(ordered, start=1):
            tp += int(matched)
            fp += int(not matched)
            rows.append(
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "rank": rank,
                    "confidence": confidence,
                    "cumulative_tp": tp,
                    "cumulative_fp": fp,
                    "precision": tp / (tp + fp),
                    "recall": tp / max(targets[class_id], 1),
                    "n_targets_class": targets[class_id],
                }
            )
    return rows


def characterize_retained_pr(
    payloads: dict[tuple[str, str, int, str], dict[str, Any]],
    selections: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    selected = {
        (row["dataset"], row["variant"]): float(row["selected_threshold"])
        for row in selections if row["epsilon"] == 0.0
    }
    all_points = []
    summaries = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            threshold = selected[(dataset, variant)]
            for seed in SEEDS:
                payload = payloads[(dataset, variant, seed, "test")]
                metrics = core.evaluate_detection_payload(payload, threshold, 0.50)
                points = class_pr_curves(payload, threshold)
                for point in points:
                    all_points.append(
                        {"dataset": dataset, "variant": variant, "seed": seed, "threshold": threshold, **point}
                    )
                precision_values = [float(point["precision"]) for point in points]
                endpoint_by_class = []
                for class_name in payload["class_names"]:
                    class_points = [point for point in points if point["class_name"] == class_name]
                    if class_points:
                        endpoint_by_class.append(
                            class_points[-1]["precision"] * class_points[-1]["recall"]
                        )
                    else:
                        endpoint_by_class.append(0.0)
                endpoint_product = metrics["precision"] * metrics["recall"]
                macro_endpoint_product = avg(endpoint_by_class)
                summaries.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "seed": seed,
                        "selected_threshold": threshold,
                        "n_retained": metrics["n_retained"],
                        "endpoint_precision": metrics["precision"],
                        "endpoint_recall": metrics["recall"],
                        "retained_mAP50": metrics["mAP50"],
                        "rank_precision_mean": avg(precision_values),
                        "rank_precision_min": min(precision_values) if precision_values else float("nan"),
                        "rank_precision_max": max(precision_values) if precision_values else float("nan"),
                        "rank_precision_variance": statistics.pvariance(precision_values) if precision_values else float("nan"),
                        "micro_endpoint_p_times_r": endpoint_product,
                        "macro_class_endpoint_p_times_r": macro_endpoint_product,
                        "ap_minus_micro_p_times_r": metrics["mAP50"] - endpoint_product,
                        "ap_minus_macro_p_times_r": metrics["mAP50"] - macro_endpoint_product,
                        "micro_relation_within_0p02": abs(metrics["mAP50"] - endpoint_product) <= 0.02,
                        "macro_relation_within_0p02": abs(metrics["mAP50"] - macro_endpoint_product) <= 0.02,
                    }
                )
    relation = []
    for dataset in (*DATASETS, "all"):
        items = summaries if dataset == "all" else [row for row in summaries if row["dataset"] == dataset]
        ap = np.asarray([row["retained_mAP50"] for row in items])
        micro = np.asarray([row["micro_endpoint_p_times_r"] for row in items])
        macro = np.asarray([row["macro_class_endpoint_p_times_r"] for row in items])
        relation.append(
            {
                "dataset": dataset,
                "n_cells": len(items),
                "micro_mae": float(np.mean(np.abs(ap - micro))),
                "micro_max_abs_error": float(np.max(np.abs(ap - micro))),
                "micro_within_0p02": int(np.sum(np.abs(ap - micro) <= 0.02)),
                "micro_correlation": float(np.corrcoef(ap, micro)[0, 1]),
                "macro_mae": float(np.mean(np.abs(ap - macro))),
                "macro_max_abs_error": float(np.max(np.abs(ap - macro))),
                "macro_within_0p02": int(np.sum(np.abs(ap - macro) <= 0.02)),
                "macro_correlation": float(np.corrcoef(ap, macro)[0, 1]),
            }
        )
    return all_points, summaries, relation


def task7_report(summary: list[dict[str, Any]], relation: list[dict[str, Any]]) -> str:
    lines = [
        "# T7 - Characterization of retained AP50",
        "",
        "Retained AP50 is recomputed from the confidence-ranked detections retained at tau, using the saved "
        "DetectionValidator IoU=0.5 TP assignment and Ultralytics 8.4.60's 101-point interpolated precision "
        "envelope. It is not a single operating-point product.",
        "",
        "| Dataset | Variant | Seed | tau | Endpoint P/R | AP50 | Rank-P mean [min,max], var | P*R | AP-(P*R) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {VARIANT_LABELS[row['variant']]} | {row['seed']} | "
            f"{row['selected_threshold']:.2f} | {row['endpoint_precision']:.3f}/{row['endpoint_recall']:.3f} | "
            f"{row['retained_mAP50']:.3f} | {row['rank_precision_mean']:.3f} "
            f"[{row['rank_precision_min']:.3f},{row['rank_precision_max']:.3f}], "
            f"{row['rank_precision_variance']:.4f} | {row['micro_endpoint_p_times_r']:.3f} | "
            f"{row['ap_minus_micro_p_times_r']:+.3f} |"
        )
    lines.extend(["", "## Empirical relation", "", "| Scope | cells | MAE AP vs micro P*R | max error | within 0.02 | correlation |", "|---|---:|---:|---:|---:|---:|"])
    for row in relation:
        lines.append(
            f"| {row['dataset']} | {row['n_cells']} | {row['micro_mae']:.4f} | "
            f"{row['micro_max_abs_error']:.4f} | {row['micro_within_0p02']}/{row['n_cells']} | "
            f"{row['micro_correlation']:.4f} |"
        )
    p4_rows = [row for row in summary if row["dataset"] == "vnwoodknot" and row["variant"] == "p4_a4_combined"]
    lines.extend(
        [
            "",
            f"VN P4+A4 satisfies the endpoint approximation within 0.02 in "
            f"{sum(bool(row['micro_relation_within_0p02']) for row in p4_rows)}/3 seeds because its retained "
            "precision is near one and nearly flat. Across all cells, however, the approximation holds in only "
            f"{next(row['micro_within_0p02'] for row in relation if row['dataset'] == 'all')}/42 cases. The "
            "P4+A4 AP-recall coincidence is therefore systematic under the stated high-flat-precision condition, "
            "but the endpoint product is not a general replacement for retained AP.",
            "",
            "## Defensible methods wording",
            "",
            "> Retained AP50 was computed after discarding predictions below the validation-selected operating "
            "threshold, while preserving the remaining confidence ranking and the validator's class-aware IoU=0.5 "
            "matches. AP was then integrated from the resulting class-wise 101-point interpolated precision "
            "envelopes. Consequently, retained AP50 is an area metric over the truncated ranking, not a single-point "
            "precision-recall product. When the precision envelope is approximately constant at p over the achieved "
            "recall range [0,R_tau], the area approaches p R_tau; thus AP50 and retained recall converge "
            "systematically as retained precision approaches one.",
            "",
            "The endpoint product is only an approximation. The CSV identifies cells where it fails because precision "
            "varies materially over the retained ranking or because macro class-wise AP and micro endpoint P/R weight "
            "classes differently.",
        ]
    )
    return "\n".join(lines)


def plot_retained_relation(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, ax = plt.subplots(figsize=(5.2, 4.3))
    markers = {"vnwoodknot": "o", "vsb_rarefirst": "s"}
    for dataset in DATASETS:
        items = [row for row in rows if row["dataset"] == dataset]
        ax.scatter(
            [row["micro_endpoint_p_times_r"] for row in items],
            [row["retained_mAP50"] for row in items],
            alpha=0.75,
            s=28,
            marker=markers[dataset],
            label="VNWoodKnot" if dataset == "vnwoodknot" else "VSB rare-first",
        )
    ax.plot([0, 1], [0, 1], color="black", linewidth=0.8, linestyle="--")
    ax.set(xlabel="Endpoint precision x recall", ylabel="Retained AP50", xlim=(0, 1), ylim=(0, 1))
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(exist_ok=True)
    fig.savefig(figure_dir / "T7_retained_ap_vs_precision_recall_product.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "T7_retained_ap_vs_precision_recall_product.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_deprecated_payloads(root: Path) -> dict[tuple[str, str, int, str], dict[str, Any]]:
    output = {}
    for path in sorted((root / "deprecated_audit" / "predictions").rglob("*_predictions.json")):
        payload = json.loads(path.read_text())
        key = (str(payload["dataset"]), str(payload["variant"]), int(payload["seed"]), str(payload["split"]))
        output[key] = payload
    expected = {
        (dataset, variant, seed, split)
        for dataset in DATASETS
        for variant in ("a1_crop", "a2_colorjitter", "p4_a4_combined")
        for seed in SEEDS
        for split in ("val", "test")
    }
    expected.update(
        {
            ("vsb_strict_clean", variant, seed, split)
            for variant in ("a1_crop", "a2_colorjitter", "p4_a4_combined")
            for seed in SEEDS
            for split in ("val", "test")
        }
    )
    if set(output) != expected:
        raise RuntimeError(
            "Deprecated payload inventory mismatch: "
            f"missing={sorted(expected - set(output))}, extra={sorted(set(output) - expected)}"
        )
    return output


def select_threshold_from_explicit_clean(
    payloads: dict[tuple[str, str, int, str], dict[str, Any]],
    dataset: str,
    variant: str,
) -> tuple[float, list[float]]:
    for threshold in THRESHOLDS:
        rates = []
        for seed in SEEDS:
            images = core.clean_images(payloads[(dataset, variant, seed, "val")])
            if not images:
                raise RuntimeError(f"No clean images in deprecated proxy {dataset}/{variant}/seed{seed}")
            rates.append(sum(max_confidence(image) >= threshold for image in images) / len(images))
        if all(rate == 0.0 for rate in rates):
            return threshold, rates
    raise RuntimeError(f"No deprecated exact-zero threshold for {dataset}/{variant}")


def best_logged_map(path: Path) -> float:
    rows = read_epoch_csv(path)
    return max(row["metrics/mAP50(B)"] for row in rows)


def analyze_deprecated_impact(
    generation_root: Path,
    primary: dict[tuple[str, str, int, str], dict[str, Any]],
    deprecated: dict[tuple[str, str, int, str], dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    bias_rows = []
    fair = core.read_csv(generation_root / "fair_eval" / "fair_metrics.csv")
    deprecated_fair = core.read_csv(generation_root / "deprecated_audit" / "fair_eval" / "fair_metrics.csv")
    for dataset in DATASETS:
        clean_dataset = "vnwoodknot" if dataset == "vnwoodknot" else "vsb_strict_clean"
        clean_basis = (
            "VN canonical 75 clean images"
            if dataset == "vnwoodknot"
            else "VSB source-disjoint strict-clean split: 996 sources / 2,988 tiles"
        )
        for variant in ("a1_crop", "a2_colorjitter", "p4_a4_combined"):
            corrected_tau, _ = select_threshold_from_explicit_clean(primary, clean_dataset, variant)
            deprecated_tau, _ = select_threshold_from_explicit_clean(deprecated, clean_dataset, variant)
            for seed in SEEDS:
                corrected_detection = primary[(dataset, variant, seed, "test")]
                deprecated_detection = deprecated[(dataset, variant, seed, "test")]
                corrected_metrics = core.evaluate_detection_payload(corrected_detection, corrected_tau, 0.50)
                deprecated_metrics = core.evaluate_detection_payload(deprecated_detection, deprecated_tau, 0.50)
                corrected_clean = core.clean_images(primary[(clean_dataset, variant, seed, "test")])
                deprecated_clean = core.clean_images(deprecated[(clean_dataset, variant, seed, "test")])
                corrected_max = np.asarray([max_confidence(image) for image in corrected_clean])
                deprecated_max = np.asarray([max_confidence(image) for image in deprecated_clean])
                rows.append(
                    {
                        "dataset": dataset,
                        "clean_evaluation_basis": clean_basis,
                        "variant": variant,
                        "seed": seed,
                        "corrected_tau": corrected_tau,
                        "deprecated_tau": deprecated_tau,
                        "delta_tau_corrected_minus_deprecated": corrected_tau - deprecated_tau,
                        "corrected_recall": corrected_metrics["recall"],
                        "deprecated_recall": deprecated_metrics["recall"],
                        "delta_recall": corrected_metrics["recall"] - deprecated_metrics["recall"],
                        "corrected_retained_AP50": corrected_metrics["mAP50"],
                        "deprecated_retained_AP50": deprecated_metrics["mAP50"],
                        "delta_retained_AP50": corrected_metrics["mAP50"] - deprecated_metrics["mAP50"],
                        "corrected_clean_mean_max_conf": float(corrected_max.mean()),
                        "deprecated_clean_mean_max_conf": float(deprecated_max.mean()),
                        "delta_clean_mean_max_conf": float(corrected_max.mean() - deprecated_max.mean()),
                        "corrected_clean_p95_max_conf": float(np.percentile(corrected_max, 95)),
                        "deprecated_clean_p95_max_conf": float(np.percentile(deprecated_max, 95)),
                        "delta_clean_p95_max_conf": float(np.percentile(corrected_max, 95) - np.percentile(deprecated_max, 95)),
                    }
                )

                old_results = (
                    generation_root / "deprecated_checkpoints" / "multiseed" / dataset / "per_seed" / "runs"
                    / f"{variant}_seed{seed}" / "ultralytics" / "train" / "results.csv"
                )
                new_results = (
                    generation_root / "multiseed" / dataset / "per_seed" / "runs"
                    / f"{variant}_seed{seed}" / "ultralytics" / "train" / "results.csv"
                )
                old_fair_val = next(
                    float(row["mAP50"]) for row in deprecated_fair
                    if row["dataset"] == dataset and row["variant"] == variant
                    and int(row["seed"]) == seed and row["split"] == "val"
                )
                new_fair_val = next(
                    float(row["mAP50"]) for row in fair
                    if row["dataset"] == dataset and row["variant"] == variant
                    and int(row["seed"]) == seed and row["split"] == "val"
                )
                old_logged = best_logged_map(old_results)
                new_logged = best_logged_map(new_results)
                bias_rows.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "seed": seed,
                        "deprecated_best_logged_validation_mAP50": old_logged,
                        "deprecated_checkpoint_fair_validation_mAP50": old_fair_val,
                        "deprecated_logged_minus_fair": old_logged - old_fair_val,
                        "corrected_best_logged_validation_mAP50": new_logged,
                        "corrected_checkpoint_fair_validation_mAP50": new_fair_val,
                        "corrected_logged_minus_fair": new_logged - new_fair_val,
                    }
                )
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["dataset"], row["variant"])].append(row)
    standard_delta = core.read_csv(
        generation_root / "deprecated_audit" / "comparison" / "deprecated_vs_corrected_summary.csv"
    )
    summary = []
    for (dataset, variant), items in sorted(grouped.items()):
        out = {
            "dataset": dataset,
            "clean_evaluation_basis": items[0]["clean_evaluation_basis"],
            "variant": variant,
            "corrected_tau": items[0]["corrected_tau"],
            "deprecated_tau": items[0]["deprecated_tau"],
            "delta_tau": items[0]["delta_tau_corrected_minus_deprecated"],
        }
        for metric in ("delta_recall", "delta_retained_AP50", "delta_clean_mean_max_conf", "delta_clean_p95_max_conf"):
            out[f"{metric}_mean"] = avg(row[metric] for row in items)
            out[f"{metric}_std"] = sd(row[metric] for row in items)
        test_standard = next(
            row for row in standard_delta
            if row["dataset"] == dataset and row["variant"] == variant and row["split"] == "test"
        )
        out["delta_standard_test_mAP50_mean"] = float(test_standard["delta_mAP50_mean"])
        out["delta_standard_test_mAP50_std"] = float(test_standard["delta_mAP50_std"])
        summary.append(out)
    return rows, summary, bias_rows


def task8_report(
    rows: list[dict[str, Any]],
    summary: list[dict[str, Any]],
    bias: list[dict[str, Any]],
) -> str:
    lines = [
        "# T8 - Deprecated-checkpoint impact",
        "",
        "Corrected minus deprecated is reported throughout. VN uses its 75 canonical clean images. VSB uses the "
        "same source-disjoint strict-clean validation and test halves for corrected and deprecated checkpoints "
        "(996 sources / 2,988 tiles per half). The former 276-empty-tile rare-first proxy is superseded and must "
        "not be cited.",
        "",
        "| Dataset | Variant | Clean basis | tau corrected/old (delta) | delta standard test mAP50 | delta retained R | delta retained AP50 | delta clean mean-max | delta clean P95-max |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(
            f"| {row['dataset']} | {VARIANT_LABELS[row['variant']]} | {row['clean_evaluation_basis']} | "
            f"{row['corrected_tau']:.2f}/{row['deprecated_tau']:.2f} ({row['delta_tau']:+.2f}) | "
            f"{pm(row['delta_standard_test_mAP50_mean'], row['delta_standard_test_mAP50_std'])} | "
            f"{pm(row['delta_recall_mean'], row['delta_recall_std'])} | "
            f"{pm(row['delta_retained_AP50_mean'], row['delta_retained_AP50_std'])} | "
            f"{pm(row['delta_clean_mean_max_conf_mean'], row['delta_clean_mean_max_conf_std'], 4)} | "
            f"{pm(row['delta_clean_p95_max_conf_mean'], row['delta_clean_p95_max_conf_std'], 4)} |"
        )
    lines.extend(
        [
            "",
            "## Does the crop-easier-validation hypothesis hold?",
            "",
            "The logged-minus-fair validation gap is diagnostic evidence, not a controlled causal decomposition: "
            "the logged maximum can occur at a different epoch from the selected best checkpoint. Seed-level values "
            "are retained in `T8_validation_selection_bias.csv`.",
            "",
            "| Dataset | Variant | Deprecated logged-fair gap mean | Corrected logged-fair gap mean |",
            "|---|---|---:|---:|",
        ]
    )
    for dataset in DATASETS:
        for variant in ("a1_crop", "a2_colorjitter", "p4_a4_combined"):
            items = [row for row in bias if row["dataset"] == dataset and row["variant"] == variant]
            lines.append(
                f"| {dataset} | {VARIANT_LABELS[variant]} | "
                f"{avg(row['deprecated_logged_minus_fair'] for row in items):+.4f} | "
                f"{avg(row['corrected_logged_minus_fair'] for row in items):+.4f} |"
            )
    lines.extend(
        [
            "",
            "The data support a large A1-specific checkpoint-generation effect on VN standard test mAP50, while A2 "
            "is nearly unchanged. However, this alone does not prove that cropping made every validation image easier: "
            "it shows that applying crop to validation materially altered checkpoint selection/generalization, whereas "
            "colour jitter did not. Use that narrower wording unless image-level paired validation evidence is added.",
        ]
    )
    return "\n".join(lines)


def write_t8_latex(output_dir: Path, summary: list[dict[str, Any]]) -> None:
    lines = [
        f"{row['dataset']} & {VARIANT_LABELS[row['variant']]} & "
        f"{row['delta_tau']:+.2f} & {row['delta_standard_test_mAP50_mean']:+.3f} & "
        f"{row['delta_recall_mean']:+.3f} & {row['delta_retained_AP50_mean']:+.3f} & "
        f"{row['delta_clean_p95_max_conf_mean']:+.3f} \\\\"
        for row in summary
    ]
    latex = output_dir / "latex"
    latex.mkdir(exist_ok=True)
    (latex / "T8_deprecated_impact_rows.tex").write_text("\n".join(lines) + "\n")


def build_master_report(output_dir: Path, reproduction: list[dict[str, Any]]) -> str:
    task_files = sorted(output_dir.glob("T[0-8]_*.md"))
    lines = [
        "# Reviewer Offline Analysis - Master Report",
        "",
        "- Frozen generation was read only.",
        "- No training or GPU inference was run.",
        f"- Base AP reproduction remained PASS ({len(reproduction)}/{len(reproduction)}), maximum absolute delta "
        f"{max(abs(float(row['delta_mAP50'])) for row in reproduction):.9f}.",
        "- Exact binomial intervals are per seed; seeds are never pooled.",
        "",
        "## Reports",
        "",
    ]
    for path in task_files:
        title = path.read_text().splitlines()[0].removeprefix("# ")
        lines.append(f"- [{title}]({path.name})")
    lines.extend(
        [
            "",
            "## Manuscript conflicts requiring correction",
            "",
            "1. Bootstrap intervals for all-zero clean outcomes must be replaced by exact binomial intervals.",
            "2. Exact-zero operating points are brittle and should not be presented as a stable global ranking.",
            "3. VN P4+A4 tau=0.70 is driven by one seed/image, not a consistent three-seed collapse.",
            "4. Retained AP50 is a truncated PR-area metric; AP approximately equals p times retained recall only "
            "when the precision envelope is nearly constant.",
            "5. Deprecated and corrected VSB augmentation checkpoints are compared on the same source-disjoint "
            "strict-clean halves; the former 276-empty-tile proxy is superseded.",
            "6. The superseded g1 attempt should be disclosed in provenance, including why g2 replaced it.",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
