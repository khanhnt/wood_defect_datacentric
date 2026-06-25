#!/usr/bin/env python3
"""Aggregate multiseed YOLOv8s experiment results.

Expected run layout:

results/multiseed/{dataset}/per_seed/runs/{variant}_seed{seed}/ultralytics/train/results.csv

The script reads the final epoch row from each Ultralytics results.csv,
aggregates metrics across seeds, and writes paper-ready CSV summaries.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import re
import statistics
from typing import Iterable


EXPECTED_SEEDS = (42, 43, 44)
DATASETS = ("vnwoodknot", "vsb_rarefirst")
METRICS = ("precision", "recall", "mAP50", "mAP50_95")
RESULT_COLUMNS = {
    "precision": "metrics/precision(B)",
    "recall": "metrics/recall(B)",
    "mAP50": "metrics/mAP50(B)",
    "mAP50_95": "metrics/mAP50-95(B)",
}
RUN_NAME_RE = re.compile(r"^(?P<variant>.+)_seed(?P<seed>\d+)$")

VARIANT_ORDER = {
    "baseline": 0,
    "p1_clahe": 1,
    "p2_illumination": 2,
    "p3_unsharp": 3,
    "a1_crop": 4,
    "a2_colorjitter": 5,
    "p4_a4_combined": 6,
}

VARIANT_LABELS = {
    "baseline": "Baseline",
    "p1_clahe": "P1 CLAHE",
    "p2_illumination": "P2 illumination",
    "p3_unsharp": "P3 unsharp",
    "a1_crop": "A1 crop",
    "a2_colorjitter": "A2 color jitter",
    "p4_a4_combined": "P4+A4 combined",
}

EXPECTED_VARIANTS = {
    "vnwoodknot": ("baseline", "p2_illumination", "a1_crop", "a2_colorjitter", "p4_a4_combined"),
    "vsb_rarefirst": (
        "baseline",
        "p1_clahe",
        "p2_illumination",
        "p3_unsharp",
        "a1_crop",
        "a2_colorjitter",
        "p4_a4_combined",
    ),
}

OLD_VN_EXPERIMENT_MAP = {
    "vn_t0_yolov8s_baseline_e50": "baseline",
    "vn_yolov8s_p2_illumination_e50": "p2_illumination",
    "vn_yolov8s_a1_crop_e50": "a1_crop",
    "vn_yolov8s_a2_colorjitter_e50": "a2_colorjitter",
    "vn_yolov8s_p4_a4_combined_e50": "p4_a4_combined",
}

OLD_VSB_VARIANT_MAP = {
    "baseline": "baseline",
    "P1_CLAHE": "p1_clahe",
    "P2_illumination": "p2_illumination",
    "P3_unsharp": "p3_unsharp",
    "A1_crop": "a1_crop",
    "A2_colorjitter": "a2_colorjitter",
    "P4_A4_combined": "p4_a4_combined",
}


@dataclass(frozen=True)
class SeedMetrics:
    dataset: str
    variant: str
    seed: int
    precision: float
    recall: float
    mAP50: float
    mAP50_95: float
    epoch: int
    run_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument(
        "--old-results-dir",
        type=Path,
        default=None,
        help="Optional old single-seed results directory for seed-42 comparison.",
    )
    parser.add_argument("--std-warning-threshold", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir.expanduser().resolve()
    if not results_dir.exists():
        raise SystemExit(f"Results directory does not exist: {results_dir}")

    records = scan_multiseed_results(results_dir)
    if not records:
        raise SystemExit(f"No completed multiseed runs found under: {results_dir / 'multiseed'}")

    by_dataset: dict[str, list[SeedMetrics]] = {dataset: [] for dataset in DATASETS}
    for record in records:
        by_dataset.setdefault(record.dataset, []).append(record)

    all_summary_rows: list[dict[str, str]] = []
    for dataset in DATASETS:
        dataset_records = sorted(by_dataset.get(dataset, []), key=record_sort_key)
        write_per_seed_csv(results_dir, dataset, dataset_records)
        summary_rows = summarize_dataset(dataset, dataset_records)
        all_summary_rows.extend(summary_rows)
        write_dataset_summary(results_dir, dataset, summary_rows)
        print_summary_table(dataset, summary_rows)

    all_summary_path = results_dir / "multiseed" / "all_variants_multiseed_summary.csv"
    write_summary_csv(all_summary_path, all_summary_rows)
    print(f"Wrote: {all_summary_path}")

    write_empty_vn_test_summary_if_missing(results_dir)
    run_validation_checks(all_summary_rows, threshold=args.std_warning_threshold)
    compare_seed42_to_old_results(results_dir, records, old_results_dir=args.old_results_dir)


def scan_multiseed_results(results_dir: Path) -> list[SeedMetrics]:
    records: list[SeedMetrics] = []
    for dataset in DATASETS:
        runs_root = results_dir / "multiseed" / dataset / "per_seed" / "runs"
        if not runs_root.exists():
            print(f"WARNING: missing runs directory: {runs_root}")
            continue
        for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
            match = RUN_NAME_RE.match(run_dir.name)
            if not match:
                print(f"WARNING: ignoring unexpected run directory name: {run_dir}")
                continue
            variant = match.group("variant")
            seed = int(match.group("seed"))
            train_dir = run_dir / "ultralytics" / "train"
            results_csv = train_dir / "results.csv"
            best_pt = train_dir / "weights" / "best.pt"
            if not results_csv.exists():
                print(f"WARNING: missing results.csv: {results_csv}")
                continue
            if not best_pt.exists():
                print(f"WARNING: missing best.pt for completed-looking run: {run_dir}")
            try:
                records.append(read_final_epoch_metrics(dataset, variant, seed, run_dir, results_csv))
            except Exception as exc:
                print(f"WARNING: failed to parse {results_csv}: {exc}")
    return records


def read_final_epoch_metrics(dataset: str, variant: str, seed: int, run_dir: Path, results_csv: Path) -> SeedMetrics:
    rows = list(csv.DictReader(results_csv.open("r", encoding="utf-8")))
    if not rows:
        raise ValueError("empty results.csv")
    row = {key.strip(): value for key, value in rows[-1].items()}
    return SeedMetrics(
        dataset=dataset,
        variant=variant,
        seed=seed,
        precision=parse_float(row[RESULT_COLUMNS["precision"]]),
        recall=parse_float(row[RESULT_COLUMNS["recall"]]),
        mAP50=parse_float(row[RESULT_COLUMNS["mAP50"]]),
        mAP50_95=parse_float(row[RESULT_COLUMNS["mAP50_95"]]),
        epoch=int(float(row.get("epoch", row.get("epoch/epoch", 0)))),
        run_dir=run_dir,
    )


def write_per_seed_csv(results_dir: Path, dataset: str, records: list[SeedMetrics]) -> None:
    path = results_dir / "multiseed" / dataset / "per_seed" / "per_seed_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["variant", "seed", "precision", "recall", "mAP50", "mAP50_95"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "variant": record.variant,
                    "seed": record.seed,
                    "precision": format_float(record.precision),
                    "recall": format_float(record.recall),
                    "mAP50": format_float(record.mAP50),
                    "mAP50_95": format_float(record.mAP50_95),
                }
            )
    print(f"Wrote: {path}")


def summarize_dataset(dataset: str, records: list[SeedMetrics]) -> list[dict[str, str]]:
    rows = []
    expected = set(EXPECTED_VARIANTS.get(dataset, ()))
    variants = sorted(expected | {record.variant for record in records}, key=variant_sort_key)
    for variant in variants:
        variant_records = sorted((record for record in records if record.variant == variant), key=lambda item: item.seed)
        row: dict[str, str] = {
            "dataset": dataset,
            "variant": variant,
            "variant_label": VARIANT_LABELS.get(variant, variant),
            "n_seeds": str(len(variant_records)),
            "seeds": " ".join(str(record.seed) for record in variant_records),
            "complete": str(set(record.seed for record in variant_records) == set(EXPECTED_SEEDS)).lower(),
        }
        for metric in METRICS:
            values = [getattr(record, metric) for record in variant_records]
            row[f"{metric}_mean"] = format_float(mean(values))
            row[f"{metric}_std"] = format_float(std(values)) if values else ""
        rows.append(row)
    return rows


def write_dataset_summary(results_dir: Path, dataset: str, rows: list[dict[str, str]]) -> None:
    if dataset == "vnwoodknot":
        path = results_dir / "multiseed" / dataset / "summary_validation.csv"
    else:
        path = results_dir / "multiseed" / dataset / "summary.csv"
    write_summary_csv(path, rows)
    print(f"Wrote: {path}")


def write_summary_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "variant",
        "variant_label",
        "n_seeds",
        "seeds",
        "complete",
        "precision_mean",
        "precision_std",
        "recall_mean",
        "recall_std",
        "mAP50_mean",
        "mAP50_std",
        "mAP50_95_mean",
        "mAP50_95_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_empty_vn_test_summary_if_missing(results_dir: Path) -> None:
    path = results_dir / "multiseed" / "vnwoodknot" / "summary_test.csv"
    if path.exists():
        print(f"Found existing VNWoodKnot test summary: {path}")
        return
    write_summary_csv(path, [])
    print(f"WARNING: no VNWoodKnot test metrics found; wrote empty CSV: {path}")


def print_summary_table(dataset: str, rows: list[dict[str, str]]) -> None:
    title = "VSB Rare-First" if dataset == "vsb_rarefirst" else "VNWoodKnot Validation"
    print()
    print(f"=== {title} (mean +/- std, expected n=3 seeds) ===")
    print(f"{'Variant':<20} | {'Precision':<15} | {'Recall':<15} | {'mAP50':<15} | {'mAP50-95':<15}")
    print("-" * 92)
    for row in rows:
        print(
            f"{row['variant_label']:<20} | "
            f"{fmt_pm(row['precision_mean'], row['precision_std']):<15} | "
            f"{fmt_pm(row['recall_mean'], row['recall_std']):<15} | "
            f"{fmt_pm(row['mAP50_mean'], row['mAP50_std']):<15} | "
            f"{fmt_pm(row['mAP50_95_mean'], row['mAP50_95_std']):<15}"
        )
    print()


def run_validation_checks(rows: list[dict[str, str]], *, threshold: float) -> None:
    for row in rows:
        if int(row["n_seeds"]) < len(EXPECTED_SEEDS):
            print(
                "WARNING: "
                f"{row['dataset']} {row['variant']} has {row['n_seeds']} seeds "
                f"({row['seeds']}); expected {len(EXPECTED_SEEDS)}."
            )
        for metric in METRICS:
            if not row[f"{metric}_std"]:
                continue
            value = parse_float(row[f"{metric}_std"])
            if value > threshold:
                print(
                    "WARNING: high seed variability: "
                    f"{row['dataset']} {row['variant']} {metric}_std={value:.4f} > {threshold:.4f}"
                )


def compare_seed42_to_old_results(results_dir: Path, records: list[SeedMetrics], old_results_dir: Path | None) -> None:
    old_dir = old_results_dir.expanduser().resolve() if old_results_dir else find_old_results_dir(results_dir)
    if old_dir is None:
        print("WARNING: no old single-seed results directory found for seed-42 comparison.")
        return
    old_metrics = load_old_single_seed_metrics(old_dir)
    if not old_metrics:
        print(f"WARNING: no old single-seed summary metrics found in {old_dir}.")
        return

    print()
    print(f"=== Seed-42 comparison against old single-seed summaries: {old_dir} ===")
    seed42 = [record for record in records if record.seed == 42]
    for record in sorted(seed42, key=record_sort_key):
        key = (record.dataset, record.variant)
        if key not in old_metrics:
            continue
        old = old_metrics[key]
        diffs = []
        for metric in METRICS:
            current = getattr(record, metric)
            previous = old[metric]
            delta = current - previous
            if abs(delta) > 1e-6:
                diffs.append(f"{metric}: {previous:.4f}->{current:.4f} ({delta:+.4f})")
        if diffs:
            print(f"{record.dataset} {record.variant}: " + "; ".join(diffs))
    print("Note: differences are expected when batch size or generated augmentation seeds changed.")
    print()


def find_old_results_dir(results_dir: Path) -> Path | None:
    candidates = sorted(results_dir.parent.glob("results_old_bs16*"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def load_old_single_seed_metrics(old_dir: Path) -> dict[tuple[str, str], dict[str, float]]:
    metrics: dict[tuple[str, str], dict[str, float]] = {}
    vn_summary = old_dir / "summaries" / "vn_experiment_comparison_summary.csv"
    if vn_summary.exists():
        with vn_summary.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                variant = OLD_VN_EXPERIMENT_MAP.get(row.get("experiment_id", ""))
                if not variant:
                    continue
                metrics[("vnwoodknot", variant)] = {
                    "precision": parse_float(row["val_best_precision"]),
                    "recall": parse_float(row["val_best_recall"]),
                    "mAP50": parse_float(row["val_best_mAP50"]),
                    "mAP50_95": parse_float(row["val_best_mAP50_95"]),
                }

    vsb_summary = old_dir / "summaries" / "vsb_rare_first_batch16_final_summary.csv"
    if vsb_summary.exists():
        with vsb_summary.open("r", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                variant = OLD_VSB_VARIANT_MAP.get(row.get("variant", ""))
                if not variant:
                    continue
                metrics[("vsb_rarefirst", variant)] = {
                    "precision": parse_float(row["precision"]),
                    "recall": parse_float(row["recall"]),
                    "mAP50": parse_float(row["map50"]),
                    "mAP50_95": parse_float(row["map50_95"]),
                }
    return metrics


def record_sort_key(record: SeedMetrics) -> tuple[int, int, int]:
    dataset_index = DATASETS.index(record.dataset) if record.dataset in DATASETS else 99
    return dataset_index, variant_sort_key(record.variant), record.seed


def variant_sort_key(variant: str) -> int:
    return VARIANT_ORDER.get(variant, 99)


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return sum(values) / len(values) if values else float("nan")


def std(values: Iterable[float]) -> float:
    values = list(values)
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def parse_float(value: str | float | int) -> float:
    return float(str(value).strip())


def format_float(value: float) -> str:
    if value != value:
        return ""
    return f"{value:.6f}"


def fmt_pm(mean_value: str, std_value: str) -> str:
    if not mean_value or not std_value:
        return "n/a"
    return f"{parse_float(mean_value):.3f} +/- {parse_float(std_value):.3f}"


if __name__ == "__main__":
    main()
