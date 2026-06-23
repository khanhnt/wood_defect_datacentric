#!/usr/bin/env python3
"""Audit Ultralytics training args for multiseed runs.

The script is CPU-only and reads:

results/multiseed/{dataset}/per_seed/runs/{variant}_seed{seed}/ultralytics/train/args.yaml

It is useful for confirming that paper-method settings such as batch size,
image size, epochs, seed, and deterministic mode match the claimed protocol.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import re
from typing import Any

import yaml


DATASETS = ("vnwoodknot", "vsb_rarefirst")
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
EXPECTED_SEEDS = (42, 43, 44)
RUN_RE = re.compile(r"^(?P<variant>.+)_seed(?P<seed>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--dataset", choices=("all", *DATASETS), default="all")
    parser.add_argument("--expected-batch", type=int, default=None)
    parser.add_argument("--expected-imgsz", type=int, default=1024)
    parser.add_argument("--expected-epochs", type=int, default=50)
    parser.add_argument("--expected-seeds", nargs="+", type=int, default=list(EXPECTED_SEEDS))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("results/corrected_common_eval_fixed/training_args_audit.csv"),
    )
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit non-zero if any expected run is missing or has mismatched settings.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = DATASETS if args.dataset == "all" else (args.dataset,)
    rows: list[dict[str, Any]] = []
    warnings: list[str] = []

    for dataset in datasets:
        runs_root = args.results_root / "multiseed" / dataset / "per_seed" / "runs"
        expected = {(variant, seed) for variant in EXPECTED_VARIANTS[dataset] for seed in args.expected_seeds}
        observed: set[tuple[str, int]] = set()
        if not runs_root.exists():
            warnings.append(f"missing_runs_root: {runs_root}")
            continue
        for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
            match = RUN_RE.match(run_dir.name)
            if not match:
                continue
            variant = match.group("variant")
            seed = int(match.group("seed"))
            if variant not in EXPECTED_VARIANTS[dataset] or seed not in args.expected_seeds:
                continue
            observed.add((variant, seed))
            row = audit_run(args, dataset=dataset, variant=variant, seed=seed, run_dir=run_dir)
            rows.append(row)
            warnings.extend(row["warnings"].split(";") if row["warnings"] else [])

        for variant, seed in sorted(expected - observed):
            run_dir = runs_root / f"{variant}_seed{seed}"
            warning = f"missing_expected_run: {dataset}/{variant}_seed{seed}"
            warnings.append(warning)
            rows.append(
                {
                    "dataset": dataset,
                    "variant": variant,
                    "seed": seed,
                    "run_dir": str(run_dir),
                    "args_yaml": str(run_dir / "ultralytics" / "train" / "args.yaml"),
                    "status": "missing_run",
                    "batch": "",
                    "imgsz": "",
                    "epochs": "",
                    "model": "",
                    "data": "",
                    "device": "",
                    "seed_arg": "",
                    "deterministic": "",
                    "workers": "",
                    "warnings": warning,
                }
            )

    write_csv(args.output_csv, rows)
    print(f"Wrote: {args.output_csv}")
    print_summary(rows)
    if warnings:
        print("\nWarnings:")
        for warning in sorted(set(warnings)):
            if warning:
                print(f"- {warning}")
    if warnings and args.fail_on_mismatch:
        raise SystemExit(1)


def audit_run(args: argparse.Namespace, *, dataset: str, variant: str, seed: int, run_dir: Path) -> dict[str, Any]:
    args_yaml = run_dir / "ultralytics" / "train" / "args.yaml"
    row: dict[str, Any] = {
        "dataset": dataset,
        "variant": variant,
        "seed": seed,
        "run_dir": str(run_dir),
        "args_yaml": str(args_yaml),
        "status": "ok",
        "batch": "",
        "imgsz": "",
        "epochs": "",
        "model": "",
        "data": "",
        "device": "",
        "seed_arg": "",
        "deterministic": "",
        "workers": "",
        "warnings": "",
    }
    if not args_yaml.exists():
        row["status"] = "missing_args_yaml"
        row["warnings"] = f"missing_args_yaml: {args_yaml}"
        return row

    data = yaml.safe_load(args_yaml.read_text(encoding="utf-8")) or {}
    row.update(
        {
            "batch": data.get("batch", ""),
            "imgsz": data.get("imgsz", ""),
            "epochs": data.get("epochs", ""),
            "model": data.get("model", ""),
            "data": data.get("data", ""),
            "device": data.get("device", ""),
            "seed_arg": data.get("seed", ""),
            "deterministic": data.get("deterministic", ""),
            "workers": data.get("workers", ""),
        }
    )

    warnings: list[str] = []
    if args.expected_batch is not None and as_int(data.get("batch")) != args.expected_batch:
        warnings.append(f"batch_mismatch: {dataset}/{variant}_seed{seed} batch={data.get('batch')}")
    if as_int(data.get("imgsz")) != args.expected_imgsz:
        warnings.append(f"imgsz_mismatch: {dataset}/{variant}_seed{seed} imgsz={data.get('imgsz')}")
    if as_int(data.get("epochs")) != args.expected_epochs:
        warnings.append(f"epochs_mismatch: {dataset}/{variant}_seed{seed} epochs={data.get('epochs')}")
    if as_int(data.get("seed")) != seed:
        warnings.append(f"seed_mismatch: {dataset}/{variant}_seed{seed} args_seed={data.get('seed')}")
    if data.get("deterministic") not in (True, "True", "true", 1, "1"):
        warnings.append(f"deterministic_not_true: {dataset}/{variant}_seed{seed} deterministic={data.get('deterministic')}")
    if warnings:
        row["status"] = "warn"
        row["warnings"] = ";".join(warnings)
    return row


def as_int(value: Any) -> int | None:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "variant",
        "seed",
        "status",
        "batch",
        "imgsz",
        "epochs",
        "model",
        "data",
        "device",
        "seed_arg",
        "deterministic",
        "workers",
        "args_yaml",
        "run_dir",
        "warnings",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (str(item["dataset"]), str(item["variant"]), int(item["seed"]))):
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def print_summary(rows: list[dict[str, Any]]) -> None:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["dataset"]), str(row["variant"])), []).append(row)
    print("\nTraining Args Summary")
    for (dataset, variant), items in sorted(grouped.items()):
        seeds = " ".join(str(item["seed"]) for item in sorted(items, key=lambda row: int(row["seed"])))
        batches = sorted({str(item["batch"]) for item in items if item["batch"] != ""})
        statuses = sorted({str(item["status"]) for item in items})
        print(f"- {dataset:13s} {variant:16s} seeds={seeds:8s} batch={','.join(batches) or 'n/a'} status={','.join(statuses)}")


if __name__ == "__main__":
    main()
