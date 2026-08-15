#!/usr/bin/env python3
"""Stage and hash the 18 augmented-validation checkpoints retained for audit only."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
import shutil

import yaml


SEEDS = (42, 43, 44)
VARIANTS = ("a1_crop", "a2_colorjitter", "p4_a4_combined")
DATASETS = ("vnwoodknot", "vsb_rarefirst")
COPY_FILES = (
    "ultralytics/train/weights/best.pt",
    "ultralytics/train/args.yaml",
    "ultralytics/train/results.csv",
    "config_used.yaml",
    "validation_metrics.json",
    "run_summary.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("copy", "hardlink"), default="copy")
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_dir(root: Path, dataset: str, variant: str, seed: int) -> Path:
    return root / "multiseed" / dataset / "per_seed" / "runs" / f"{variant}_seed{seed}"


def place(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(source) != sha256(target):
            raise SystemExit(f"Refusing to replace different deprecated artifact: {target}")
        return
    if mode == "hardlink":
        os.link(source, target)
    else:
        shutil.copy2(source, target)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.expanduser().resolve()
    audit_root = args.generation_root.expanduser().resolve() / "deprecated_checkpoints"
    rows: list[dict[str, object]] = []
    for dataset in DATASETS:
        for variant in VARIANTS:
            for seed in SEEDS:
                source_run = run_dir(source_root, dataset, variant, seed)
                target_run = run_dir(audit_root, dataset, variant, seed)
                best = source_run / "ultralytics" / "train" / "weights" / "best.pt"
                if not best.exists() or best.stat().st_size < 10_000_000:
                    raise SystemExit(f"Missing or suspicious deprecated checkpoint: {best}")
                args_path = source_run / "ultralytics" / "train" / "args.yaml"
                if not args_path.exists():
                    raise SystemExit(f"Missing training args for deprecated checkpoint: {args_path}")
                training_args = yaml.safe_load(args_path.read_text(encoding="utf-8")) or {}
                expected_args = {"seed": seed, "batch": 40, "epochs": 50, "imgsz": 1024}
                mismatches = {
                    key: {"expected": expected, "actual": training_args.get(key)}
                    for key, expected in expected_args.items()
                    if training_args.get(key) != expected
                }
                if mismatches:
                    raise SystemExit(f"Deprecated checkpoint training args differ for {source_run}: {mismatches}")
                for relative in COPY_FILES:
                    source = source_run / relative
                    if source.exists():
                        place(source, target_run / relative, args.mode)
                target_best = target_run / "ultralytics" / "train" / "weights" / "best.pt"
                rows.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "seed": seed,
                        "checkpoint_role": "DEPRECATED_augmented_validation_selection",
                        "source_snapshot_root": str(source_root),
                        "source_run_path": str(source_run),
                        "source_best_path": str(best),
                        "source_best_sha256": sha256(best),
                        "source_args_path": str(args_path),
                        "source_args_sha256": sha256(args_path),
                        "training_batch": training_args["batch"],
                        "training_epochs": training_args["epochs"],
                        "training_imgsz": training_args["imgsz"],
                        "best_path": str(target_best),
                        "best_size": target_best.stat().st_size,
                        "best_sha256": sha256(target_best),
                        "status": "PASS",
                    }
                )
    output = args.output_csv or (audit_root / "deprecated_checkpoint_registry.csv")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote: {output}")
    print(f"DEPRECATED CHECKPOINT REGISTRY: PASS ({len(rows)}/18)")


if __name__ == "__main__":
    main()
