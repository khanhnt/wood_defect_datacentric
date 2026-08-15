#!/usr/bin/env python3
"""Stage surviving checkpoints and build the single-generation checkpoint registry."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
from pathlib import Path
import shutil


SEEDS = (42, 43, 44)
ALL_VARIANTS = {
    "vnwoodknot": ("baseline", "p1_clahe", "p2_illumination", "p3_unsharp", "a1_crop", "a2_colorjitter", "p4_a4_combined"),
    "vsb_rarefirst": ("baseline", "p1_clahe", "p2_illumination", "p3_unsharp", "a1_crop", "a2_colorjitter", "p4_a4_combined"),
}
NEW_RUNS = {
    ("vnwoodknot", variant, seed)
    for variant in ("p1_clahe", "p3_unsharp", "a1_crop", "a2_colorjitter", "p4_a4_combined")
    for seed in SEEDS
} | {
    ("vsb_rarefirst", variant, seed)
    for variant in ("a1_crop", "a2_colorjitter", "p4_a4_combined")
    for seed in SEEDS
}
SURVIVORS = {
    (dataset, variant, seed)
    for dataset, variants in {
        "vnwoodknot": ("baseline", "p2_illumination"),
        "vsb_rarefirst": ("baseline", "p1_clahe", "p2_illumination", "p3_unsharp"),
    }.items()
    for variant in variants
    for seed in SEEDS
}
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
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--survivor-root", type=Path, help="Archived results root containing multiseed/.")
    parser.add_argument("--stage-survivors", action="store_true")
    parser.add_argument("--mode", choices=("copy", "hardlink"), default="copy")
    parser.add_argument("--allow-missing", action="store_true")
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def run_dir(root: Path, dataset: str, variant: str, seed: int) -> Path:
    return root / "multiseed" / dataset / "per_seed" / "runs" / f"{variant}_seed{seed}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage_file(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(source) != sha256(target):
            raise SystemExit(f"Refusing to replace different staged file: {target}")
        return
    if mode == "hardlink":
        os.link(source, target)
    else:
        shutil.copy2(source, target)


def stage_survivors(source_root: Path, generation_root: Path, mode: str) -> None:
    for dataset, variant, seed in sorted(SURVIVORS):
        source_run = run_dir(source_root, dataset, variant, seed)
        target_run = run_dir(generation_root, dataset, variant, seed)
        best = source_run / "ultralytics/train/weights/best.pt"
        if not best.exists() or best.stat().st_size < 10_000_000:
            raise SystemExit(f"Missing or suspicious survivor checkpoint: {best}")
        for relative in COPY_FILES:
            source = source_run / relative
            if source.exists():
                stage_file(source, target_run / relative, mode)
    print(f"Staged {len(SURVIVORS)} surviving runs into {generation_root}")


def main() -> None:
    args = parse_args()
    generation_root = args.generation_root.expanduser().resolve()
    if args.stage_survivors:
        if not args.survivor_root:
            raise SystemExit("--stage-survivors requires --survivor-root")
        stage_survivors(args.survivor_root.expanduser().resolve(), generation_root, args.mode)

    rows = []
    missing = []
    for dataset, variants in ALL_VARIANTS.items():
        for variant in variants:
            for seed in SEEDS:
                key = (dataset, variant, seed)
                source_type = "corrected_retrain" if key in NEW_RUNS else "archived_survivor"
                directory = run_dir(generation_root, dataset, variant, seed)
                best = directory / "ultralytics/train/weights/best.pt"
                last = directory / "ultralytics/train/weights/last.pt"
                best_ok = best.exists() and best.stat().st_size >= 10_000_000
                last_required = key in NEW_RUNS
                last_ok = last.exists() and last.stat().st_size >= 10_000_000
                status = "PASS" if best_ok and (last_ok or not last_required) else "FAIL"
                if status == "FAIL":
                    missing.append(f"{dataset}:{variant}:seed{seed}")
                rows.append(
                    {
                        "dataset": dataset,
                        "variant": variant,
                        "seed": seed,
                        "source_type": source_type,
                        "best_path": str(best),
                        "best_size": best.stat().st_size if best.exists() else 0,
                        "best_sha256": sha256(best) if best_ok else "",
                        "last_required": str(last_required).lower(),
                        "last_path": str(last),
                        "last_size": last.stat().st_size if last.exists() else 0,
                        "last_sha256": sha256(last) if last_ok else "",
                        "status": status,
                    }
                )

    output = args.output_csv or (generation_root / "provenance" / "checkpoint_registry.csv")
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote: {output}")
    print(f"CHECKPOINT REGISTRY: {'PASS' if not missing else 'FAIL'} ({len(rows) - len(missing)}/{len(rows)})")
    if missing and not args.allow_missing:
        raise SystemExit("Missing generation checkpoints: " + ", ".join(missing))


if __name__ == "__main__":
    main()
