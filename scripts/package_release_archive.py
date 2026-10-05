#!/usr/bin/env python3
"""Build a versioned, checksummed archive layout for external experiment artifacts."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import shutil
from collections import Counter
from pathlib import Path


YOLO_GROUPS = {
    "primary_checkpoint": ("multiseed/**/weights/best.pt", "multiseed/**/weights/last.pt"),
    "deprecated_checkpoint": (
        "deprecated_checkpoints/**/weights/best.pt",
        "deprecated_checkpoints/**/weights/last.pt",
    ),
    "primary_prediction": ("predictions/**/*_predictions.json",),
    "deprecated_prediction": ("deprecated_audit/predictions/**/*_predictions.json",),
    "training_record": (
        "multiseed/**/results.csv",
        "multiseed/**/args.yaml",
        "multiseed/**/config_used.yaml",
        "multiseed/**/run_summary.json",
        "multiseed/**/validation_metrics.json",
        "gpu_optimization/run_log.csv",
        "gpu_optimization/generated_configs/*.yaml",
    ),
    "provenance": ("provenance/**/*", "eval_maps/**/*"),
}

FRCNN_GROUPS = {
    "checkpoint": ("fasterrcnn/**/weights/best.pt", "fasterrcnn/**/weights/last.pt"),
    "prediction": ("predictions/**/*_predictions.json",),
    "training_record": (
        "fasterrcnn/**/results.csv",
        "fasterrcnn/**/training_summary.json",
        "fasterrcnn/**/config_used.yaml",
        "fasterrcnn/**/environment.json",
        "fasterrcnn/run_log.csv",
    ),
    "provenance": ("provenance/**/*",),
    "analysis": ("fasterrcnn/analysis/**/*", "fasterrcnn/negative_aware/**/*"),
}

STRICT_EXPECTED = {
    ("yolov8s", "primary_checkpoint", "best.pt"): 42,
    ("yolov8s", "deprecated_checkpoint", "best.pt"): 18,
    ("yolov8s", "primary_prediction", "prediction"): 126,
    ("yolov8s", "deprecated_prediction", "prediction"): 54,
    ("fasterrcnn", "checkpoint", "best.pt"): 9,
    ("fasterrcnn", "checkpoint", "last.pt"): 9,
    ("fasterrcnn", "prediction", "prediction"): 18,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Archive version label, e.g. discover-ai-v1.")
    parser.add_argument("--yolo-generation", type=Path, help="Frozen YOLOv8s generation root.")
    parser.add_argument("--fasterrcnn-generation", type=Path, help="Frozen Faster R-CNN generation root.")
    parser.add_argument("--output-root", type=Path, default=Path("release_archives"))
    parser.add_argument("--mode", choices=("copy", "hardlink"), default="copy")
    parser.add_argument("--strict", action="store_true", help="Require the expected checkpoint/export counts.")
    parser.add_argument("--dry-run", action="store_true", help="Inventory inputs without writing the archive.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", args.version):
        raise SystemExit("--version may contain only letters, digits, dot, underscore, and hyphen.")
    if args.yolo_generation is None and args.fasterrcnn_generation is None:
        raise SystemExit("Provide at least one generation root.")

    archive = args.output_root.expanduser().resolve() / f"wood-defect-datacentric-{args.version}"
    if archive.exists() and not args.dry_run:
        raise SystemExit(f"Archive already exists; choose a new version or output root: {archive}")

    records: list[dict[str, object]] = []
    inventories = (
        ("yolov8s", args.yolo_generation, YOLO_GROUPS),
        ("fasterrcnn", args.fasterrcnn_generation, FRCNN_GROUPS),
    )
    for detector, root_arg, groups in inventories:
        if root_arg is None:
            continue
        root = root_arg.expanduser().resolve()
        if not root.is_dir():
            raise SystemExit(f"Missing {detector} generation root: {root}")
        collect_generation(detector, root, groups, archive, args.mode, args.dry_run, records)

    validate_inventory(records, strict=args.strict)
    if args.dry_run:
        print_summary(archive, records, dry_run=True)
        return

    archive.mkdir(parents=True, exist_ok=True)
    write_manifests(archive, records)
    print_summary(archive, records, dry_run=False)


def collect_generation(
    detector: str,
    root: Path,
    groups: dict[str, tuple[str, ...]],
    archive: Path,
    mode: str,
    dry_run: bool,
    records: list[dict[str, object]],
) -> None:
    claimed: set[Path] = set()
    for role, patterns in groups.items():
        for pattern in patterns:
            for source in sorted(root.glob(pattern)):
                if not source.is_file() or source in claimed:
                    continue
                claimed.add(source)
                relative = source.relative_to(root)
                destination = archive / detector / relative
                digest = sha256(source)
                if not dry_run:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    if mode == "hardlink":
                        os.link(source, destination)
                    else:
                        shutil.copy2(source, destination)
                records.append(
                    {
                        "detector": detector,
                        "role": role,
                        "source_relative_path": relative.as_posix(),
                        "archive_path": destination.relative_to(archive).as_posix(),
                        "size_bytes": source.stat().st_size,
                        "sha256": digest,
                    }
                )


def validate_inventory(records: list[dict[str, object]], strict: bool) -> None:
    counts: Counter[tuple[str, str, str]] = Counter()
    for record in records:
        archive_path = str(record["archive_path"])
        name = Path(archive_path).name
        kind = "prediction" if name.endswith("_predictions.json") else name
        counts[(str(record["detector"]), str(record["role"]), kind)] += 1

    failures = []
    for key, expected in STRICT_EXPECTED.items():
        detector = key[0]
        detector_included = any(str(record["detector"]) == detector for record in records)
        if not detector_included:
            continue
        observed = counts[key]
        if observed != expected:
            failures.append(f"{key}: {observed}/{expected}")

    if failures:
        message = "Incomplete archive inventory: " + "; ".join(failures)
        if strict:
            raise SystemExit(message)
        print(f"WARNING: {message}")


def write_manifests(archive: Path, records: list[dict[str, object]]) -> None:
    fields = (
        "detector",
        "role",
        "source_relative_path",
        "archive_path",
        "size_bytes",
        "sha256",
    )
    with (archive / "MANIFEST.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    lines = [f"{record['sha256']}  {record['archive_path']}" for record in records]
    (archive / "SHA256SUMS").write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def print_summary(archive: Path, records: list[dict[str, object]], dry_run: bool) -> None:
    by_role = Counter((str(row["detector"]), str(row["role"])) for row in records)
    total_bytes = sum(int(row["size_bytes"]) for row in records)
    print("RELEASE ARCHIVE INVENTORY")
    print(f"- destination: {archive}")
    print(f"- mode: {'dry-run' if dry_run else 'written'}")
    for (detector, role), count in sorted(by_role.items()):
        print(f"- {detector} {role}: {count}")
    print(f"- files: {len(records)}")
    print(f"- logical size: {total_bytes / (1024 ** 3):.2f} GiB")
    if not dry_run:
        print(f"- checksums: {archive / 'SHA256SUMS'}")


if __name__ == "__main__":
    main()
