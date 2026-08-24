#!/usr/bin/env python3
"""Verify all dataset and job invariants before Faster R-CNN training."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from second_detector.data import list_images, split_paths
from second_detector.protocol import build_jobs, locked_config


EXPECTED = {
    "train": {"images": 1059, "labels": 1059},
    "val": {"images": 226, "labels": 226, "empty": 75, "boxes": 151},
    "test": {"images": 229, "labels": 229, "empty": 75, "boxes": 155},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_stats(data_yaml: Path, split: str) -> dict:
    image_root, label_root, _ = split_paths(data_yaml, split)
    images = list_images(image_root)
    labels = []
    empty = 0
    boxes = 0
    missing = []
    for image in images:
        label = label_root / image.relative_to(image_root).with_suffix(".txt")
        if not label.exists():
            missing.append(str(label))
            continue
        labels.append(label)
        lines = [line for line in label.read_text(encoding="utf-8").splitlines() if line.strip()]
        boxes += len(lines)
        empty += int(not lines)
    return {
        "images": len(images),
        "labels": len(labels),
        "empty": empty,
        "boxes": boxes,
        "missing_labels": len(missing),
        "image_root": str(image_root),
        "label_root": str(label_root),
        "images_by_relative": {str(path.relative_to(image_root)): path for path in images},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuilt-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    jobs = build_jobs(rebuilt_root=args.rebuilt_root, results_root=args.results_root)
    canonical_yaml = args.rebuilt_root / "canonical" / "vnwoodknot" / "dataset.yaml"
    canonical = {split: split_stats(canonical_yaml, split) for split in ("train", "val", "test")}
    rows = []
    failures = []
    for job in jobs:
        for split in ("train", "val", "test"):
            actual = split_stats(job.data_yaml, split)
            expected = EXPECTED[split]
            count_ok = all(actual[key] == value for key, value in expected.items()) and actual["missing_labels"] == 0
            common_eval_ok = True
            differing_images = 0
            if split in {"val", "test"}:
                reference = canonical[split]["images_by_relative"]
                candidate = actual["images_by_relative"]
                if set(reference) != set(candidate):
                    common_eval_ok = False
                    differing_images = len(set(reference) ^ set(candidate))
                else:
                    differing_images = sum(sha256(reference[key]) != sha256(candidate[key]) for key in reference)
                    common_eval_ok = differing_images == 0
            status = "PASS" if count_ok and common_eval_ok else "FAIL"
            row = {
                "job_id": job.job_id,
                "variant": job.variant,
                "seed": job.seed,
                "split": split,
                "data_yaml": str(job.data_yaml),
                "images": actual["images"],
                "labels": actual["labels"],
                "empty_labels": actual["empty"],
                "boxes": actual["boxes"],
                "missing_labels": actual["missing_labels"],
                "common_eval_differing_images": differing_images,
                "status": status,
            }
            rows.append(row)
            if status != "PASS":
                failures.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "verification_gate.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    manifest = {
        "status": "PASS" if not failures else "FAIL",
        "protocol": locked_config(epochs=args.epochs, batch_size=args.batch_size, workers=args.workers),
        "jobs": [job.to_dict() for job in jobs],
        "verification_rows": len(rows),
        "failures": failures,
    }
    json_path = args.output_dir / "job_manifest.json"
    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {json_path}")
    print(f"FASTER R-CNN VERIFICATION: {manifest['status']} ({len(rows) - len(failures)}/{len(rows)})")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
