#!/usr/bin/env python3
"""Audit held-out VNWoodKnot knot-free images for negative-aware evaluation."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_YAML = Path(
    os.environ.get(
        "WOOD_DC_VN_BASELINE_DATASET_YAML",
        "/workspace/data/vnwoodknot/benchmarks/vnwoodknot_live_dead_2class_yolo/dataset.yaml",
    )
)
DEFAULT_MANIFEST = PROJECT_ROOT / "data" / "processed" / "vnwoodknot_manifest.jsonl"
DEFAULT_PREDICTIONS_DIR = PROJECT_ROOT / "results" / "negative_aware" / "predictions"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS_DIR)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "results" / "negative_aware" / "knot_free_audit.txt")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sections = []

    data_yaml = args.data_yaml.expanduser()
    if data_yaml.exists():
        sections.append(audit_yolo_dataset(data_yaml.resolve()))
    else:
        sections.append(f"YOLO dataset audit: skipped; dataset YAML not found: {data_yaml}")

    predictions_dir = args.predictions_dir.expanduser()
    if predictions_dir.exists():
        prediction_section = audit_predictions(predictions_dir.resolve())
        if prediction_section:
            sections.append(prediction_section)

    manifest = args.manifest.expanduser()
    if manifest.exists():
        sections.append(audit_manifest(manifest.resolve()))
    else:
        sections.append(f"Manifest fallback audit: skipped; manifest not found: {manifest}")

    report = "\n\n".join(sections) + "\n"
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report, encoding="utf-8")
    print(report)
    print(f"Wrote: {output}")


def audit_yolo_dataset(data_yaml: Path) -> str:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    dataset_root = resolve_dataset_root(data_yaml, data)
    rows = load_materialized_samples(dataset_root)
    if rows:
        counts, split_ids, concerns = count_from_materialized_samples(rows)
        source = f"materialized_samples.csv under {dataset_root}"
    else:
        counts, split_ids, concerns = count_from_yolo_files(data_yaml, data, dataset_root)
        source = f"YOLO image/label files under {dataset_root}"

    lines = [
        "# VNWoodKnot Knot-Free Audit",
        "",
        "## YOLO Dataset Audit",
        "",
        f"- Dataset YAML: `{data_yaml}`",
        f"- Source: `{source}`",
        "",
        "| Split | Images | Positive/labelled | Knot-free/empty-label |",
        "|---|---:|---:|---:|",
    ]
    for split in SPLITS:
        item = counts.get(split, Counter())
        lines.append(
            f"| {split} | {item['images']} | {item['positive']} | {item['knot_free']} |"
        )
    lines.extend(
        [
            "",
            f"- Held-out test knot-free count: {counts.get('test', Counter()).get('knot_free', 0)}",
            f"- Test knot-free images also present in train/val: {len(concerns)}",
        ]
    )
    if concerns:
        lines.append(f"- Concern examples: {', '.join(sorted(concerns)[:10])}")
    else:
        lines.append("- Held-out check: pass; no test knot-free image IDs were found in train/val.")
    return "\n".join(lines)


def audit_predictions(predictions_dir: Path) -> str:
    files = sorted(predictions_dir.glob("*_predictions.json"))
    if not files:
        return ""
    path = files[0]
    payload = json.loads(path.read_text(encoding="utf-8"))
    images = payload.get("images") or []
    knot_free = [image for image in images if bool(image.get("is_knot_free", False))]
    positive = [image for image in images if not bool(image.get("is_knot_free", False))]
    lines = [
        "## Prediction JSON Audit",
        "",
        f"- Example prediction file: `{path}`",
        f"- Checkpoint: `{payload.get('checkpoint', '')}`",
        f"- Dataset YAML recorded by inference: `{payload.get('dataset_yaml', '')}`",
        f"- Test images in prediction JSON: {len(images)}",
        f"- Positive test images: {len(positive)}",
        f"- Knot-free test images: {len(knot_free)}",
        "- Note: prediction JSONs contain the evaluation split only; train/val held-out checks require the YOLO dataset or manifest.",
    ]
    return "\n".join(lines)


def audit_manifest(manifest: Path) -> str:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_ids: dict[str, set[str]] = defaultdict(set)
    knot_free_with_boxes = []
    with manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            split = normalize_split(raw.get("split"))
            annotations = raw.get("annotations") or []
            image_id = str(raw.get("image_id", ""))
            is_knot_free = raw.get("source_category") == "knot_free" or len(annotations) == 0
            counts[split]["images"] += 1
            if is_knot_free:
                counts[split]["knot_free"] += 1
                if annotations:
                    knot_free_with_boxes.append(image_id)
            else:
                counts[split]["positive"] += 1
            split_ids[split].add(canonical_id(image_id))

    test_train_overlap = split_ids["test"] & split_ids["train"]
    test_val_overlap = split_ids["test"] & split_ids["val"]
    lines = [
        "## Manifest Fallback Audit",
        "",
        f"- Manifest: `{manifest}`",
        "",
        "| Split | Records | Positive records | Knot-free/empty records |",
        "|---|---:|---:|---:|",
    ]
    for split in SPLITS:
        item = counts.get(split, Counter())
        lines.append(f"| {split} | {item['images']} | {item['positive']} | {item['knot_free']} |")
    lines.extend(
        [
            "",
            f"- Manifest test knot-free count: {counts.get('test', Counter()).get('knot_free', 0)}",
            f"- Knot-free records with boxes: {len(knot_free_with_boxes)}",
            f"- Canonical test/train overlap in raw manifest: {len(test_train_overlap)}",
            f"- Canonical test/val overlap in raw manifest: {len(test_val_overlap)}",
        ]
    )
    if test_train_overlap:
        lines.append(f"- Raw overlap examples: {', '.join(sorted(test_train_overlap)[:10])}")
        lines.append("- Note: raw-manifest overlap should be checked against materialized YOLO files; prior materialization skipped missing img_3671 records.")
    return "\n".join(lines)


def load_materialized_samples(dataset_root: Path) -> list[dict[str, str]]:
    path = dataset_root / "materialized_samples.csv"
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def count_from_materialized_samples(rows: list[dict[str, str]]) -> tuple[dict[str, Counter[str]], dict[str, set[str]], set[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_ids: dict[str, set[str]] = defaultdict(set)
    test_knot_free_ids: set[str] = set()
    train_val_ids: set[str] = set()
    for row in rows:
        split = normalize_split(row.get("split"))
        num_labels = int(float(row.get("num_labels") or 0))
        source_category = row.get("source_category") or ""
        image_key = canonical_id(row.get("source_image") or row.get("target_image") or row.get("image_id") or "")
        counts[split]["images"] += 1
        split_ids[split].add(image_key)
        is_knot_free = source_category == "knot_free" or num_labels == 0
        if is_knot_free:
            counts[split]["knot_free"] += 1
            if split == "test":
                test_knot_free_ids.add(image_key)
        else:
            counts[split]["positive"] += 1
        if split in {"train", "val"}:
            train_val_ids.add(image_key)
    return counts, split_ids, test_knot_free_ids & train_val_ids


def count_from_yolo_files(data_yaml: Path, data: dict[str, Any], dataset_root: Path) -> tuple[dict[str, Counter[str]], dict[str, set[str]], set[str]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    split_ids: dict[str, set[str]] = defaultdict(set)
    test_knot_free_ids: set[str] = set()
    train_val_ids: set[str] = set()
    for split in SPLITS:
        for split_dir in resolve_split_dirs(data_yaml, data, split):
            for image_path in sorted(path for path in split_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS):
                label_path = label_for_image(image_path, dataset_root=dataset_root, split=split, split_dir=split_dir)
                image_key = canonical_id(image_path)
                is_empty = not label_path.exists() or label_path.stat().st_size == 0
                counts[split]["images"] += 1
                split_ids[split].add(image_key)
                if is_empty:
                    counts[split]["knot_free"] += 1
                    if split == "test":
                        test_knot_free_ids.add(image_key)
                else:
                    counts[split]["positive"] += 1
                if split in {"train", "val"}:
                    train_val_ids.add(image_key)
    return counts, split_ids, test_knot_free_ids & train_val_ids


def resolve_dataset_root(data_yaml: Path, data: dict[str, Any]) -> Path:
    root = Path(str(data.get("path") or data_yaml.parent)).expanduser()
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.resolve()


def resolve_split_dirs(data_yaml: Path, data: dict[str, Any], split: str) -> list[Path]:
    root = resolve_dataset_root(data_yaml, data)
    value = data.get(split)
    values = value if isinstance(value, list) else [value]
    dirs = []
    for item in values:
        if not item:
            continue
        path = Path(str(item)).expanduser()
        dirs.append(path.resolve() if path.is_absolute() else (root / path).resolve())
    return dirs


def label_for_image(image_path: Path, *, dataset_root: Path, split: str, split_dir: Path) -> Path:
    try:
        rel = image_path.relative_to(split_dir)
        return (dataset_root / "labels" / split / rel).with_suffix(".txt")
    except ValueError:
        pass
    try:
        rel = image_path.relative_to(dataset_root)
        parts = list(rel.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            return (dataset_root / Path(*parts)).with_suffix(".txt")
    except ValueError:
        pass
    return image_path.with_suffix(".txt")


def normalize_split(value: Any) -> str:
    text = str(value or "").strip().lower()
    return {"validation": "val", "valid": "val"}.get(text, text)


def canonical_id(value: Any) -> str:
    return Path(str(value).replace("\\", "/")).stem.lower()


if __name__ == "__main__":
    main()
