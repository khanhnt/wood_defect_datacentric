#!/usr/bin/env python3
"""Materialize a YOLO dataset from a project manifest JSONL.

The script does not modify source images or manifests. It creates a new YOLO
folder with images/{train,val,test}, labels/{train,val,test}, dataset.yaml,
and a materialization report.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import shutil
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent

if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from wood_defect_datacentric.datasets.adapters import normalize_split, read_jsonl  # noqa: E402


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
VALID_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--dataset-name", required=True)
    parser.add_argument("--classes", nargs="+", required=True)
    parser.add_argument(
        "--split-strategy",
        choices=("manifest", "random"),
        default="manifest",
        help="Use manifest split labels, or create deterministic random splits when labels are missing.",
    )
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="How to place images in the YOLO folder.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output folder.")
    parser.add_argument(
        "--keep-records-with-unknown-classes",
        action="store_true",
        help="Keep records that contain labels outside --classes, dropping only those unknown labels.",
    )
    parser.add_argument(
        "--keep-invalid-box-records",
        action="store_true",
        help="Keep records after dropping invalid boxes. By default records with invalid boxes are skipped.",
    )
    parser.add_argument(
        "--max-missing-examples",
        type=int,
        default=20,
        help="Number of missing-image examples to store in the report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = args.manifest.expanduser().resolve()
    images_root = args.images_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    class_to_id = {class_name: index for index, class_name in enumerate(args.classes)}

    validate_args(args, manifest, images_root, output_root)
    prepare_output_root(output_root, overwrite=args.overwrite)

    image_index = build_image_index(images_root)
    raw_records = list(read_jsonl(manifest))
    records = assign_splits(raw_records, args)

    counters: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    class_box_counts: Counter[str] = Counter()
    missing_examples: list[dict[str, str]] = []
    written_rows: list[dict[str, Any]] = []

    for raw in records:
        counters["records_seen"] += 1
        split = normalize_split(raw.get("split"))
        if split not in VALID_SPLITS:
            counters["skipped_missing_split"] += 1
            continue

        labels, label_status = yolo_labels_from_annotations(
            raw.get("annotations") or [],
            class_to_id=class_to_id,
            keep_unknown=args.keep_records_with_unknown_classes,
            keep_invalid=args.keep_invalid_box_records,
        )
        counters.update(label_status)
        if label_status.get("skip_record", 0):
            continue

        source_image = resolve_image_path(raw, image_index)
        if source_image is None:
            counters["skipped_missing_image"] += 1
            if len(missing_examples) < args.max_missing_examples:
                missing_examples.append(
                    {
                        "image_id": str(raw.get("image_id", "")),
                        "image_path": str(raw.get("image_path", "")),
                    }
                )
            continue

        rel_image = output_relative_image_path(raw, source_image, split)
        target_image = output_root / "images" / split / rel_image
        target_label = output_root / "labels" / split / rel_image.with_suffix(".txt")
        place_image(source_image, target_image, mode=args.link_mode)
        write_label_file(target_label, labels)

        counters["records_written"] += 1
        split_counts[split] += 1
        for label in labels:
            class_box_counts[args.classes[int(label.split()[0])]] += 1
        written_rows.append(
            {
                "split": split,
                "image_id": str(raw.get("image_id", "")),
                "source_image": str(source_image),
                "target_image": str(target_image),
                "target_label": str(target_label),
                "num_labels": len(labels),
                "source_category": str(raw.get("source_category") or ""),
            }
        )

    dataset_yaml = write_dataset_yaml(output_root, args.dataset_name, args.classes)
    report = {
        "manifest": str(manifest),
        "images_root": str(images_root),
        "output_root": str(output_root),
        "dataset_yaml": str(dataset_yaml),
        "dataset_name": args.dataset_name,
        "classes": list(args.classes),
        "split_strategy": args.split_strategy,
        "link_mode": args.link_mode,
        "counters": dict(counters),
        "split_counts": dict(sorted(split_counts.items())),
        "class_box_counts": dict(sorted(class_box_counts.items())),
        "missing_image_examples": missing_examples,
    }
    report_path = output_root / "materialization_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    write_manifest_csv(output_root / "materialized_samples.csv", written_rows)

    print(f"Wrote YOLO dataset: {output_root}")
    print(f"Wrote dataset YAML: {dataset_yaml}")
    print(f"Wrote report: {report_path}")
    print(f"records_seen={counters['records_seen']} records_written={counters['records_written']}")
    print(f"split_counts={dict(sorted(split_counts.items()))}")
    if counters["skipped_missing_image"] or counters["skipped_missing_split"] or counters["skipped_unknown_class_record"] or counters["skipped_invalid_box_record"]:
        print("Warnings:")
        for key in (
            "skipped_missing_image",
            "skipped_missing_split",
            "skipped_unknown_class_record",
            "skipped_invalid_box_record",
        ):
            if counters[key]:
                print(f"- {key}={counters[key]}")


def validate_args(args: argparse.Namespace, manifest: Path, images_root: Path, output_root: Path) -> None:
    if not manifest.exists():
        raise SystemExit(f"Manifest does not exist: {manifest}")
    if not images_root.exists():
        raise SystemExit(f"Images root does not exist: {images_root}")
    total = args.train_ratio + args.val_ratio + args.test_ratio
    if args.split_strategy == "random" and abs(total - 1.0) > 1e-6:
        raise SystemExit(f"Random split ratios must sum to 1.0, got {total}")
    if output_root == images_root or images_root in output_root.parents:
        raise SystemExit("Refusing to place YOLO output inside images-root.")


def prepare_output_root(output_root: Path, *, overwrite: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise SystemExit(f"Output root exists and is not empty: {output_root}. Use --overwrite intentionally.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def build_image_index(images_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    stem_candidates: dict[str, list[Path]] = defaultdict(list)
    for path in images_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        rel = path.relative_to(images_root).as_posix()
        rel_no_suffix = path.relative_to(images_root).with_suffix("").as_posix()
        index.setdefault(rel, path)
        index.setdefault(rel_no_suffix, path)
        stem_candidates[path.stem].append(path)
    for stem, paths in stem_candidates.items():
        if len(paths) == 1:
            index.setdefault(stem, paths[0])
    return index


def assign_splits(raw_records: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.split_strategy == "manifest":
        return raw_records

    records = [dict(record) for record in raw_records]
    rng = random.Random(args.seed)
    rng.shuffle(records)
    train_end = int(round(len(records) * args.train_ratio))
    val_end = train_end + int(round(len(records) * args.val_ratio))
    for index, record in enumerate(records):
        if index < train_end:
            record["split"] = "train"
        elif index < val_end:
            record["split"] = "val"
        else:
            record["split"] = "test"
    return records


def yolo_labels_from_annotations(
    annotations: list[dict[str, Any]],
    *,
    class_to_id: dict[str, int],
    keep_unknown: bool,
    keep_invalid: bool,
) -> tuple[list[str], Counter[str]]:
    labels: list[str] = []
    status: Counter[str] = Counter()
    unknown_class_found = False
    invalid_box_found = False

    for annotation in annotations:
        class_name = str(annotation.get("class_name", ""))
        if class_name not in class_to_id:
            unknown_class_found = True
            status["unknown_class_boxes"] += 1
            continue
        bbox = annotation.get("bbox_xyxy_norm")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            invalid_box_found = True
            status["invalid_boxes"] += 1
            continue
        x1, y1, x2, y2 = [float(value) for value in bbox]
        if not (0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0):
            invalid_box_found = True
            status["invalid_boxes"] += 1
            continue
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        labels.append(f"{class_to_id[class_name]} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}")

    if unknown_class_found and not keep_unknown:
        status["skipped_unknown_class_record"] += 1
        status["skip_record"] += 1
    if invalid_box_found and not keep_invalid:
        status["skipped_invalid_box_record"] += 1
        status["skip_record"] += 1
    return labels, status


def resolve_image_path(raw: dict[str, Any], image_index: dict[str, Path]) -> Path | None:
    raw_path = Path(str(raw.get("image_path") or "")).expanduser()
    if raw_path.exists():
        return raw_path

    image_id = str(raw.get("image_id") or "").strip().replace("\\", "/")
    candidates = [image_id]
    if image_id:
        candidates.append(str(Path(image_id).with_suffix("")))
        candidates.append(Path(image_id).name)
        candidates.append(Path(image_id).stem)
    raw_name = raw_path.name
    raw_stem = raw_path.stem
    if raw_name:
        candidates.append(raw_name)
    if raw_stem:
        candidates.append(raw_stem)
    for key in candidates:
        if key in image_index:
            return image_index[key]
    return None


def output_relative_image_path(raw: dict[str, Any], source_image: Path, split: str) -> Path:
    image_id = str(raw.get("image_id") or source_image.stem).strip().replace("\\", "/")
    rel = Path(image_id)
    parts = rel.parts
    if len(parts) >= 2 and parts[0] == "images" and normalize_split(parts[1]) == split:
        rel = Path(*parts[2:]) if len(parts) > 2 else Path(source_image.stem)
    elif parts and normalize_split(parts[0]) == split:
        rel = Path(*parts[1:]) if len(parts) > 1 else Path(source_image.stem)
    if rel.suffix.lower() not in IMAGE_EXTENSIONS:
        rel = rel.with_suffix(source_image.suffix.lower())
    return rel


def place_image(source: Path, target: Path, *, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        target.unlink()
    if mode == "copy":
        shutil.copy2(source, target)
    elif mode == "hardlink":
        os.link(source, target)
    else:
        target.symlink_to(source)


def write_label_file(target: Path, labels: list[str]) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(labels) + ("\n" if labels else ""), encoding="utf-8")


def write_dataset_yaml(output_root: Path, dataset_name: str, classes: list[str]) -> Path:
    data = {
        "path": str(output_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: class_name for index, class_name in enumerate(classes)},
    }
    output_path = output_root / "dataset.yaml"
    output_path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return output_path


def write_manifest_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    fieldnames = ["split", "image_id", "source_image", "target_image", "target_label", "num_labels", "source_category"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
