#!/usr/bin/env python3
"""Materialize a preprocessed YOLO dataset from an existing YOLO dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent

if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from wood_defect_datacentric.preprocessing.variants import (  # noqa: E402
    apply_preprocessing,
    load_variant_config,
    save_rgb_image,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-yaml", type=Path, required=True)
    parser.add_argument("--variant-config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--image-format", choices=("png", "jpg"), default="png")
    parser.add_argument("--jpg-quality", type=int, default=95)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_yaml = args.source_yaml.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    variant = load_variant_config(args.variant_config.expanduser().resolve())

    if not source_yaml.exists():
        raise SystemExit(f"Missing source dataset YAML: {source_yaml}")
    prepare_output(output_root, overwrite=args.overwrite)

    source = yaml.safe_load(source_yaml.read_text(encoding="utf-8")) or {}
    source_root = resolve_source_root(source_yaml, source)
    names = source.get("names") or []
    if isinstance(names, dict):
        names = {int(key): value for key, value in names.items()}

    report: dict[str, Any] = {
        "source_yaml": str(source_yaml),
        "source_root": str(source_root),
        "output_root": str(output_root),
        "variant": variant.name,
        "splits": {},
        "warnings": [],
    }

    for split in SPLITS:
        source_images_dir = resolve_split_dir(source_yaml, source, split)
        if source_images_dir is None or not source_images_dir.exists():
            report["warnings"].append(f"Missing split {split}: {source_images_dir}")
            report["splits"][split] = {"images": 0, "labels": 0, "missing_labels": 0}
            continue
        split_counts = materialize_split(
            source_images_dir=source_images_dir,
            source_root=source_root,
            output_root=output_root,
            split=split,
            variant=variant,
            image_format=args.image_format,
            jpg_quality=args.jpg_quality,
        )
        report["splits"][split] = split_counts

    dataset_yaml = output_root / "dataset.yaml"
    dataset_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(output_root),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": names,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    report["dataset_yaml"] = str(dataset_yaml)
    report_path = output_root / "materialization_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {dataset_yaml}")
    print(f"Wrote: {report_path}")
    for split, counts in report["splits"].items():
        print(f"{split}: images={counts['images']} labels={counts['labels']} missing_labels={counts['missing_labels']}")


def prepare_output(output_root: Path, *, overwrite: bool) -> None:
    if output_root.exists() and any(output_root.iterdir()):
        if not overwrite:
            raise SystemExit(f"Output root exists and is not empty: {output_root}. Use --overwrite intentionally.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)


def resolve_source_root(source_yaml: Path, source: dict[str, Any]) -> Path:
    root_value = source.get("path") or source_yaml.parent
    root = Path(str(root_value)).expanduser()
    if not root.is_absolute():
        root = source_yaml.parent / root
    return root.resolve()


def resolve_split_dir(source_yaml: Path, source: dict[str, Any], split: str) -> Path | None:
    value = source.get(split)
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0] if value else ""
    split_path = Path(str(value)).expanduser()
    if split_path.is_absolute():
        return split_path
    return (resolve_source_root(source_yaml, source) / split_path).resolve()


def materialize_split(
    *,
    source_images_dir: Path,
    source_root: Path,
    output_root: Path,
    split: str,
    variant,
    image_format: str,
    jpg_quality: int,
) -> dict[str, int]:
    images = sorted(path for path in source_images_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    counts = {"images": 0, "labels": 0, "missing_labels": 0}
    for source_image in images:
        rel_image = source_image.relative_to(source_images_dir)
        output_image = (output_root / "images" / split / rel_image).with_suffix(f".{image_format}")
        source_label = label_for_image(source_image, source_root)
        output_label = (output_root / "labels" / split / rel_image).with_suffix(".txt")

        with Image.open(source_image) as image:
            processed = apply_preprocessing(image.convert("RGB"), variant)
        if image_format == "jpg":
            output_image.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(processed).save(output_image, quality=int(jpg_quality))
        else:
            save_rgb_image(processed, output_image)
        counts["images"] += 1

        output_label.parent.mkdir(parents=True, exist_ok=True)
        if source_label.exists():
            shutil.copy2(source_label, output_label)
            counts["labels"] += 1
        else:
            output_label.write_text("", encoding="utf-8")
            counts["missing_labels"] += 1
    return counts


def label_for_image(image_path: Path, source_root: Path) -> Path:
    rel = image_path.relative_to(source_root)
    parts = list(rel.parts)
    if "images" in parts:
        index = parts.index("images")
        parts[index] = "labels"
        return (source_root / Path(*parts)).with_suffix(".txt")
    return image_path.with_suffix(".txt")


if __name__ == "__main__":
    main()
