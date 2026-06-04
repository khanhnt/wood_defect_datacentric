#!/usr/bin/env python3
"""Generate before/after previews for preprocessing variants.

The script is read-only with respect to source datasets. It writes preview
panels and a sanity CSV under results/preprocessing_preview/.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Any

import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent

if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from wood_defect_datacentric.datasets.adapters import (  # noqa: E402
    DatasetRecord,
    load_manifest_dataset,
    resolve_repo_path,
)
from wood_defect_datacentric.preprocessing.variants import (  # noqa: E402
    PreprocessingVariant,
    apply_preprocessing,
    load_all_variant_configs,
    save_rgb_image,
)


CLASS_COLORS = {
    "live_knot": (36, 134, 68),
    "dead_knot": (194, 89, 45),
    "resin": (45, 111, 201),
    "knot_with_crack": (152, 78, 163),
    "crack": (32, 32, 32),
    "marrow": (232, 162, 38),
    "knot_missing": (111, 87, 63),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs" / "project.yaml")
    parser.add_argument(
        "--variant-config-dir",
        type=Path,
        default=PROJECT_ROOT / "configs" / "preprocessing",
    )
    parser.add_argument("--dataset", choices=("vnwoodknot", "vsb"), default="vnwoodknot")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "preprocessing_preview",
    )
    parser.add_argument("--panel-width", type=int, default=360)
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def load_dataset(config: dict[str, Any], dataset_key: str):
    paths = config["paths"]
    datasets = config["datasets"]
    if dataset_key == "vnwoodknot":
        return load_manifest_dataset(
            dataset_key="vnwoodknot",
            manifest_path=resolve_repo_path(paths["vnwoodknot_manifest_path"], PROJECT_ROOT),
            expected_classes=datasets["vnwoodknot"]["positive_classes"],
            negative_source_category=datasets["vnwoodknot"]["negative_class"],
            check_image_exists=True,
        )
    return load_manifest_dataset(
        dataset_key="vsb",
        manifest_path=resolve_repo_path(paths["vsb_manifest_path"], PROJECT_ROOT),
        expected_classes=datasets["vsb_curated"]["classes"],
        check_image_exists=True,
    )


def can_open_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def select_preview_records(records: list[DatasetRecord], *, split: str, num_samples: int) -> list[DatasetRecord]:
    split_records = [record for record in records if record.split == split]
    if not split_records:
        split_records = records

    candidates = [
        record
        for record in split_records
        if record.image_path.exists()
        and can_open_image(record.image_path)
        and not any(issue.startswith("bbox_") or issue == "non_positive_bbox" for issue in record.issues)
    ]
    if not candidates:
        raise SystemExit(f"No readable preview candidates found for split={split!r}.")

    selected: list[DatasetRecord] = []

    def add_first(predicate) -> None:
        for record in candidates:
            if record in selected:
                continue
            if predicate(record):
                selected.append(record)
                return

    add_first(lambda record: any(ann.class_name == "live_knot" for ann in record.annotations))
    add_first(lambda record: any(ann.class_name == "dead_knot" for ann in record.annotations))
    add_first(lambda record: record.is_negative)
    add_first(lambda record: record.is_positive and len(record.annotations) <= 2)

    for record in candidates:
        if len(selected) >= num_samples:
            break
        if record not in selected:
            selected.append(record)

    return selected[:num_samples]


def labels_signature(record: DatasetRecord) -> tuple[tuple[str, tuple[float, float, float, float]], ...]:
    return tuple((ann.class_name, tuple(round(value, 8) for value in ann.bbox_xyxy_norm)) for ann in record.annotations)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def resized_with_boxes(
    image: np.ndarray,
    record: DatasetRecord,
    *,
    width: int,
    title: str,
) -> Image.Image:
    pil = Image.fromarray(image)
    scale = width / pil.width
    height = int(round(pil.height * scale))
    resized = pil.resize((width, height), Image.Resampling.LANCZOS)
    draw = ImageDraw.Draw(resized)
    font = ImageFont.load_default()

    for ann in record.annotations:
        color = CLASS_COLORS.get(ann.class_name, (20, 20, 20))
        x1, y1, x2, y2 = ann.bbox_xyxy_norm
        box = (x1 * width, y1 * height, x2 * width, y2 * height)
        line_width = max(2, int(round(width / 180)))
        draw.rectangle(box, outline=color, width=line_width)
        label = short_label(ann.class_name)
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_x = int(max(0, min(box[0], width - text_width - 4)))
        label_y = int(max(0, box[1] - text_height - 4))
        draw.rectangle(
            (label_x, label_y, label_x + text_width + 4, label_y + text_height + 3),
            fill=(255, 255, 255),
        )
        draw.text((label_x + 2, label_y + 1), label, fill=color, font=font)

    header_h = 28
    canvas = Image.new("RGB", (width, height + header_h), "white")
    canvas.paste(resized, (0, header_h))
    draw = ImageDraw.Draw(canvas)
    draw.text((6, 8), title, fill=(30, 30, 30), font=font)
    return canvas


def short_label(class_name: str) -> str:
    return {
        "live_knot": "live",
        "dead_knot": "dead",
        "knot_with_crack": "knot+crack",
        "knot_missing": "missing",
    }.get(class_name, class_name)


def make_variant_panel(
    *,
    variant: PreprocessingVariant,
    records: list[DatasetRecord],
    output_dir: Path,
    panel_width: int,
) -> list[dict[str, Any]]:
    rows: list[Image.Image] = []
    manifest_rows: list[dict[str, Any]] = []
    processed_dir = output_dir / "processed_samples" / variant.name

    for index, record in enumerate(records, start=1):
        original = load_rgb(record.image_path)
        processed = apply_preprocessing(original, variant)
        labels_before = labels_signature(record)
        labels_after = labels_signature(record)
        dims_unchanged = original.shape == processed.shape
        labels_unchanged = labels_before == labels_after
        if not dims_unchanged:
            raise ValueError(f"{variant.name} changed dimensions for {record.image_id}")
        if not labels_unchanged:
            raise ValueError(f"{variant.name} changed labels for {record.image_id}")

        processed_path = processed_dir / f"sample{index:02d}_{safe_stem(record.image_id)}.png"
        save_rgb_image(processed, processed_path)
        with Image.open(processed_path) as check:
            check.verify()

        before = resized_with_boxes(original, record, width=panel_width, title="Before")
        after = resized_with_boxes(processed, record, width=panel_width, title=variant.name)
        strip = hstack([before, after], gap=10)
        rows.append(strip)
        manifest_rows.append(
            {
                "variant": variant.name,
                "sample_index": index,
                "dataset_key": record.dataset_key,
                "split": record.split,
                "image_id": record.image_id,
                "source_category": record.source_category or "",
                "source_image": str(record.image_path),
                "processed_sample": str(processed_path),
                "width": original.shape[1],
                "height": original.shape[0],
                "num_boxes": len(record.annotations),
                "dimensions_unchanged": dims_unchanged,
                "labels_unchanged": labels_unchanged,
                "processed_image_valid": True,
            }
        )

    panel = vstack(rows, gap=12, background="white")
    output_path = output_dir / f"{variant.name}_before_after.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)
    for row in manifest_rows:
        row["before_after_panel"] = str(output_path)
    return manifest_rows


def make_overview_panel(
    *,
    variants: list[PreprocessingVariant],
    records: list[DatasetRecord],
    output_dir: Path,
    panel_width: int,
) -> Path:
    overview_rows: list[Image.Image] = []
    overview_width = max(260, int(panel_width * 0.72))
    for record in records:
        original = load_rgb(record.image_path)
        cells = [resized_with_boxes(original, record, width=overview_width, title="Original")]
        for variant in variants:
            processed = apply_preprocessing(original, variant)
            cells.append(resized_with_boxes(processed, record, width=overview_width, title=variant.name.replace("_", " ")))
        overview_rows.append(hstack(cells, gap=8))
    panel = vstack(overview_rows, gap=12, background="white")
    output_path = output_dir / "preprocessing_variants_overview.png"
    panel.save(output_path)
    return output_path


def hstack(images: list[Image.Image], gap: int = 8, background: str = "white") -> Image.Image:
    width = sum(image.width for image in images) + gap * (len(images) - 1)
    height = max(image.height for image in images)
    canvas = Image.new("RGB", (width, height), background)
    x = 0
    for image in images:
        canvas.paste(image, (x, 0))
        x += image.width + gap
    return canvas


def vstack(images: list[Image.Image], gap: int = 8, background: str = "white") -> Image.Image:
    width = max(image.width for image in images)
    height = sum(image.height for image in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), background)
    y = 0
    for image in images:
        canvas.paste(image, (0, y))
        y += image.height + gap
    return canvas


def safe_stem(value: str) -> str:
    return "".join(char if char.isalnum() or char in ("-", "_") else "_" for char in value)[-80:]


def write_manifest(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "variant",
        "sample_index",
        "dataset_key",
        "split",
        "image_id",
        "source_category",
        "source_image",
        "processed_sample",
        "before_after_panel",
        "width",
        "height",
        "num_boxes",
        "dimensions_unchanged",
        "labels_unchanged",
        "processed_image_valid",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    dataset = load_dataset(config, args.dataset)
    variants = load_all_variant_configs(args.variant_config_dir)
    records = select_preview_records(list(dataset.records), split=args.split, num_samples=args.num_samples)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for variant in variants:
        all_rows.extend(
            make_variant_panel(
                variant=variant,
                records=records,
                output_dir=args.output_dir,
                panel_width=args.panel_width,
            )
        )
    overview = make_overview_panel(
        variants=variants,
        records=records,
        output_dir=args.output_dir,
        panel_width=args.panel_width,
    )
    manifest_path = args.output_dir / "preprocessing_preview_manifest.csv"
    write_manifest(all_rows, manifest_path)

    print(f"Dataset: {dataset.dataset_key}")
    print(f"Selected samples: {len(records)}")
    for record in records:
        print(f"- {record.split} | {record.image_id} | boxes={len(record.annotations)} | source={record.source_category}")
    print(f"Wrote overview: {overview}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote panels under: {args.output_dir}")


if __name__ == "__main__":
    main()
