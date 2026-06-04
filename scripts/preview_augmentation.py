#!/usr/bin/env python3
"""Generate before/after previews for augmentation variants."""

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

from wood_defect_datacentric.augmentation.variants import (  # noqa: E402
    AugmentationContext,
    AugmentationResult,
    BoxLabel,
    apply_augmentation,
    load_all_variant_configs,
    record_to_labels,
)
from wood_defect_datacentric.datasets.adapters import (  # noqa: E402
    DatasetRecord,
    load_manifest_dataset,
    resolve_repo_path,
)
from wood_defect_datacentric.preprocessing.variants import save_rgb_image  # noqa: E402


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
        default=PROJECT_ROOT / "configs" / "augmentation",
    )
    parser.add_argument("--dataset", choices=("vnwoodknot", "vsb"), default="vnwoodknot")
    parser.add_argument("--split", default="test")
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "augmentation_preview",
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


def readable_records(records: list[DatasetRecord]) -> list[DatasetRecord]:
    return [
        record
        for record in records
        if record.image_path.exists()
        and can_open_image(record.image_path)
        and not any(issue.startswith("bbox_") or issue == "non_positive_bbox" for issue in record.issues)
    ]


def select_preview_records(records: list[DatasetRecord], *, split: str, num_samples: int) -> list[DatasetRecord]:
    split_records = [record for record in records if record.split == split]
    candidates = readable_records(split_records or records)
    if not candidates:
        raise SystemExit(f"No readable augmentation preview candidates found for split={split!r}.")

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


def build_context(records: list[DatasetRecord]) -> AugmentationContext:
    readable = readable_records(records)
    positives = tuple(record for record in readable if record.annotations)
    negatives = tuple(record for record in readable if record.is_negative)
    return AugmentationContext(object_records=positives, background_records=negatives)


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def labels_signature(labels: tuple[BoxLabel, ...]) -> tuple[tuple[str, tuple[float, float, float, float]], ...]:
    return tuple((label.class_name, tuple(round(value, 8) for value in label.bbox_xyxy_norm)) for label in labels)


def render_with_boxes(
    image: np.ndarray,
    labels: tuple[BoxLabel, ...],
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

    for label in labels:
        color = CLASS_COLORS.get(label.class_name, (20, 20, 20))
        if label.source == "copy_paste":
            color = tuple(min(255, int(channel * 1.25)) for channel in color)
        x1, y1, x2, y2 = label.bbox_xyxy_norm
        box = (x1 * width, y1 * height, x2 * width, y2 * height)
        line_width = max(2, int(round(width / 180)))
        draw.rectangle(box, outline=color, width=line_width)
        tag = short_label(label.class_name)
        if label.source == "copy_paste":
            tag = f"{tag}*"
        text_bbox = draw.textbbox((0, 0), tag, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        label_x = int(max(0, min(box[0], width - text_width - 4)))
        label_y = int(max(0, box[1] - text_height - 4))
        draw.rectangle(
            (label_x, label_y, label_x + text_width + 4, label_y + text_height + 3),
            fill=(255, 255, 255),
        )
        draw.text((label_x + 2, label_y + 1), tag, fill=color, font=font)

    header_h = 28
    canvas = Image.new("RGB", (width, height + header_h), "white")
    canvas.paste(resized, (0, header_h))
    ImageDraw.Draw(canvas).text((6, 8), title, fill=(30, 30, 30), font=font)
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
    variant,
    records: list[DatasetRecord],
    context: AugmentationContext,
    seed: int,
    output_dir: Path,
    panel_width: int,
) -> list[dict[str, Any]]:
    rows: list[Image.Image] = []
    manifest_rows: list[dict[str, Any]] = []
    sample_dir = output_dir / "augmented_samples" / variant.name

    for index, record in enumerate(records, start=1):
        original = load_rgb(record.image_path)
        original_labels = record_to_labels(record)
        rng = np.random.default_rng(seed + index * 1009 + stable_variant_offset(variant.name))
        result = apply_augmentation(original, original_labels, variant, rng=rng, context=context)
        sanity_check(record, original, original_labels, result, variant.name)

        sample_path = sample_dir / f"sample{index:02d}_{safe_stem(record.image_id)}.png"
        save_rgb_image(result.image, sample_path)
        with Image.open(sample_path) as check:
            check.verify()

        before = render_with_boxes(original, original_labels, width=panel_width, title="Before")
        after = render_with_boxes(result.image, result.labels, width=panel_width, title=variant.name)
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
                "augmented_sample": str(sample_path),
                "before_after_panel": "",
                "width": original.shape[1],
                "height": original.shape[0],
                "original_num_boxes": len(original_labels),
                "augmented_num_boxes": len(result.labels),
                "num_added_boxes": result.num_added_boxes,
                "dimensions_unchanged": original.shape == result.image.shape,
                "boxes_inside_bounds": result.boxes_inside_bounds,
                "min_visibility": round(result.min_visibility, 6),
                "labels_unchanged": labels_signature(original_labels) == labels_signature(result.labels),
                "notes": ";".join(result.notes),
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
    variants,
    records: list[DatasetRecord],
    context: AugmentationContext,
    seed: int,
    output_dir: Path,
    panel_width: int,
) -> Path:
    overview_rows: list[Image.Image] = []
    overview_width = max(260, int(panel_width * 0.72))
    for index, record in enumerate(records, start=1):
        original = load_rgb(record.image_path)
        original_labels = record_to_labels(record)
        cells = [render_with_boxes(original, original_labels, width=overview_width, title="Original")]
        for variant in variants:
            rng = np.random.default_rng(seed + index * 1009 + stable_variant_offset(variant.name))
            result = apply_augmentation(original, original_labels, variant, rng=rng, context=context)
            cells.append(render_with_boxes(result.image, result.labels, width=overview_width, title=variant.name.replace("_", " ")))
        overview_rows.append(hstack(cells, gap=8))
    panel = vstack(overview_rows, gap=12, background="white")
    output_path = output_dir / "augmentation_variants_overview.png"
    panel.save(output_path)
    return output_path


def sanity_check(
    record: DatasetRecord,
    original: np.ndarray,
    original_labels: tuple[BoxLabel, ...],
    result: AugmentationResult,
    variant_name: str,
) -> None:
    if original.shape != result.image.shape:
        raise ValueError(f"{variant_name} changed dimensions for {record.image_id}")
    if not result.boxes_inside_bounds:
        raise ValueError(f"{variant_name} produced box outside bounds for {record.image_id}")
    if original_labels and not result.labels and variant_name != "A3_copy_paste_defects":
        raise ValueError(f"{variant_name} dropped all boxes for positive image {record.image_id}")
    if variant_name in {"A1_defect_preserving_crop", "A4_combined_best"} and result.min_visibility < 0.90:
        raise ValueError(f"{variant_name} violated visibility guard for {record.image_id}")
    if variant_name == "A3_copy_paste_defects" and result.num_added_boxes <= 0:
        raise ValueError(f"{variant_name} did not add a pasted defect for preview image {record.image_id}")


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


def stable_variant_offset(name: str) -> int:
    return sum((idx + 1) * ord(char) for idx, char in enumerate(name))


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
        "augmented_sample",
        "before_after_panel",
        "width",
        "height",
        "original_num_boxes",
        "augmented_num_boxes",
        "num_added_boxes",
        "dimensions_unchanged",
        "boxes_inside_bounds",
        "min_visibility",
        "labels_unchanged",
        "notes",
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
    all_records = list(dataset.records)
    records = select_preview_records(all_records, split=args.split, num_samples=args.num_samples)
    context = build_context(all_records)
    if not context.object_records:
        print("Warning: copy-paste object bank is empty; A3 will be skipped.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for variant in variants:
        all_rows.extend(
            make_variant_panel(
                variant=variant,
                records=records,
                context=context,
                seed=args.seed,
                output_dir=args.output_dir,
                panel_width=args.panel_width,
            )
        )
    overview = make_overview_panel(
        variants=variants,
        records=records,
        context=context,
        seed=args.seed,
        output_dir=args.output_dir,
        panel_width=args.panel_width,
    )
    manifest_path = args.output_dir / "augmentation_preview_manifest.csv"
    write_manifest(all_rows, manifest_path)

    print(f"Dataset: {dataset.dataset_key}")
    print(f"Selected samples: {len(records)}")
    print(f"Object bank: {len(context.object_records)} readable positive records")
    print(f"Background bank: {len(context.background_records)} readable negative records")
    for record in records:
        print(f"- {record.split} | {record.image_id} | boxes={len(record.annotations)} | source={record.source_category}")
    print(f"Wrote overview: {overview}")
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote panels under: {args.output_dir}")


if __name__ == "__main__":
    main()
