"""Helpers for materializing augmented YOLO-compatible samples."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image
import numpy as np

from wood_defect_datacentric.augmentation.variants import (
    AugmentationContext,
    AugmentationVariant,
    apply_augmentation,
    labels_to_yolo_lines,
    record_to_labels,
)
from wood_defect_datacentric.datasets.adapters import DatasetRecord
from wood_defect_datacentric.preprocessing.variants import save_rgb_image


@dataclass(frozen=True)
class MaterializedAugmentedSample:
    image_id: str
    output_image: Path
    output_label: Path
    num_labels: int
    boxes_inside_bounds: bool


def materialize_augmented_record(
    *,
    record: DatasetRecord,
    variant: AugmentationVariant,
    output_root: Path,
    rng: np.random.Generator,
    context: AugmentationContext,
) -> MaterializedAugmentedSample:
    """Write one augmented image and matching YOLO label file."""
    with Image.open(record.image_path) as image:
        original = image.convert("RGB")
        result = apply_augmentation(original, record_to_labels(record), variant, rng=rng, context=context)

    safe_id = record.image_id.replace("/", "__")
    output_image = output_root / variant.name / "images" / record.split / f"{safe_id}.png"
    output_label = output_root / variant.name / "labels" / record.split / f"{safe_id}.txt"
    save_rgb_image(result.image, output_image)
    output_label.parent.mkdir(parents=True, exist_ok=True)
    output_label.write_text("\n".join(labels_to_yolo_lines(result.labels)) + ("\n" if result.labels else ""), encoding="utf-8")
    return MaterializedAugmentedSample(
        image_id=record.image_id,
        output_image=output_image,
        output_label=output_label,
        num_labels=len(result.labels),
        boxes_inside_bounds=result.boxes_inside_bounds,
    )

