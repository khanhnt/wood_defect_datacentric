"""Helpers for safely materializing preprocessed image datasets.

The project prefers generated image folders for Ultralytics YOLO experiments
because YOLO training reads image paths directly. This module provides reusable
building blocks only; the current step does not generate full datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import shutil

from PIL import Image

from wood_defect_datacentric.datasets.adapters import DatasetRecord
from wood_defect_datacentric.preprocessing.variants import (
    PreprocessingVariant,
    apply_preprocessing,
    save_rgb_image,
)


@dataclass(frozen=True)
class MaterializedImage:
    image_id: str
    source_image: Path
    output_image: Path
    width: int
    height: int
    labels_unchanged: bool


def materialize_record_image(
    *,
    record: DatasetRecord,
    variant: PreprocessingVariant,
    output_root: Path,
    relative_image_path: Path | None = None,
) -> MaterializedImage:
    """Write one preprocessed image while leaving annotations untouched."""
    if not record.image_path.exists():
        raise FileNotFoundError(f"Missing source image: {record.image_path}")

    with Image.open(record.image_path) as image:
        rgb = image.convert("RGB")
        original_size = rgb.size
        processed = apply_preprocessing(rgb, variant)

    relative_path = relative_image_path or Path(record.split) / f"{record.image_id.replace('/', '__')}.png"
    output_image = output_root / variant.name / "images" / relative_path
    save_rgb_image(processed, output_image)

    with Image.open(output_image) as written:
        if written.size != original_size:
            raise ValueError(f"Image size changed for {record.image_id}: {original_size} -> {written.size}")
        written.verify()

    return MaterializedImage(
        image_id=record.image_id,
        source_image=record.image_path,
        output_image=output_image,
        width=original_size[0],
        height=original_size[1],
        labels_unchanged=True,
    )


def copy_label_file(source_label: Path, output_label: Path) -> None:
    """Copy a label file byte-for-byte to preserve annotation coordinates."""
    output_label.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_label, output_label)

