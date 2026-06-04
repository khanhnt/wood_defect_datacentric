"""Defect-preserving augmentation variants for wood-defect detection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import yaml
from PIL import Image, ImageEnhance

from wood_defect_datacentric.datasets.adapters import Annotation, DatasetRecord
from wood_defect_datacentric.preprocessing.variants import ensure_rgb_uint8


ImageArray = np.ndarray


@dataclass(frozen=True)
class BoxLabel:
    class_name: str
    class_id: int | None
    bbox_xyxy_norm: tuple[float, float, float, float]
    source: str = "original"
    visibility: float = 1.0


@dataclass(frozen=True)
class AugmentationVariant:
    name: str
    method: str
    params: dict[str, Any]
    description: str = ""


@dataclass(frozen=True)
class AugmentationResult:
    image: ImageArray
    labels: tuple[BoxLabel, ...]
    notes: tuple[str, ...]
    min_visibility: float
    boxes_inside_bounds: bool
    num_added_boxes: int = 0


@dataclass(frozen=True)
class AugmentationContext:
    object_records: tuple[DatasetRecord, ...] = ()
    background_records: tuple[DatasetRecord, ...] = ()


AugmentationFn = Callable[
    [ImageArray, tuple[BoxLabel, ...], dict[str, Any], np.random.Generator, AugmentationContext],
    AugmentationResult,
]


def record_to_labels(record: DatasetRecord) -> tuple[BoxLabel, ...]:
    return tuple(annotation_to_label(ann) for ann in record.annotations)


def annotation_to_label(annotation: Annotation) -> BoxLabel:
    return BoxLabel(
        class_name=annotation.class_name,
        class_id=annotation.class_id,
        bbox_xyxy_norm=annotation.bbox_xyxy_norm,
        source="original",
        visibility=1.0,
    )


def a0_default(
    image: ImageArray,
    labels: tuple[BoxLabel, ...],
    params: dict[str, Any],
    rng: np.random.Generator,
    context: AugmentationContext,
) -> AugmentationResult:
    rgb = ensure_rgb_uint8(image).copy()
    return make_result(rgb, labels, notes=("default_yolo_training_aug_reference",))


def a1_defect_preserving_crop(
    image: ImageArray,
    labels: tuple[BoxLabel, ...],
    params: dict[str, Any],
    rng: np.random.Generator,
    context: AugmentationContext,
) -> AugmentationResult:
    rgb = ensure_rgb_uint8(image)
    if not labels:
        return _negative_safe_crop(rgb, labels, params, rng)

    min_visibility = float(params.get("min_box_visibility", 0.90))
    context_scale = float(params.get("context_scale", 2.25))
    min_crop_scale = float(params.get("min_crop_scale", 0.62))
    jitter = float(params.get("center_jitter", 0.04))

    height, width = rgb.shape[:2]
    pixel_boxes = np.array([norm_to_pixel(label.bbox_xyxy_norm, width, height) for label in labels], dtype=np.float32)
    union = np.array(
        [
            np.min(pixel_boxes[:, 0]),
            np.min(pixel_boxes[:, 1]),
            np.max(pixel_boxes[:, 2]),
            np.max(pixel_boxes[:, 3]),
        ],
        dtype=np.float32,
    )
    union_w = max(1.0, float(union[2] - union[0]))
    union_h = max(1.0, float(union[3] - union[1]))
    center_x = float((union[0] + union[2]) / 2.0)
    center_y = float((union[1] + union[3]) / 2.0)

    target_w = min(float(width), max(width * min_crop_scale, union_w * context_scale))
    target_h = min(float(height), max(height * min_crop_scale, union_h * context_scale))
    center_x += float(rng.uniform(-jitter, jitter) * target_w)
    center_y += float(rng.uniform(-jitter, jitter) * target_h)

    candidate_windows = [
        window_from_center(center_x, center_y, target_w, target_h, width, height),
        window_from_center(center_x, center_y, min(width, target_w * 1.18), min(height, target_h * 1.18), width, height),
        (0, 0, width, height),
    ]

    for crop_window in candidate_windows:
        transformed, visibility = transform_labels_for_crop(labels, crop_window, width, height, min_visibility)
        if len(transformed) == len(labels):
            cropped = crop_and_resize(rgb, crop_window, width, height)
            return make_result(cropped, transformed, notes=("crop_resize_preserved_all_boxes",), min_visibility=min(visibility))

    return make_result(rgb.copy(), labels, notes=("crop_skipped_visibility_guard",))


def a2_texture_aware_color_jitter(
    image: ImageArray,
    labels: tuple[BoxLabel, ...],
    params: dict[str, Any],
    rng: np.random.Generator,
    context: AugmentationContext,
) -> AugmentationResult:
    rgb = ensure_rgb_uint8(image)
    pil = Image.fromarray(rgb)
    brightness = sample_factor(rng, float(params.get("brightness", 0.10)))
    contrast = sample_factor(rng, float(params.get("contrast", 0.10)))
    saturation = sample_factor(rng, float(params.get("saturation", 0.06)))
    pil = ImageEnhance.Brightness(pil).enhance(brightness)
    pil = ImageEnhance.Contrast(pil).enhance(contrast)
    pil = ImageEnhance.Color(pil).enhance(saturation)
    return make_result(
        ensure_rgb_uint8(pil),
        labels,
        notes=(f"jitter_b={brightness:.3f}_c={contrast:.3f}_s={saturation:.3f}",),
    )


def a3_copy_paste_defects(
    image: ImageArray,
    labels: tuple[BoxLabel, ...],
    params: dict[str, Any],
    rng: np.random.Generator,
    context: AugmentationContext,
) -> AugmentationResult:
    rgb = ensure_rgb_uint8(image).copy()
    if not context.object_records:
        return make_result(rgb, labels, notes=("copy_paste_skipped_no_object_bank",))

    max_objects = int(params.get("max_objects_per_image", 1))
    max_iou = float(params.get("max_iou_with_existing", 0.05))
    feather_px = int(params.get("feather_px", 10))
    alpha_strength = float(params.get("alpha", 0.88))
    scale_min, scale_max = tuple(params.get("scale_range", [0.75, 1.05]))
    height, width = rgb.shape[:2]
    current_labels = list(labels)
    added = 0
    notes: list[str] = []

    donors = rank_donor_records(context.object_records)
    for donor in donors:
        if added >= max_objects:
            break
        donor_label = choose_donor_label(donor)
        if donor_label is None or not donor.image_path.exists():
            continue
        try:
            donor_image = load_rgb(donor.image_path)
        except Exception:
            continue
        patch, patch_label = extract_donor_patch(donor_image, donor_label, params)
        if patch.size == 0:
            continue
        scale = float(rng.uniform(float(scale_min), float(scale_max)))
        patch, patch_label = resize_patch_and_label(patch, patch_label, scale)
        placement = find_paste_location(
            image_shape=rgb.shape,
            patch_shape=patch.shape,
            new_label_in_patch=patch_label,
            existing_labels=tuple(current_labels),
            max_iou=max_iou,
            rng=rng,
            max_attempts=int(params.get("max_attempts", 60)),
        )
        if placement is None:
            notes.append(f"copy_paste_no_safe_location:{donor.image_id}")
            continue
        paste_x, paste_y = placement
        rgb = feathered_paste(rgb, patch, paste_x, paste_y, feather_px=feather_px, alpha_strength=alpha_strength)
        patch_h, patch_w = patch.shape[:2]
        new_label = patch_label_to_image_label(patch_label, paste_x, paste_y, patch_w, patch_h, width, height)
        current_labels.append(new_label)
        added += 1
        notes.append(f"copy_paste_added:{new_label.class_name}")

    if added == 0:
        notes.append("copy_paste_skipped_visibility_or_overlap_guard")
    return make_result(rgb, tuple(current_labels), notes=tuple(notes), num_added_boxes=added)


def a4_combined_best(
    image: ImageArray,
    labels: tuple[BoxLabel, ...],
    params: dict[str, Any],
    rng: np.random.Generator,
    context: AugmentationContext,
) -> AugmentationResult:
    crop_params = params.get("crop", {})
    jitter_params = params.get("color_jitter", {})
    cropped = a1_defect_preserving_crop(image, labels, crop_params, rng, context)
    jittered = a2_texture_aware_color_jitter(cropped.image, cropped.labels, jitter_params, rng, context)
    notes = tuple(cropped.notes) + tuple(jittered.notes) + ("combined_safe_crop_plus_color",)
    return make_result(
        jittered.image,
        jittered.labels,
        notes=notes,
        min_visibility=cropped.min_visibility,
        num_added_boxes=0,
    )


AUGMENTATION_REGISTRY: dict[str, AugmentationFn] = {
    "A0_default": a0_default,
    "A1_defect_preserving_crop": a1_defect_preserving_crop,
    "A2_texture_aware_color_jitter": a2_texture_aware_color_jitter,
    "A3_copy_paste_defects": a3_copy_paste_defects,
    "A4_combined_best": a4_combined_best,
}


def apply_augmentation(
    image: ImageArray | Image.Image,
    labels: tuple[BoxLabel, ...],
    variant: AugmentationVariant,
    *,
    rng: np.random.Generator | None = None,
    context: AugmentationContext | None = None,
) -> AugmentationResult:
    if variant.name not in AUGMENTATION_REGISTRY:
        raise KeyError(f"Unknown augmentation variant: {variant.name}")
    generator = rng or np.random.default_rng(42)
    ctx = context or AugmentationContext()
    original = ensure_rgb_uint8(image)
    result = AUGMENTATION_REGISTRY[variant.name](original, labels, variant.params, generator, ctx)
    result_image = ensure_rgb_uint8(result.image)
    if result_image.shape != original.shape:
        raise ValueError(f"{variant.name} changed output shape from {original.shape} to {result_image.shape}")
    if not valid_labels(result.labels):
        raise ValueError(f"{variant.name} produced invalid labels")
    return AugmentationResult(
        image=result_image,
        labels=result.labels,
        notes=result.notes,
        min_visibility=result.min_visibility,
        boxes_inside_bounds=result.boxes_inside_bounds,
        num_added_boxes=result.num_added_boxes,
    )


def make_result(
    image: ImageArray,
    labels: tuple[BoxLabel, ...],
    *,
    notes: tuple[str, ...],
    min_visibility: float = 1.0,
    num_added_boxes: int = 0,
) -> AugmentationResult:
    return AugmentationResult(
        image=ensure_rgb_uint8(image),
        labels=tuple(labels),
        notes=notes,
        min_visibility=float(min_visibility),
        boxes_inside_bounds=valid_labels(labels),
        num_added_boxes=num_added_boxes,
    )


def sample_factor(rng: np.random.Generator, magnitude: float) -> float:
    magnitude = max(0.0, magnitude)
    return float(rng.uniform(1.0 - magnitude, 1.0 + magnitude))


def norm_to_pixel(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return x1 * width, y1 * height, x2 * width, y2 * height


def pixel_to_norm(box: tuple[float, float, float, float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (
        float(np.clip(x1 / width, 0.0, 1.0)),
        float(np.clip(y1 / height, 0.0, 1.0)),
        float(np.clip(x2 / width, 0.0, 1.0)),
        float(np.clip(y2 / height, 0.0, 1.0)),
    )


def window_from_center(
    center_x: float,
    center_y: float,
    target_w: float,
    target_h: float,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    target_w = min(float(width), max(2.0, target_w))
    target_h = min(float(height), max(2.0, target_h))
    x1 = int(round(center_x - target_w / 2.0))
    y1 = int(round(center_y - target_h / 2.0))
    x1 = max(0, min(x1, width - int(round(target_w))))
    y1 = max(0, min(y1, height - int(round(target_h))))
    x2 = min(width, x1 + int(round(target_w)))
    y2 = min(height, y1 + int(round(target_h)))
    return x1, y1, x2, y2


def crop_and_resize(image: ImageArray, crop_window: tuple[int, int, int, int], width: int, height: int) -> ImageArray:
    x1, y1, x2, y2 = crop_window
    cropped = image[y1:y2, x1:x2]
    return cv2.resize(cropped, (width, height), interpolation=cv2.INTER_LINEAR)


def transform_labels_for_crop(
    labels: tuple[BoxLabel, ...],
    crop_window: tuple[int, int, int, int],
    width: int,
    height: int,
    min_visibility: float,
) -> tuple[tuple[BoxLabel, ...], list[float]]:
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_window
    crop_w = max(1, crop_x2 - crop_x1)
    crop_h = max(1, crop_y2 - crop_y1)
    transformed: list[BoxLabel] = []
    visibilities: list[float] = []
    for label in labels:
        x1, y1, x2, y2 = norm_to_pixel(label.bbox_xyxy_norm, width, height)
        original_area = max(1e-6, (x2 - x1) * (y2 - y1))
        clipped_x1 = max(x1, crop_x1)
        clipped_y1 = max(y1, crop_y1)
        clipped_x2 = min(x2, crop_x2)
        clipped_y2 = min(y2, crop_y2)
        clipped_area = max(0.0, clipped_x2 - clipped_x1) * max(0.0, clipped_y2 - clipped_y1)
        visibility = clipped_area / original_area
        if visibility < min_visibility:
            continue
        new_box = (
            (clipped_x1 - crop_x1) / crop_w,
            (clipped_y1 - crop_y1) / crop_h,
            (clipped_x2 - crop_x1) / crop_w,
            (clipped_y2 - crop_y1) / crop_h,
        )
        if not is_valid_box(new_box):
            continue
        transformed.append(
            BoxLabel(
                class_name=label.class_name,
                class_id=label.class_id,
                bbox_xyxy_norm=tuple(float(np.clip(value, 0.0, 1.0)) for value in new_box),
                source=label.source,
                visibility=float(visibility),
            )
        )
        visibilities.append(float(visibility))
    return tuple(transformed), visibilities


def _negative_safe_crop(
    image: ImageArray,
    labels: tuple[BoxLabel, ...],
    params: dict[str, Any],
    rng: np.random.Generator,
) -> AugmentationResult:
    height, width = image.shape[:2]
    min_scale = float(params.get("negative_min_crop_scale", 0.76))
    crop_scale = float(rng.uniform(min_scale, 0.96))
    crop_w = int(round(width * crop_scale))
    crop_h = int(round(height * crop_scale))
    x1 = int(rng.integers(0, max(1, width - crop_w + 1)))
    y1 = int(rng.integers(0, max(1, height - crop_h + 1)))
    crop_window = (x1, y1, x1 + crop_w, y1 + crop_h)
    cropped = crop_and_resize(image, crop_window, width, height)
    return make_result(cropped, labels, notes=("negative_background_crop_resize",))


def valid_labels(labels: tuple[BoxLabel, ...]) -> bool:
    return all(is_valid_box(label.bbox_xyxy_norm) for label in labels)


def is_valid_box(box: tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = box
    return 0.0 <= x1 < x2 <= 1.0 and 0.0 <= y1 < y2 <= 1.0


def load_rgb(path: Path) -> ImageArray:
    with Image.open(path) as image:
        return ensure_rgb_uint8(image)


def rank_donor_records(records: tuple[DatasetRecord, ...]) -> list[DatasetRecord]:
    def score(record: DatasetRecord) -> tuple[float, str]:
        areas = [ann.area_norm for ann in record.annotations]
        return (min(areas) if areas else 1.0, record.image_id)

    return sorted((record for record in records if record.annotations), key=score)


def choose_donor_label(record: DatasetRecord) -> BoxLabel | None:
    if not record.annotations:
        return None
    annotation = min(record.annotations, key=lambda ann: ann.area_norm)
    return annotation_to_label(annotation)


def extract_donor_patch(
    donor_image: ImageArray,
    donor_label: BoxLabel,
    params: dict[str, Any],
) -> tuple[ImageArray, BoxLabel]:
    height, width = donor_image.shape[:2]
    x1, y1, x2, y2 = norm_to_pixel(donor_label.bbox_xyxy_norm, width, height)
    pad_ratio = float(params.get("context_pad_ratio", 0.18))
    pad_x = (x2 - x1) * pad_ratio
    pad_y = (y2 - y1) * pad_ratio
    patch_x1 = int(max(0, np.floor(x1 - pad_x)))
    patch_y1 = int(max(0, np.floor(y1 - pad_y)))
    patch_x2 = int(min(width, np.ceil(x2 + pad_x)))
    patch_y2 = int(min(height, np.ceil(y2 + pad_y)))
    patch = donor_image[patch_y1:patch_y2, patch_x1:patch_x2].copy()
    patch_h, patch_w = patch.shape[:2]
    if patch_w < 2 or patch_h < 2:
        return patch, donor_label
    label_in_patch = BoxLabel(
        class_name=donor_label.class_name,
        class_id=donor_label.class_id,
        bbox_xyxy_norm=(
            (x1 - patch_x1) / patch_w,
            (y1 - patch_y1) / patch_h,
            (x2 - patch_x1) / patch_w,
            (y2 - patch_y1) / patch_h,
        ),
        source="copy_paste",
        visibility=1.0,
    )
    return patch, label_in_patch


def resize_patch_and_label(patch: ImageArray, label: BoxLabel, scale: float) -> tuple[ImageArray, BoxLabel]:
    height, width = patch.shape[:2]
    new_w = max(2, int(round(width * scale)))
    new_h = max(2, int(round(height * scale)))
    resized = cv2.resize(patch, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, label


def find_paste_location(
    *,
    image_shape: tuple[int, int, int],
    patch_shape: tuple[int, int, int],
    new_label_in_patch: BoxLabel,
    existing_labels: tuple[BoxLabel, ...],
    max_iou: float,
    rng: np.random.Generator,
    max_attempts: int,
) -> tuple[int, int] | None:
    height, width = image_shape[:2]
    patch_h, patch_w = patch_shape[:2]
    if patch_w >= width or patch_h >= height:
        return None
    existing = [label.bbox_xyxy_norm for label in existing_labels]
    for _ in range(max_attempts):
        paste_x = int(rng.integers(0, max(1, width - patch_w)))
        paste_y = int(rng.integers(0, max(1, height - patch_h)))
        candidate = patch_label_to_image_label(new_label_in_patch, paste_x, paste_y, patch_w, patch_h, width, height)
        if not is_valid_box(candidate.bbox_xyxy_norm):
            continue
        if all(box_iou(candidate.bbox_xyxy_norm, box) <= max_iou for box in existing):
            return paste_x, paste_y
    return None


def patch_label_to_image_label(
    patch_label: BoxLabel,
    paste_x: int,
    paste_y: int,
    patch_w: int,
    patch_h: int,
    image_w: int,
    image_h: int,
) -> BoxLabel:
    x1, y1, x2, y2 = patch_label.bbox_xyxy_norm
    return BoxLabel(
        class_name=patch_label.class_name,
        class_id=patch_label.class_id,
        bbox_xyxy_norm=(
            (paste_x + x1 * patch_w) / image_w,
            (paste_y + y1 * patch_h) / image_h,
            (paste_x + x2 * patch_w) / image_w,
            (paste_y + y2 * patch_h) / image_h,
        ),
        source="copy_paste",
        visibility=1.0,
    )


def feathered_paste(
    background: ImageArray,
    patch: ImageArray,
    paste_x: int,
    paste_y: int,
    *,
    feather_px: int,
    alpha_strength: float,
) -> ImageArray:
    patch_h, patch_w = patch.shape[:2]
    output = background.copy()
    region = output[paste_y : paste_y + patch_h, paste_x : paste_x + patch_w].astype(np.float32)
    patch_float = patch.astype(np.float32)
    alpha = make_feather_mask(patch_h, patch_w, feather_px)[:, :, None] * float(np.clip(alpha_strength, 0.0, 1.0))
    blended = patch_float * alpha + region * (1.0 - alpha)
    output[paste_y : paste_y + patch_h, paste_x : paste_x + patch_w] = np.clip(blended, 0, 255).astype(np.uint8)
    return output


def make_feather_mask(height: int, width: int, feather_px: int) -> ImageArray:
    if feather_px <= 0:
        return np.ones((height, width), dtype=np.float32)
    yy, xx = np.mgrid[0:height, 0:width]
    dist = np.minimum.reduce([xx, yy, width - 1 - xx, height - 1 - yy]).astype(np.float32)
    return np.clip(dist / max(1.0, float(feather_px)), 0.0, 1.0)


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_variant_config(config_path: Path) -> AugmentationVariant:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return AugmentationVariant(
        name=str(raw["name"]),
        method=str(raw.get("method", raw["name"])),
        params=dict(raw.get("params") or {}),
        description=str(raw.get("description", "")),
    )


def load_all_variant_configs(config_dir: Path) -> list[AugmentationVariant]:
    variants = [load_variant_config(path) for path in sorted(config_dir.glob("A*.yaml"))]
    ordered_names = list(AUGMENTATION_REGISTRY)
    return sorted(variants, key=lambda item: ordered_names.index(item.name))


def labels_to_yolo_lines(labels: tuple[BoxLabel, ...]) -> list[str]:
    lines: list[str] = []
    for label in labels:
        class_id = 0 if label.class_id is None else int(label.class_id)
        x1, y1, x2, y2 = label.bbox_xyxy_norm
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        width = x2 - x1
        height = y2 - y1
        lines.append(f"{class_id} {cx:.8f} {cy:.8f} {width:.8f} {height:.8f}")
    return lines
