"""Safe preprocessing variants for wood-defect detection experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
import yaml
from PIL import Image


ImageArray = np.ndarray
PreprocessFn = Callable[[ImageArray, dict[str, Any]], ImageArray]


@dataclass(frozen=True)
class PreprocessingVariant:
    name: str
    method: str
    params: dict[str, Any]
    description: str = ""


def ensure_rgb_uint8(image: ImageArray | Image.Image) -> ImageArray:
    """Return an RGB uint8 image array without resizing."""
    if isinstance(image, Image.Image):
        image = np.asarray(image.convert("RGB"))
    arr = np.asarray(image)
    if arr.ndim == 2:
        arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
    if arr.ndim == 3 and arr.shape[2] == 4:
        arr = arr[:, :, :3]
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def p0_baseline(image: ImageArray, params: dict[str, Any] | None = None) -> ImageArray:
    return ensure_rgb_uint8(image).copy()


def p1_clahe_luminance(image: ImageArray, params: dict[str, Any] | None = None) -> ImageArray:
    params = params or {}
    rgb = ensure_rgb_uint8(image)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    tile_grid_size = int(params.get("tile_grid_size", 8))
    clahe = cv2.createCLAHE(
        clipLimit=float(params.get("clip_limit", 1.8)),
        tileGridSize=(tile_grid_size, tile_grid_size),
    )
    enhanced_l = clahe.apply(l_channel)
    blend = float(np.clip(float(params.get("blend", 0.55)), 0.0, 1.0))
    enhanced_l = cv2.addWeighted(enhanced_l, blend, l_channel, 1.0 - blend, 0)
    enhanced_lab = cv2.merge((enhanced_l, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def p2_illumination_normalization(image: ImageArray, params: dict[str, Any] | None = None) -> ImageArray:
    """Normalize luminance with clipped auto-gamma on the LAB L channel.

    This is deliberately conservative: only luminance is adjusted, and the
    gamma range is clipped to avoid washing out subtle defects.
    """
    params = params or {}
    rgb = ensure_rgb_uint8(image)
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    luminance = max(float(np.mean(l_channel)) / 255.0, 1e-4)
    target_mean = float(params.get("target_luminance", 0.50))
    gamma = np.log(target_mean) / np.log(luminance)
    gamma = float(np.clip(gamma, float(params.get("gamma_min", 0.85)), float(params.get("gamma_max", 1.18))))
    lut = np.array([((value / 255.0) ** gamma) * 255.0 for value in range(256)], dtype=np.float32)
    adjusted_l = cv2.LUT(l_channel, np.clip(lut, 0, 255).astype(np.uint8))
    adjusted_lab = cv2.merge((adjusted_l, a_channel, b_channel))
    return cv2.cvtColor(adjusted_lab, cv2.COLOR_LAB2RGB)


def p3_mild_unsharp(image: ImageArray, params: dict[str, Any] | None = None) -> ImageArray:
    params = params or {}
    rgb = ensure_rgb_uint8(image)
    sigma = float(params.get("sigma", 1.0))
    amount = float(params.get("amount", 0.25))
    blur = cv2.GaussianBlur(rgb, ksize=(0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = cv2.addWeighted(rgb, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def p4_combined_safe(image: ImageArray, params: dict[str, Any] | None = None) -> ImageArray:
    params = params or {}
    illum_params = params.get("illumination", {})
    clahe_params = params.get("clahe", {})
    normalized = p2_illumination_normalization(image, illum_params)
    return p1_clahe_luminance(normalized, clahe_params)


PREPROCESSING_REGISTRY: dict[str, PreprocessFn] = {
    "P0_baseline": p0_baseline,
    "P1_CLAHE_luminance": p1_clahe_luminance,
    "P2_illumination_normalization": p2_illumination_normalization,
    "P3_mild_unsharp": p3_mild_unsharp,
    "P4_combined_safe": p4_combined_safe,
}


def apply_preprocessing(image: ImageArray | Image.Image, variant: PreprocessingVariant) -> ImageArray:
    if variant.name not in PREPROCESSING_REGISTRY:
        raise KeyError(f"Unknown preprocessing variant: {variant.name}")
    original = ensure_rgb_uint8(image)
    processed = PREPROCESSING_REGISTRY[variant.name](original, variant.params)
    processed = ensure_rgb_uint8(processed)
    if processed.shape != original.shape:
        raise ValueError(
            f"Preprocessing variant {variant.name} changed image shape from {original.shape} to {processed.shape}"
        )
    return processed


def load_variant_config(config_path: Path) -> PreprocessingVariant:
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return PreprocessingVariant(
        name=str(raw["name"]),
        method=str(raw.get("method", raw["name"])),
        params=dict(raw.get("params") or {}),
        description=str(raw.get("description", "")),
    )


def load_all_variant_configs(config_dir: Path) -> list[PreprocessingVariant]:
    variants = [load_variant_config(path) for path in sorted(config_dir.glob("P*.yaml"))]
    ordered_names = list(PREPROCESSING_REGISTRY)
    return sorted(variants, key=lambda item: ordered_names.index(item.name))


def save_rgb_image(image: ImageArray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(ensure_rgb_uint8(image)).save(output_path)
