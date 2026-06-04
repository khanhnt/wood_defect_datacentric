"""Offline augmentation builders for wood defect detection."""
"""Augmentation variants for data-centric wood-defect experiments."""

from wood_defect_datacentric.augmentation.variants import (
    AUGMENTATION_REGISTRY,
    AugmentationContext,
    AugmentationResult,
    AugmentationVariant,
    BoxLabel,
    apply_augmentation,
    load_all_variant_configs,
    load_variant_config,
    record_to_labels,
)

__all__ = [
    "AUGMENTATION_REGISTRY",
    "AugmentationContext",
    "AugmentationResult",
    "AugmentationVariant",
    "BoxLabel",
    "apply_augmentation",
    "load_all_variant_configs",
    "load_variant_config",
    "record_to_labels",
]
