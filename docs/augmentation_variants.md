# Augmentation Variants

This note documents the first augmentation variants for the fixed-detector YOLOv8s data-centric study. The goal is defect-preserving augmentation, not aggressive synthetic expansion.

## Variants

| ID | Variant | Operation | Label handling | Status |
|---|---|---|---|---|
| A0 | `A0_default` | YOLOv8 default augmentation during training; preview is no-op reference | unchanged | baseline |
| A1 | `A1_defect_preserving_crop` | Crop around defects with context and resize back to original image size | boxes transformed after crop/resize | safe |
| A2 | `A2_texture_aware_color_jitter` | Mild brightness, contrast, and saturation jitter | unchanged | safe |
| A3 | `A3_copy_paste_defects` | Conservative feathered defect patch copy-paste with overlap guard | pasted boxes added | experimental |
| A4 | `A4_combined_best` | Defect-preserving crop plus controlled color jitter | boxes transformed by crop; color jitter leaves boxes unchanged | safe |

## Safeguards

- All preview outputs preserve image dimensions.
- Bounding boxes are checked to remain within normalized image bounds.
- A1 and A4 use a minimum box visibility guard and fall back if the crop would truncate a defect excessively.
- A2 avoids hue shifts and uses small brightness/contrast/saturation ranges to preserve realistic wood color.
- A3 uses only one pasted object by default, includes wood context around the defect patch, feather-blends patch borders, and avoids overlap with existing boxes.

## Copy-Paste Feasibility

Copy-paste is feasible as an experimental variant because VNWoodKnot has readable positive defect images and `knot_free` backgrounds. It should remain conservative in the paper because rectangular defect patches can still introduce subtle texture discontinuities. For the first IJACSA experiment matrix, A3 should be reported separately and not folded into the default "best" variant unless preview and validation behavior are clean.

## YOLO Training Strategy

For YOLOv8, the reproducible route is to materialize generated image folders and matching YOLO label files per augmentation variant. This project includes helper code for materialization but the current step only generates previews.

## Preview Command

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/preview_augmentation.py \
  --dataset vnwoodknot \
  --split test \
  --num-samples 4
```

Outputs are written to:

```text
results/augmentation_preview/
```

Expected files include:

- `A0_default_before_after.png`
- `A1_defect_preserving_crop_before_after.png`
- `A2_texture_aware_color_jitter_before_after.png`
- `A3_copy_paste_defects_before_after.png`
- `A4_combined_best_before_after.png`
- `augmentation_variants_overview.png`
- `augmentation_preview_manifest.csv`

Current local note: VNWoodKnot image paths are accessible and used for preview generation. The current VSB manifest image paths are not mounted locally, so VSB augmentation previews require mounting or reconfiguring the curated VSB image root first.

