# Preprocessing Variants

This note documents the first data-centric preprocessing variants for the fixed-detector YOLOv8s study. The methods are intentionally conservative so they can be justified as defect-preserving image normalization rather than aggressive image enhancement.

## Variants

| ID | Variant | Operation | Geometry | Labels |
|---|---|---|---|---|
| P0 | `P0_baseline` | No extra preprocessing | unchanged | unchanged |
| P1 | `P1_CLAHE_luminance` | Mild blended CLAHE on LAB luminance only | unchanged | unchanged |
| P2 | `P2_illumination_normalization` | Clipped auto-gamma on LAB luminance | unchanged | unchanged |
| P3 | `P3_mild_unsharp` | Mild unsharp masking | unchanged | unchanged |
| P4 | `P4_combined_safe` | Conservative luminance normalization followed by mild blended CLAHE | unchanged | unchanged |

## Design Rationale

- All variants operate on RGB images and return RGB `uint8` images with the same height and width.
- Bounding-box coordinates are never modified because preprocessing does not crop, resize, pad, rotate, or warp images.
- Luminance-only methods are preferred for CLAHE and illumination normalization to avoid artificial class cues from color shifts.
- Parameter values are mild by design: the goal is to test whether safe preprocessing helps, not to maximize contrast manually.

## YOLO Training Strategy

For Ultralytics YOLO experiments, the safer reproducible route is to materialize generated image folders per preprocessing variant and copy label files byte-for-byte. This avoids hidden on-the-fly transforms inside YOLO dataloaders and makes each run auditable. The current step implements reusable materialization helpers but only generates preview samples.

## Preview Command

From the repository root:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/preview_preprocessing.py \
  --dataset vnwoodknot \
  --split test \
  --num-samples 4
```

Outputs are written to:

```text
results/preprocessing_preview/
```

Expected files include:

- `P0_baseline_before_after.png`
- `P1_CLAHE_luminance_before_after.png`
- `P2_illumination_normalization_before_after.png`
- `P3_mild_unsharp_before_after.png`
- `P4_combined_safe_before_after.png`
- `preprocessing_variants_overview.png`
- `preprocessing_preview_manifest.csv`

## Sanity Checks

The preview script checks:

- source image exists and can be opened;
- processed image can be written and reopened;
- image dimensions are unchanged;
- annotation signatures are unchanged;
- no label files or original images are overwritten.

Current local note: VNWoodKnot image paths are accessible on this machine. The current VSB manifest points to image paths that are not mounted locally, so VSB preview generation will require mounting or reconfiguring the curated VSB image root first.
