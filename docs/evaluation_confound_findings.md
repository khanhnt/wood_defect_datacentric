# Corrected Common Evaluation Audit

Date: 2026-06-22

## Verdict

The original multiseed materialized datasets are not valid for cross-variant
evaluation when an offline augmentation variant is involved. The materialization
code applies augmentation to `train`, `val`, and `test`, not train only.

This means A1 crop, A2 color jitter, and the A4 augmentation part of P4+A4 were
baked into validation/test images for their own generated YOLO datasets. Baseline
and pure preprocessing variants do not have this geometry-changing augmentation
confound.

## Code Evidence

- `scripts/materialize_augmented_yolo.py` defines `SPLITS = ("train", "val", "test")`
  and loops through every split. Inside `materialize_split`, it calls
  `apply_augmentation(...)` for every image it writes.
- `scripts/materialize_preprocessed_yolo.py` also defines `SPLITS = ("train", "val", "test")`
  and applies `apply_preprocessing(...)` to every split. This is acceptable for
  preprocessing variants because the image geometry and label membership are
  preserved.
- `scripts/run_all_experiments.py` uses:
  - `materialize_preprocess(...)` for P1/P2/P3;
  - `materialize_augment(...)` for A1/A2;
  - shared preprocessing followed by `materialize_augment(...)` for P4+A4.

Therefore P4+A4 validation/test had both P4 preprocessing and A4 augmentation
applied, including the defect-preserving crop.

## Transform x Split Table

| Variant group | Transform | Train | Val | Test | Correct for final eval? |
|---|---|---:|---:|---:|---|
| Baseline | No offline preprocessing/augmentation | no | no | no | yes |
| P1 CLAHE | Full-image CLAHE preprocessing | yes | yes | yes | yes |
| P2 illumination | Full-image illumination preprocessing | yes | yes | yes | yes |
| P3 unsharp | Full-image unsharp preprocessing | yes | yes | yes | yes |
| A1 crop | Defect-preserving crop augmentation | yes | yes | yes | no; train-only intended |
| A2 color jitter | Texture-aware color jitter augmentation | yes | yes | yes | no; train-only intended |
| P4+A4 | P4 full-image preprocessing | yes | yes | yes | yes |
| P4+A4 | A4 crop + color jitter augmentation | yes | yes | yes | no; train-only intended |

## Variant-Specific Consequence

For final corrected evaluation:

- Baseline, A1, and A2 should be evaluated on the same raw canonical YOLO dataset.
- P1/P2/P3 should be evaluated on full canonical images with only their respective
  preprocessing applied.
- P4+A4 should be evaluated on full canonical images with P4 preprocessing only;
  no A4 crop or color jitter should be present in val/test.

The checkpoint itself remains the existing trained `best.pt`. The caveat to report
is that for A1/A2/P4+A4 the saved `best.pt` was selected using a transformed
validation set, so final common-set evaluation is post-hoc and may not match the
checkpoint-selection validation number.

## Negative-Aware Sweep

The original `scripts/threshold_sweep_inference.py` resolves the dataset YAML from
each run's `config_used.yaml` unless a fixed override is passed. Prediction JSONs
from the previous sweep record generated per-variant dataset paths such as:

- `.../seed42/A1_defect_preserving_crop/dataset.yaml`
- `.../seed42/A2_texture_aware_color_jitter/dataset.yaml`
- `.../seed42/P4_combined_safe__A4_combined_best/dataset.yaml`

Thus the previous negative-aware sweep used transformed positive test images for
augmentation variants. A1 and P4+A4 positive test images were cropped; A2 positive
test images were color-jittered. The sweep should be regenerated on corrected
full-image test data:

- raw canonical test for Baseline/A1/A2;
- P2-preprocessed canonical test for P2;
- P4-preprocessed canonical test for P4+A4.

`scripts/threshold_sweep_inference.py` now supports per-variant corrected dataset
overrides via repeated `--variant-data-yaml VARIANT=PATH` arguments.

## VSB Note

The VSB rare-first labels are tile-level labels. If the `images/` symlink tree is
missing, do not manually copy `/workspace/data/main_dataset/images/{val,test}`;
that can select wrong tiles and produce artificially low mAP. Rebuild the YOLO
benchmark from:

`/workspace/data/main_dataset/benchmarks/vsb7_3600_rare_first/manifest.jsonl`

with `scripts/materialize_yolo_from_manifest.py` and `--link-mode symlink`.
