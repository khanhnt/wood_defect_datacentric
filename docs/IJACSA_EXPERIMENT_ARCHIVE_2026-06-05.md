# IJACSA Wood Defect Data-Centric Experiment Archive

Date: 2026-06-05

This document records the experimental state before backing up the Vast.ai
workspace and starting any additional experiments. It is intended as a compact
handoff for manuscript writing, result recovery, and future reruns.

## Project Scope

This is an independent IJACSA-oriented project separated from the earlier CMC
paper, "From Public Benchmarks to a Low-Resource Target Domain: A Comparative
Study of Wood Surface Defect Detection."

The IJACSA direction is data-centric rather than detector-comparison-oriented:

- Fixed detector: YOLOv8s.
- Data-centric preprocessing variants.
- Data-centric augmentation variants.
- Negative-aware evaluation on VNWoodKnot.
- Main datasets: VSB rare-first curated benchmark and VNWoodKnot.

The project should avoid overlap with the accepted CMC contribution, which
focused on detector-family comparison, benchmark/protocol sensitivity,
source-to-target transfer, and YOLOv8s versus Faster R-CNN/refinements.

## Server Environment

Main training was performed on Vast.ai.

Observed working environment:

- GPU: NVIDIA GeForce RTX 3090, 24 GB VRAM.
- Server had two RTX 3090 GPUs available.
- Driver/CUDA from `nvidia-smi`: NVIDIA driver 550.78, CUDA 12.4.
- Python: 3.12.13.
- Working PyTorch: 2.6.0+cu124.
- Ultralytics: 8.4.60.

Important environment note:

- The initial environment used `torch 2.12.0+cu130`, which was incompatible
  with the CUDA 12.4 driver and caused `torch.cuda.is_available() == False`.
- The fix was to reinstall the PyTorch stack with CUDA 12.4 wheels.

## Dataset Materialization

### VNWoodKnot

Source folder on Vast:

```text
/workspace/data/vnwoodknot
```

YOLO materialized dataset:

```text
/workspace/data/vnwoodknot/benchmarks/vnwoodknot_live_dead_2class_yolo
```

Materialization command:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/materialize_yolo_from_manifest.py \
  --manifest /workspace/data/vnwoodknot/manifest.jsonl \
  --images-root /workspace/data/vnwoodknot/images \
  --output-root /workspace/data/vnwoodknot/benchmarks/vnwoodknot_live_dead_2class_yolo \
  --dataset-name vnwoodknot_live_dead_2class_yolo \
  --classes live_knot dead_knot \
  --split-strategy manifest \
  --link-mode symlink
```

Final materialization status:

| Split | Images | Labels | Empty labels |
|---|---:|---:|---:|
| train | 1059 | 1059 | 350 |
| val | 226 | 226 | 75 |
| test | 228 | 228 | 75 |

Other VNWoodKnot details:

- `records_seen`: 1515.
- `records_written`: 1513.
- `skipped_missing_image`: 2.
- Missing image examples: `img_3671` in train/test `dead_knot`.
- Class box counts: `dead_knot=500`, `live_knot=519`.
- Negative-only images exist and are important for negative-aware evaluation.

### VSB Rare-First 3600

Source metadata folder on Vast:

```text
/workspace/data/main_dataset/benchmarks/vsb7_3600_rare_first
```

YOLO materialized dataset:

```text
/workspace/data/main_dataset/benchmarks/vsb7_3600_rare_first_yolo
```

Materialization command:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/materialize_yolo_from_manifest.py \
  --manifest /workspace/data/main_dataset/benchmarks/vsb7_3600_rare_first/manifest.jsonl \
  --images-root /workspace/data/main_dataset/images \
  --output-root /workspace/data/main_dataset/benchmarks/vsb7_3600_rare_first_yolo \
  --dataset-name vsb7_3600_rare_first_yolo \
  --classes live_knot dead_knot resin knot_with_crack crack marrow knot_missing \
  --split-strategy manifest \
  --link-mode symlink
```

Rare-first source metadata checks:

- `selected_source_ids.txt`: 3600 unique source images.
- `manifest.jsonl`: 9628 tile-level records.
- Manifest source IDs exactly match the selected source IDs.
- Duplicate image IDs: 0.
- Duplicate image paths: 0.
- Bad class IDs: 0.
- Bad boxes: 0.

VSB rare-first split:

| Split | Source images | Tile records |
|---|---:|---:|
| train | 2880 | 7679 |
| val | 360 | 977 |
| test | 360 | 972 |

Negative/empty tile counts:

| Split | Empty tiles |
|---|---:|
| train | 2297 |
| val | 276 |
| test | 276 |
| total | 2849 |

VSB rare-first class box counts:

| Class | Boxes |
|---|---:|
| live_knot | 3085 |
| dead_knot | 2421 |
| crack | 2284 |
| marrow | 1233 |
| knot_with_crack | 1149 |
| resin | 990 |
| knot_missing | 503 |

The earlier VSB random 3600 subset produced poor baseline performance and
should not be used as the main protocol. Rare-first is the current VSB protocol
for the IJACSA experiments.

## Experiment Families

Preprocessing variants:

- P0 baseline/no extra preprocessing.
- P1 CLAHE on luminance channel.
- P2 illumination/gamma normalization.
- P3 mild unsharp masking.
- P4 safe combined preprocessing.

Augmentation variants:

- A0 default YOLO augmentation.
- A1 defect-preserving crop.
- A2 texture-aware color jitter.
- A3 copy-paste defects, optional/experimental, not used in the main result set.
- A4 best combined augmentation.

Main VNWoodKnot experiment set:

- Baseline.
- P2 illumination.
- A1 crop.
- P4+A4 combined.

Main VSB rare-first experiment set:

- Baseline.
- P1 CLAHE.
- P2 illumination.
- P3 unsharp.
- A1 crop.
- A2 color jitter.
- P4+A4 combined.

## VNWoodKnot Validation Results

| Variant | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Baseline | 0.810 | 0.823 | 0.814 | 0.458 |
| P2 Illumination | 0.772 | 0.793 | 0.837 | 0.476 |
| A1 Crop | 0.861 | 0.904 | 0.917 | 0.494 |
| P4+A4 Combined | 0.860 | 0.859 | 0.911 | 0.505 |

Interpretation:

- A1 crop achieved the best validation mAP50.
- P4+A4 achieved the best validation mAP50-95.
- P2 improved over baseline in mAP while using only preprocessing.
- Validation metrics alone would favor A1 or P4+A4.

## VNWoodKnot Negative-Aware Results

Negative-aware evaluation was run on VNWoodKnot with threshold sensitivity.
The key metrics include AP50, precision, recall, and false-positive image
counts on `knot_free` images.

### Threshold 0.25

| Variant | AP50 | Precision | Recall | Knot-free FP images |
|---|---:|---:|---:|---:|
| Baseline | 0.779 | 0.614 | 0.871 | 1/75 |
| P2 Illumination | 0.760 | 0.581 | 0.858 | 0/75 |
| A1 Crop | 0.664 | 0.559 | 0.761 | 4/75 |
| P4+A4 Combined | 0.728 | 0.612 | 0.794 | 0/75 |

### Threshold 0.50

| Variant | AP50 | Precision | Recall | Knot-free FP images |
|---|---:|---:|---:|---:|
| Baseline | 0.507 | 0.847 | 0.535 | 0/75 |
| P2 Illumination | 0.550 | 0.856 | 0.574 | 0/75 |
| A1 Crop | 0.504 | 0.816 | 0.542 | 0/75 |
| P4+A4 Combined | 0.621 | 0.824 | 0.665 | 0/75 |

Interpretation:

- A1 crop is strong on validation but weak under negative-aware evaluation at
  threshold 0.25, producing false positives on 4/75 knot-free images.
- P2 illumination gives a clean zero-FP operating point at threshold 0.25 with
  modest AP/recall trade-off.
- P4+A4 gives the strongest zero-FP operating point at threshold 0.50, with
  AP50 0.621 and recall 0.665.
- Negative-aware evaluation changes the model-selection conclusion compared
  with standard validation metrics.

## VSB Rare-First Batch-16 Final Results

All listed VSB results below are batch-size controlled at `batch=16`.

| Variant | Run ID | Best epoch | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---:|---:|---:|---:|---:|
| Baseline | `vsb_rf_retry_20260605_002506_baseline_b16_e50` | 43 | 0.79039 | 0.79806 | 0.83286 | 0.48916 |
| P1 CLAHE | `vsb_rf_b16_20260605_071859_p1_clahe_e50` | 42 | 0.76975 | 0.77722 | 0.82402 | 0.47063 |
| P2 Illumination | `vsb_rf_retry_20260605_002506_p2_illumination_b16_e50` | 43 | 0.80443 | 0.77663 | 0.82393 | 0.47707 |
| P3 Unsharp | `vsb_rf_b16_20260605_071859_p3_unsharp_e50` | 29 | 0.75371 | 0.79584 | 0.82132 | 0.47029 |
| A1 Crop | `vsb_rf_retry_20260605_002506_a1_crop_b16_e50` | 45 | 0.80787 | 0.78978 | 0.83711 | 0.48668 |
| A2 Color Jitter | `vsb_rf_b16_20260605_071859_a2_colorjitter_e50` | 45 | 0.78680 | 0.77835 | 0.82775 | 0.47936 |
| P4+A4 Combined | `vsb_rf_retry_20260605_002506_p4_a4_combined_b16_e50` | 49 | 0.82835 | 0.76244 | 0.83342 | 0.48163 |

Interpretation:

- Baseline YOLOv8s on VSB rare-first is already strong: mAP50 0.833 and
  mAP50-95 0.489.
- A1 crop gives the highest mAP50: 0.837, a small improvement over baseline.
- Baseline keeps the highest mAP50-95: 0.489.
- P4+A4 gives the highest precision: 0.828, but lower recall.
- P1/P2/P3 preprocessing alone does not clearly outperform baseline.
- The VSB result suggests that when class coverage and protocol are controlled,
  additional preprocessing/augmentation provides only marginal gains.

## Current Paper-Level Findings

The current evidence supports a complete IJACSA manuscript draft.

Main findings:

1. With the detector fixed to YOLOv8s, data-centric augmentation and
   preprocessing can change validation conclusions, especially on VNWoodKnot.
2. Negative-aware evaluation on VNWoodKnot changes model selection: A1 crop is
   best by validation mAP50 but produces more false positives on knot-free
   images at low confidence threshold.
3. P2 illumination and P4+A4 provide more operationally safe choices when
   false positives on negative-only images matter.
4. On VSB rare-first, baseline YOLOv8s is already strong and most data-centric
   variants produce only marginal changes, showing that benefits are
   dataset/protocol dependent.
5. The earlier random VSB 3600 subset was not suitable as the main protocol;
   rare-first sampling produced stable and interpretable VSB results.

Suggested manuscript framing:

> A fixed-detector data-centric study of wood surface defect detection showing
> that preprocessing, augmentation, dataset sampling, and negative-aware
> evaluation can change conclusions even when the detector architecture is held
> constant.

## Recommended Manuscript Tables and Figures

Tables:

- Table 1: Dataset statistics for VSB rare-first and VNWoodKnot.
- Table 2: VNWoodKnot validation results.
- Table 3: VNWoodKnot negative-aware threshold results.
- Table 4: VSB rare-first batch-16 results.

Figures:

- Data-centric pipeline figure: manifest to materialized YOLO dataset,
  preprocessing/augmentation, YOLOv8s, standard and negative-aware evaluation.
- VNWoodKnot threshold sensitivity: AP50, precision/recall, FP per knot-free
  image.
- VSB mAP50/mAP50-95 bar chart by variant.

## Backup Checklist Before Additional Experiments

Back up these directories/files from Vast.ai before deleting or starting a new
experiment:

```text
/workspace/wood_defect_datacentric/results/runs
/workspace/wood_defect_datacentric/results/negative_eval
/workspace/wood_defect_datacentric/results/summaries
/workspace/wood_defect_datacentric/configs/experiments
/workspace/wood_defect_datacentric/scripts
/workspace/data/main_dataset/benchmarks/vsb7_3600_rare_first
/workspace/data/main_dataset/benchmarks/vsb7_3600_rare_first_yolo
/workspace/data/vnwoodknot/benchmarks/vnwoodknot_live_dead_2class_yolo
```

Optional but useful to back up:

```text
/workspace/wood_defect_datacentric/docs
/workspace/wood_defect_datacentric/configs
/workspace/data/main_dataset/manifest.jsonl
/workspace/data/main_dataset/metadata.json
/workspace/data/vnwoodknot/manifest.jsonl
/workspace/data/vnwoodknot/metadata.json
```

Large source image folders are needed only if the server instance will be
destroyed and the data cannot be reconstructed elsewhere:

```text
/workspace/data/main_dataset/images
/workspace/data/vnwoodknot/images
```

Safe-to-delete or reproducible folders after backup:

```text
/workspace/data/wood_defect_datacentric/generated_yolo
/workspace/data/wood_defect_datacentric/generated_yolo/vsb_rf_tmp_*
/workspace/data/wood_defect_datacentric/generated_yolo/vsb_rf_b16_tmp_*
/workspace/data/wood_defect_datacentric/generated_yolo/vnwoodknot
```

Do not delete the current main YOLO datasets unless they have been backed up:

```text
/workspace/data/main_dataset/benchmarks/vsb7_3600_rare_first_yolo
/workspace/data/vnwoodknot/benchmarks/vnwoodknot_live_dead_2class_yolo
```

## Caveats and Remaining Work

- Experiments use a fixed YOLOv8s detector and should not be claimed as a
  detector-family comparison.
- Main runs are single-seed; report this as a limitation.
- Negative-aware evaluation is implemented and emphasized for VNWoodKnot; VSB
  is mainly used for controlled data-centric variant comparison.
- A3 copy-paste remains optional/experimental and is not part of the main
  result set.
- Before submission, generate final publication-quality plots and ensure all
  CSV summaries are saved locally and in cloud backup.
