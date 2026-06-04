# Prompt 6.5 Server Setup Report

Date: 2026-06-04

## Verdict

Not ready for Prompt 7 training from this environment.

This audit was executed in the current local workspace, not on the target Vast.ai RTX 3090 server. The codebase is structurally ready for server verification, but the actual server checks are blocked here because GPU/CUDA, Ultralytics, `/workspace` paths, and mounted datasets are unavailable.

No full training was started. No source datasets were modified. Existing standard outputs under `results/runs/`, `results/preprocessing_preview/`, `results/augmentation_preview/`, and `results/negative_eval/` were left untouched.

## Files Added

- `configs/server.yaml`: expected Vast.ai paths, environment overrides, Prompt 6.5 sanity checks, and RTX 3090 batch recommendation.
- `scripts/server_setup_sanity.py`: reusable verification-only helper for environment, dataset, YOLO dataset YAML, baseline dry-run, and synthetic negative-eval prediction setup.

## Evidence Directory

Prompt 6.5 outputs were written under:

`results/server_setup_20260604_local_audit_final/`

Important files:

- `server_setup_sanity_summary.json`
- `dataset_split_verification.csv`
- `dataset_manifest_summary.json`
- `yolo_dataset_path_checks.json`
- `baseline_training_dry_run_checks.json`
- `launcher_dry_runs/runs/vsb_yolov8s_baseline_e50/validation_status.json`
- `launcher_dry_runs/runs/vn_t0_yolov8s_baseline_e50/validation_status.json`
- `negative_eval/prompt_6_5_synthetic_empty_negative_eval_threshold_metrics.csv`
- `negative_eval_report.md`

## Environment Check

Observed environment:

- Platform: macOS ARM, not Vast.ai Linux.
- GPU: unavailable; `nvidia-smi` not found.
- CUDA: unavailable; PyTorch reports `cuda_available=False` and `cuda_version=None`.
- Python used by sanity script: `/Users/ntkhanh/miniforge3/bin/python`, version 3.9.7.
- PyTorch: 2.4.0, CPU/no CUDA in this environment.
- Ultralytics: not installed in the active Python environment.
- YOLO CLI: not found.
- Disk free at project root: about 7.7 GiB, insufficient for training artifacts.

Blocker: this environment cannot verify RTX 3090 VRAM, CUDA runtime, GPU training readiness, or Ultralytics execution.

## Path Check

Expected Vast paths from `configs/server.yaml`:

- Code root: `/workspace/wood_defect_datacentric`
- Dataset root: `/workspace/data`
- VSB YAML: `/workspace/data/main_dataset/benchmarks/vsb7_3600_rare_first_yolo/dataset.yaml`
- VNWoodKnot YAML: `/workspace/data/vnwoodknot/benchmarks/vnwoodknot_live_dead_2class_yolo/dataset.yaml`
- Results root: `/workspace/wood_defect_datacentric/results`

All `/workspace/...` paths are absent in the current local environment. The YOLO dataset image/label matching check therefore could not run here.

## Dataset Verification

VNWoodKnot manifest verification passed for split structure, classes, bbox validity, and negative retention:

- Train: 1,060 images, 710 positive, 350 `knot_free`.
- Val: 226 images, 151 positive, 75 `knot_free`.
- Test: 229 images, 154 positive, 75 `knot_free`.
- Classes observed: `dead_knot`, `live_knot`.
- `knot_free` images with boxes: 0.
- Bbox validity failures: 0.

Warning: all VNWoodKnot image paths are missing locally because the source dataset is not mounted.

VSB manifest status:

- 20,276 manifest records are present.
- The manifest is audit context, not the final curated YOLO split.
- Split labels are unspecified in this source.
- Local image paths are unavailable.
- Unexpected labels outside the configured seven-class set are present in the raw/audit manifest: `blue_stain`, `overgrown`, `quartzity`.
- 156 records have invalid normalized boxes in this audit manifest.

Action before Prompt 7: rely on and verify the curated VSB YOLO dataset YAML on Vast, not the raw VSB audit manifest.

## Preview Checks

Preprocessing preview command was run with a scoped output path:

`python scripts/preview_preprocessing.py --dataset vnwoodknot --split test --num-samples 2 --output-dir results/server_setup_20260604_local_audit_final/preprocessing_preview`

Result: blocked because no readable image candidates were found. This is expected locally because VNWoodKnot image files are not mounted.

Augmentation preview command was run with a scoped output path:

`python scripts/preview_augmentation.py --dataset vnwoodknot --split test --num-samples 2 --output-dir results/server_setup_20260604_local_audit_final/augmentation_preview`

Result: blocked for the same missing-image reason.

## Training Dry-Runs

Safe launcher dry-runs were executed through copied configs whose `output_root` points inside:

`results/server_setup_20260604_local_audit_final/launcher_dry_runs/`

Both dry-runs stopped before training, as intended:

- `vsb_yolov8s_baseline_e50`: blocked by missing VSB dataset YAML.
- `vn_t0_yolov8s_baseline_e50`: blocked by missing VNWoodKnot dataset YAML.

No full training was executed.

## Negative-Aware Evaluation Dry-Run

The negative-aware evaluator was run using synthetic empty predictions for all VNWoodKnot test records. This checks alignment, threshold metric calculation, CSV writing, plots, and Markdown report generation without requiring a checkpoint.

Result: passed.

Evaluated records:

- Test images: 229.
- Targets: 155.
- `knot_free` test images: 75.

As expected for empty predictions, AP50, precision, recall, false-positive image rate, and mean false-positive confidence are all 0.0 at thresholds 0.10, 0.25, and 0.50.

## RTX 3090 Batch Recommendation

Recommended initial Prompt 7 batch size for YOLOv8s at `imgsz=1024` on RTX 3090 24 GB:

- Start with `batch=16`.
- If the first short probe has comfortable VRAM headroom, try `batch=24`.
- If CUDA OOM occurs, fall back to `batch=8`.

The current project configs default to `batch=32`; treat that as optimistic for 1024-pixel YOLOv8s training until the RTX 3090 memory probe passes.

## Blockers Before Prompt 7

Prompt 7 training should not start until all of these pass on Vast.ai:

1. `nvidia-smi` shows an RTX 3090 with about 24 GB VRAM.
2. PyTorch imports with CUDA available.
3. Ultralytics imports and the `yolo` CLI is available.
4. `/workspace/wood_defect_datacentric` exists and contains this repo.
5. Both dataset YAMLs exist under `/workspace/data/...`.
6. YOLO dataset image/label matching passes for train/val/test.
7. VNWoodKnot `knot_free` negatives are present in train/val/test and have no labels.
8. Preprocessing and augmentation previews can read real images and write scoped preview outputs.
9. Baseline VSB and VNWoodKnot dry-runs pass without missing-data errors.

## Rerun On Vast

After copying code and data to the Vast instance, run:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/server_setup_sanity.py --output-dir results/server_setup_prompt_6_5_vast --write-launcher-dry-run
PYTHONDONTWRITEBYTECODE=1 python scripts/preview_preprocessing.py --dataset vnwoodknot --split test --num-samples 2 --output-dir results/server_setup_prompt_6_5_vast/preprocessing_preview
PYTHONDONTWRITEBYTECODE=1 python scripts/preview_augmentation.py --dataset vnwoodknot --split test --num-samples 2 --output-dir results/server_setup_prompt_6_5_vast/augmentation_preview
PYTHONDONTWRITEBYTECODE=1 python scripts/run_negative_eval.py --predictions-jsonl results/server_setup_prompt_6_5_vast/synthetic_empty_predictions_test.jsonl --experiment-name prompt_6_5_synthetic_empty_negative_eval --split test --thresholds 0.10 0.25 0.50 --output-dir results/server_setup_prompt_6_5_vast/negative_eval --docs-output results/server_setup_prompt_6_5_vast/negative_eval_report.md
```

Only after those checks pass should Prompt 7 training begin.
