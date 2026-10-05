# Beyond mAP: Negative-Aware Evaluation of Data-Centric Pipelines for Wood Knot Detection

This repository accompanies a study of data-centric wood-knot detection under both standard and negative-aware evaluation. The primary experiments keep YOLOv8s fixed while varying preprocessing and augmentation across VNWoodKnot and VSB; a separate Faster R-CNN block tests the main conclusions with a two-stage detector. Thresholds are selected only on validation clean material and then applied unchanged to held-out test data.

## Contents

- Training, evaluation, and analysis code for YOLOv8s and Faster R-CNN.
- Configurations for Baseline, P1 CLAHE, P2 illumination normalization, P3 unsharp masking, A1 defect-preserving crop, A2 texture-aware colour jitter, and P4+A4 combined.
- Frozen split and tiling manifests needed to reconstruct the benchmark datasets.
- Per-seed and aggregate numerical artifacts in `results/tables/`.
- Publication figures in `figures/` and `results/figures/`.
- Reproduction commands in `REPRODUCE.md` and a release-claim inventory in `docs/RELEASE_AUDIT.md`.

Raw images are not redistributed.

## Environment

The reported runs used Python 3.12, PyTorch 2.6.0 with CUDA 12.4, Ultralytics 8.4.60, and OpenCV 4.10.0. Multiseed training used two NVIDIA RTX 3090 GPUs, seeds 42/43/44, image size 1024, and 50 epochs.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Install the CUDA-enabled PyTorch build appropriate for the host before installing the remaining pinned dependencies.

## Datasets

Download the raw data from the original public sources:

- [VNWoodKnot, Mendeley Data](https://doi.org/10.17632/vnst548g5n.1)
- [VSB wood surface defects, Zenodo](https://doi.org/10.5281/zenodo.4694695)

The released manifests define the exact image-level and tile-level assignments. VNWoodKnot contains 1,060/226/229 manifest records for train/validation/test; the explicitly recorded duplicate-stem exclusion leaves 1,059 materialized training images. The VSB strict-clean set contains 1,992 defect-free source boards, three recorded tiles per board, and a source-disjoint 996/996 validation/test partition.

## Verify The Lightweight Release

No GPU is required to validate the checked-in tables, manifests, and provenance summaries:

```bash
python scripts/release_integrity_check.py
python -m unittest discover -s tests -p 'test_*.py'
```

The authoritative table package is detector-separated:

- `results/tables/yolov8s/`: seven variants, two datasets, three seeds.
- `results/tables/fasterrcnn/`: Baseline, A1, and A2 on VNWoodKnot, three seeds.
- `results/_deprecated/superseded_tables/`: superseded numerical artifacts retained only for audit history.

## Paper Item Map

| Item | Script or configuration | Released output |
|---|---|---|
| T1 Datasets | `scripts/dataset_stats.py`, `scripts/release_integrity_check.py` | `data/*_split/`, `data/vsb_clean_manifest/` |
| T2 Variant configuration | `configs/preprocessing/`, `configs/augmentation/` | YAML files in those directories |
| T3 Component identifiers | `configs/experiments/`, `configs/second_detector/` | YAML files in those directories |
| T4 Held-out mAP | `scripts/evaluate_corrected_common.py`, `scripts/finalize_fasterrcnn_results.py` | `results/tables/yolov8s/fair_metrics_summary.csv`, `results/tables/fasterrcnn/standard/summary.csv` |
| T5 Calibration and clean-wood behaviour | `scripts/calibration_analysis.py`, `analysis/fasterrcnn_negative_aware.py` | `results/tables/yolov8s/calibration_summary.csv`, `results/tables/yolov8s/clean_max_confidence_summary.csv`, `results/tables/fasterrcnn/negative_aware/` |
| T6 VNWoodKnot FP rate and exact intervals | `analysis/offline_audit_analysis.py` | `results/tables/yolov8s/robustness_audits/T1_exact_binomial_intervals.csv` |
| T7 Strict operational selection | `analysis/analyze_generation.py`, `analysis/fasterrcnn_negative_aware.py` | `results/tables/yolov8s/locked_test_operating_points_summary.csv`, `results/tables/fasterrcnn/negative_aware/test_operating_summary.csv` |
| T8 Tolerance sensitivity | same analysis entry points as T7 | `results/tables/yolov8s/locked_test_sensitivity_summary.csv`, `results/tables/fasterrcnn/negative_aware/test_operating_summary.csv` |
| T9 Validation-leakage effect | `scripts/compare_deprecated_checkpoints.py`, `analysis/offline_audit_analysis.py` | `results/tables/yolov8s/deprecated_vs_corrected_summary.csv`, `results/tables/yolov8s/robustness_audits/T8_validation_selection_bias.csv` |
| A1 Per-seed strict selection | `analysis/analyze_generation.py` | `results/tables/yolov8s/locked_test_operating_points_per_seed.csv` |
| A2 Per-seed FP counts | `analysis/offline_audit_analysis.py` | `results/tables/yolov8s/robustness_audits/T2_fp_rate_and_fppi_per_seed.csv` |
| A3 Faster R-CNN negative-aware points | `analysis/fasterrcnn_negative_aware.py` | `results/tables/fasterrcnn/negative_aware/test_operating_metrics_per_seed.csv` |
| F1 Dataset samples | `scripts/fig_dataset_samples.py` | `figures/dataset_samples.pdf` |
| F2 AP50/precision/recall vs threshold | `scripts/generate_manuscript_figures.py` | `figures/detection_performance_vs_threshold.pdf` |
| F3 Clean max-confidence CDF and reliability | `scripts/generate_manuscript_figures.py` | `figures/clean_max_confidence_cdf.pdf`, `figures/reliability_curve.pdf` |
| F4 Clean boards flagged vs threshold | `scripts/generate_manuscript_figures.py` | `figures/false_positive_behavior_vs_threshold.pdf` |
| F5 Single-board seed brittleness | no dedicated released figure located | tracked as TODO in `docs/RELEASE_AUDIT.md` |
| F6 Threshold vs clean-set size | `scripts/generate_manuscript_figures.py` | `figures/threshold_vs_cleanset_size.pdf` |
| A1 Retained recall vs threshold | `scripts/generate_manuscript_figures.py` | `figures/operational_selection_recall_fp_tradeoff.pdf` |
| A2 Validation mAP50 per epoch | `scripts/generate_manuscript_figures.py` | `figures/convergence_map50.pdf` |

## Rebuild And Train

Dataset materialization uses split-consistent resolution. Augmentation variants modify `train` only; preprocessing variants are applied to every split. See `REPRODUCE.md` for exact commands and the verification gate.

```bash
python scripts/run_all_experiments.py \
  --job-set corrected24 --dataset all --batch-size 40 \
  --epochs 50 --imgsz 1024 --gpus 0,1 \
  --rebuilt-root /path/to/datasets_rebuilt \
  --results-root /path/to/generation

python scripts/run_fasterrcnn_experiments.py \
  --rebuilt-root /path/to/datasets_frcnn_rebuilt \
  --results-root /path/to/fasterrcnn_generation \
  --gpus 0,1 --batch-size 4 --epochs 50 --workers 4
```

## Large Artifacts

Git contains the lightweight tables and provenance summaries, not the checkpoint and per-seed prediction payloads. The external release archive will contain primary and deprecated YOLOv8s checkpoints, Faster R-CNN checkpoints, per-seed prediction exports used by the analyses, frozen provenance records, and a SHA-256 manifest.

The archive will be deposited on Zenodo: `ZENODO_DOI_TBD`. Until that DOI is assigned, the checked-in audit identifies these items as `NOT IN GIT`.

Maintainers can prepare the versioned directory without uploading it:

```bash
python scripts/package_release_archive.py \
  --version discover-ai-v1 \
  --yolo-generation /path/to/yolo_generation \
  --fasterrcnn-generation /path/to/fasterrcnn_generation \
  --output-root /path/to/release_archives
```

## Citation

Citation metadata and the current manuscript status are in [`CITATION.cff`](CITATION.cff). The manuscript has been submitted to *Discover Artificial Intelligence*.
