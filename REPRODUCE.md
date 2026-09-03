# Reproduction Map

This document maps the revised manuscript artifacts to released files and replay commands. Commands assume the repository root as the working directory. Training is optional; the lightweight table package is sufficient to inspect every reported numerical result.

## 1. Verify the Release

```bash
python scripts/release_integrity_check.py
python -m unittest discover -s tests -p 'test_*.py'
```

The first command checks artifact cardinalities, key manuscript values, clean-set provenance, packaged checksums, and release-path hygiene.

## 2. Released Result Layout

| Result block | Primary files |
|---|---|
| YOLOv8s standard metrics | `results/tables/yolov8s/fair_metrics_per_seed.csv`, `results/tables/yolov8s/fair_metrics_summary.csv` |
| YOLOv8s validation threshold selection | `results/tables/yolov8s/validation_threshold_selection.csv` |
| YOLOv8s locked test operating points | `results/tables/yolov8s/locked_test_operating_points_per_seed.csv`, `results/tables/yolov8s/locked_test_operating_points_summary.csv` |
| YOLOv8s tolerance analysis | `results/tables/yolov8s/locked_test_sensitivity_summary.csv` |
| YOLOv8s clean false alarms | `results/tables/yolov8s/clean_fp_sweep_per_seed.csv`, `results/tables/yolov8s/clean_fp_sweep_summary.csv` |
| YOLOv8s calibration and clean confidence | `results/tables/yolov8s/calibration_*.csv`, `results/tables/yolov8s/clean_max_confidence_*.csv`, `results/tables/yolov8s/reliability_bins.csv` |
| Reviewer-requested YOLOv8s audits | `results/tables/yolov8s/reviewer_audits/` |
| Faster R-CNN standard metrics | `results/tables/fasterrcnn/standard/per_seed_metrics.csv`, `results/tables/fasterrcnn/standard/summary.csv` |
| Faster R-CNN negative-aware results | `results/tables/fasterrcnn/negative_aware/test_operating_metrics_per_seed.csv`, `results/tables/fasterrcnn/negative_aware/test_operating_summary.csv` |
| Faster R-CNN threshold stability | `results/tables/fasterrcnn/negative_aware/zero_fp_binding_audit.csv`, `results/tables/fasterrcnn/negative_aware/epsilon_rank_audit.csv`, `results/tables/fasterrcnn/negative_aware/operational_stability_audit.json` |
| Faster R-CNN provenance | `results/tables/fasterrcnn/provenance/` |

Pre-revision flat table files are preserved under `results/_deprecated/pre_revision_tables/` for audit history. They are not inputs to the revised manuscript.

## 3. Paper Artifact Map

| Paper content | Source |
|---|---|
| Dataset and split counts | `data/README.md`, `data/vnwoodknot_split/manifest.jsonl`, `data/vsb_rarefirst_split/manifest.jsonl`, `data/vsb_clean_manifest/clean_tile_manifest.csv` |
| Data-centric variant definitions | `configs/preprocessing/*.yaml`, `configs/augmentation/*.yaml`, `configs/experiments/*.yaml` |
| YOLOv8s fair validation/test tables | `results/tables/yolov8s/fair_metrics_summary.csv` |
| YOLOv8s false-alarm, operating-point, and sensitivity tables | `results/tables/yolov8s/clean_fp_sweep_summary.csv`, `results/tables/yolov8s/locked_test_operating_points_summary.csv`, `results/tables/yolov8s/locked_test_sensitivity_summary.csv` |
| YOLOv8s calibration tables | `results/tables/yolov8s/calibration_summary.csv`, `results/tables/yolov8s/clean_max_confidence_summary.csv` |
| Faster R-CNN robustness tables | `results/tables/fasterrcnn/standard/summary.csv`, `results/tables/fasterrcnn/negative_aware/test_operating_summary.csv` |
| Final quantitative figures | `scripts/generate_revision_figures.py`; released copies in `figures/` and `results/figures/` |
| Dataset samples and detection scenarios | `scripts/fig_dataset_samples.py`, `scripts/fig_detection_scenarios.py` |
| Inference cost | `analysis/inference_cost.py` |

LaTeX-ready row fragments used for direct cross-checking are under `results/tables/yolov8s/latex/`.

## 4. Rebuild the Lightweight Table Package

Maintainers with the two frozen generations restored locally can rebuild the public package without GPU inference:

```bash
python scripts/build_release_artifacts.py \
  --yolo-analysis-dir /path/to/access_r1_g2_analysis \
  --fasterrcnn-generation-dir /path/to/access_r1_g3_fasterrcnn \
  --output-root results/tables
```

This operation copies only final aggregate and audit artifacts, separates detector families, and regenerates `results/tables/SHA256SUMS`.

## 5. Replay Analysis From Prediction Exports

Large prediction JSON files are not stored in GitHub. Restore the external frozen-generation archive before running these commands.

YOLOv8s retained metrics and negative-aware analysis:

```bash
python analysis/retained_metrics.py
python analysis/vsb_negative_aware.py --skip-inference --bootstrap-samples 10000
```

Faster R-CNN standard aggregation:

```bash
python scripts/finalize_fasterrcnn_results.py \
  --rebuilt-root /path/to/datasets_frcnn_rebuilt \
  --results-root /path/to/fasterrcnn_generation \
  --predictions-root /path/to/fasterrcnn_generation/predictions/vnwoodknot \
  --output-dir /path/to/fasterrcnn_generation/fasterrcnn/analysis
```

Use `python analysis/fasterrcnn_negative_aware.py --help` to bind the negative-aware analysis to the restored generation and dataset paths; no retraining is required.

## 6. Rebuild Datasets

Obtain the raw public datasets first. Then materialize the canonical trees with split-consistent path resolution:

```bash
python scripts/materialize_yolo_from_manifest.py \
  --manifest data/vnwoodknot_split/manifest.jsonl \
  --images-root /path/to/VNWoodKnot/images \
  --output-root /path/to/datasets_rebuilt/canonical/vnwoodknot \
  --dataset-name vnwoodknot \
  --classes live_knot dead_knot \
  --split-strategy manifest \
  --link-mode copy \
  --exclude-image-id train/2/img_3671

python scripts/materialize_yolo_from_manifest.py \
  --manifest data/vsb_rarefirst_split/manifest.jsonl \
  --images-root /path/to/VSB/tiles \
  --output-root /path/to/datasets_rebuilt/canonical/vsb_rarefirst \
  --dataset-name vsb_rarefirst \
  --classes live_knot dead_knot resin knot_with_crack crack marrow knot_missing \
  --split-strategy manifest \
  --link-mode copy
```

The accepted canonical counts are VNWoodKnot `1059/226/229` and VSB rare-first `7679/977/972` for train/validation/test. Rebuild the VSB strict-clean set from the 1,992 defect-free source images with `analysis/vsb_negative_aware.py --prepare-only`; the released clean-tile manifest records all 5,976 actual tile origins.

## 7. Optional Training

The final YOLOv8s generation contains 42 runs. Eighteen unaffected checkpoints were retained and 24 runs were executed with the corrected train-only augmentation protocol:

```bash
python scripts/run_all_experiments.py \
  --job-set corrected24 --dataset all --batch-size 40 \
  --epochs 50 --imgsz 1024 --gpus 0,1 \
  --rebuilt-root /path/to/datasets_rebuilt \
  --results-root /path/to/yolov8s_generation
```

The second-detector robustness block contains nine VNWoodKnot runs:

```bash
python scripts/run_fasterrcnn_experiments.py \
  --rebuilt-root /path/to/datasets_frcnn_rebuilt \
  --results-root /path/to/fasterrcnn_generation \
  --gpus 0,1 --batch-size 4 --epochs 50 --workers 4
```

All runs use seeds 42, 43, and 44. YOLOv8s and Faster R-CNN are separate experimental blocks; absolute metrics are not compared across them.
