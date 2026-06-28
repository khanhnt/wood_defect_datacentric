# Reproduction Map

This file maps each paper artifact to the script, command, and output file used to reproduce it. Commands assume the repository root as the working directory.

The commands below do not retrain models unless explicitly marked as training. Large per-seed prediction JSON files are not stored in GitHub; place the external prediction archive under the paths described in `results/per_seed/README.md` before rerunning threshold-level analyses from raw detections.

## Core Commands

Build lightweight release artifacts from locally synced analysis results:

```bash
python scripts/build_release_artifacts.py
```

Regenerate the AP50-vs-tolerance figure:

```bash
python analysis/plot_ap50_vs_tolerance.py
```

Run release checks:

```bash
python scripts/release_integrity_check.py
```

## Paper Artifact Map

| Paper artifact | Script/source | Command | Output |
|---|---|---|---|
| Tables 1-2: dataset and split summary | `data/README.md`, released manifests | `python scripts/build_release_artifacts.py` | `data/vnwoodknot_split/manifest.jsonl`, `data/vsb_rarefirst_split/split_summary.json`, `data/vsb_clean_manifest/clean_tile_manifest.csv` |
| Table 3: data-centric variant definitions | `configs/preprocessing/*.yaml`, `configs/augmentation/*.yaml`, `configs/experiments/*.yaml` | No computation; inspect versioned YAML files | `configs/` |
| VNWoodKnot fair test mAP table | `scripts/evaluate_corrected_common.py` output bundled by release builder | `python scripts/build_release_artifacts.py` | `results/tables/vnwoodknot_fair_test_best_summary.csv` |
| VSB rare-first fair test mAP table | `scripts/evaluate_corrected_common.py` output bundled by release builder | `python scripts/build_release_artifacts.py` | `results/tables/vsb_rarefirst_fair_test_best_summary.csv` |
| VNWoodKnot false-positive rate table | `scripts/threshold_analysis.py` and corrected prediction exports | `python scripts/build_release_artifacts.py` | `results/tables/vnwoodknot_fp_bootstrap_summary.csv` |
| VNWoodKnot operational selection | `analysis/retained_metrics.py` | `python analysis/retained_metrics.py` | `results/tables/vnwoodknot_operational_selection.csv`, `results/tables/vnwoodknot_retained_metrics.json` |
| VNWoodKnot sensitivity analysis | `analysis/retained_metrics.py` | `python analysis/retained_metrics.py` | `results/tables/vnwoodknot_sensitivity.csv` |
| VNWoodKnot calibration table | `scripts/calibration_analysis.py` output bundled by release builder | `python scripts/build_release_artifacts.py` | `results/tables/vnwoodknot_calibration_summary.csv`, `results/tables/vnwoodknot_clean_max_confidence_summary.csv` |
| VSB clean operational selection | `analysis/vsb_negative_aware.py` | `python analysis/vsb_negative_aware.py --skip-inference --bootstrap-samples 10000` | `results/tables/vsb_clean_operational_selection.csv`, `results/tables/vsb_clean_retained_metrics.json` |
| VSB clean sensitivity analysis | `analysis/vsb_negative_aware.py` | `python analysis/vsb_negative_aware.py --skip-inference --bootstrap-samples 10000` | `results/tables/vsb_clean_sensitivity.csv` |
| VSB clean false-positive table | `analysis/vsb_negative_aware.py` | `python analysis/vsb_negative_aware.py --skip-inference --bootstrap-samples 10000` | `results/tables/vsb_clean_fp_bootstrap_summary.csv` |
| VSB clean calibration table | `analysis/vsb_negative_aware.py` | `python analysis/vsb_negative_aware.py --skip-inference --bootstrap-samples 10000` | `results/tables/vsb_clean_calibration_summary.csv`, `results/tables/vsb_clean_max_confidence_summary.csv` |
| Inference cost table/sentence | `analysis/inference_cost.py` | `python analysis/inference_cost.py --checkpoint /path/to/best.pt --data-yaml /path/to/dataset.yaml --device 0 --precision fp32 --imgsz 1024 --conf 0.001 --iou 0.7 --timed-images 300 --warmup 30 --output-json results/inference_cost.json` | `results/inference_cost.json` |
| Figure: dataset samples | `scripts/fig_dataset_samples.py` | `python scripts/fig_dataset_samples.py --seed 42` | `results/qualitative/dataset_samples.pdf`, `figures/dataset_samples.pdf` |
| Figure: detection scenarios | `scripts/fig_detection_scenarios.py` | `python scripts/fig_detection_scenarios.py --seed 42 --render-only` | `results/qualitative/detection_scenarios.pdf`, `figures/detection_scenarios.pdf` |
| Figure: detection performance vs threshold | `scripts/generate_plots.py` output bundled by release builder | `python scripts/build_release_artifacts.py` | `figures/detection_performance_vs_threshold.pdf` |
| Figure: false-positive behavior vs threshold | `scripts/generate_plots.py` output bundled by release builder | `python scripts/build_release_artifacts.py` | `figures/false_positive_behavior_vs_threshold.pdf` |
| Figure: operational selection trade-off | `scripts/generate_plots.py` output bundled by release builder | `python scripts/build_release_artifacts.py` | `figures/operational_selection_recall_fp_tradeoff.pdf` |
| Figure: VNWoodKnot reliability curve | `scripts/calibration_analysis.py` output bundled by release builder | `python scripts/build_release_artifacts.py` | `figures/vnwoodknot_reliability_curve.pdf` |
| Figure: clean-wood max-confidence CDF | `scripts/calibration_analysis.py`, `analysis/vsb_negative_aware.py` | `python scripts/build_release_artifacts.py` | `figures/vnwoodknot_clean_max_confidence_cdf.pdf`, `figures/vsb_clean_max_confidence_cdf.pdf` |
| Figure 9: AP50 vs tolerated clean FP rate | `analysis/plot_ap50_vs_tolerance.py` | `python analysis/plot_ap50_vs_tolerance.py` | `figures/ap50_vs_tolerance_vnwk_vsb.pdf` |

## Optional Full Training

Full training is not needed to reproduce the paper tables from released predictions. To retrain every multiseed run from scratch on a two-GPU server:

```bash
python scripts/run_all_experiments.py --batch-size 40 --gpus 0,1 --dataset all
```

This launches 36 jobs with seeds 42, 43, and 44. Runtime depends on server I/O and GPU clocks.

## Data Reconstruction

VNWoodKnot:

```bash
python data/prepare_vnwoodknot.py \
  --images-root /path/to/VNWoodKnot \
  --output-root /path/to/vnwoodknot_live_dead_2class_yolo
```

VSB rare-first:

```bash
python data/prepare_vsb.py \
  --manifest /path/to/vsb7_3600_rare_first/manifest.jsonl \
  --images-root /path/to/VSB/images \
  --output-root /path/to/vsb7_3600_rare_first_yolo
```

VSB clean wood:

```bash
python analysis/vsb_negative_aware.py \
  --prepare-only \
  --clean-images-root /path/to/VSB_defect_free_images \
  --clean-ids-file configs/datasets/vsb_clean_source_ids.txt
```
