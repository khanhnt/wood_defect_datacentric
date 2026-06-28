# Beyond mAP: Negative-Aware Evaluation of Data-Centric Wood Knot Detection

This repository contains the code, configurations, public-release manifests, tables, and figures for the IEEE Access paper:

> Beyond mAP: Negative-Aware Evaluation of Data-Centric Pipelines for Wood Knot Detection

The study keeps the detector fixed as YOLOv8s and evaluates data-centric preprocessing and augmentation choices on two wood-defect datasets. The central result is that standard mAP alone hides operational differences that appear when false alarms on clean wood are measured explicitly.

## What Is Included

- Training and evaluation scripts for YOLOv8s experiments.
- Variant configs for baseline, P1 CLAHE, P2 illumination normalization, P3 unsharp masking, A1 defect-preserving crop, A2 texture-aware colour jitter, and P4+A4 combined.
- Split and tiling manifests needed to reconstruct the benchmark datasets from the original raw datasets.
- CSV/JSON files behind the paper tables in `results/tables/`.
- Final figure PDFs in `figures/` and mirrored figure assets in `results/figures/`.
- Reproducibility commands in `REPRODUCE.md`.

Raw images, YOLO materialized datasets, checkpoints, and large per-seed prediction JSON files are not committed to GitHub.

## Datasets

The raw datasets must be obtained from their original sources:

- VNWoodKnot: Data in Brief, DOI `10.1016/j.dib.2025.112039`.
- VSB/Kodytek large-scale wood surface defects: F1000Research, DOI `10.12688/f1000research.52903.x`.

The VSB clean-wood set contains 1,992 defect-free source images identified by empty `*_anno.txt` files. These are tiled at 1024 px with 128 px overlap into 5,976 clean tiles. See `data/README.md`.

## Environment

The paper runs used:

- Python 3.12
- PyTorch 2.6.0 with CUDA 12.4 on the original Vast.ai instance for training
- Ultralytics 8.4.60
- 2x NVIDIA RTX 3090, 24 GB VRAM each
- Seeds 42, 43, 44
- Image size 1024
- Batch size 40
- 50 epochs

Install:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA-enabled PyTorch, install the wheel matching your driver from the official PyTorch index before installing the remaining requirements.

## Quick Reproduction Without Retraining

The release tables can be regenerated from the released CSV/JSON artifacts:

```bash
python analysis/plot_ap50_vs_tolerance.py
python scripts/release_integrity_check.py
```

If the large per-seed prediction JSON archive is downloaded separately, the corrected negative-aware analyses can be rerun without retraining:

```bash
python analysis/retained_metrics.py
python analysis/vsb_negative_aware.py --skip-inference --bootstrap-samples 10000
```

See `REPRODUCE.md` for the full paper-artifact map.

## Training From Scratch

Full multiseed training is optional and GPU-intensive. The original experiments used the queue launcher:

```bash
python scripts/run_all_experiments.py --batch-size 40 --gpus 0,1 --dataset all
```

This runs 36 jobs: 15 VNWoodKnot jobs and 21 VSB rare-first jobs across seeds 42, 43, and 44. The launcher pins each job to one GPU with `CUDA_VISIBLE_DEVICES`.

## Main Analysis Entry Points

- `analysis/retained_metrics.py`: retained recall/AP50 at zero-FP operating points.
- `analysis/vsb_negative_aware.py`: VSB clean-wood negative-aware analysis.
- `analysis/inference_cost.py`: latency, model size, and preprocessing overhead.
- `analysis/plot_ap50_vs_tolerance.py`: AP50-vs-FP-tolerance figure.
- `scripts/threshold_analysis.py`: shared threshold-sweep and AP calculations.
- `scripts/evaluate_corrected_common.py`: fair common-evaluation mapping.

## Integrity Check

Run before release:

```bash
python scripts/release_integrity_check.py
```

The check verifies paper-table values within rounding, strict VSB clean denominator 5,976, zero VSB clean leakage, the deprecated 6,252-denominator note, and absence of personal absolute paths or credential-like strings in tracked release files.

## Citation

Please cite the paper and this repository. A repository citation template is provided in `CITATION.cff`.
