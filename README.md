# Beyond mAP: Negative-Aware Evaluation of Data-Centric Wood Knot Detection

This repository contains the code, configurations, public-release manifests, tables, and figures for the IEEE Access paper:

> Beyond mAP: Negative-Aware Evaluation of Data-Centric Pipelines for Wood Knot Detection

The primary study keeps YOLOv8s fixed while evaluating data-centric preprocessing and augmentation choices on two wood-defect datasets. A separate, minimal Faster R-CNN experiment tests whether the central conclusions persist under a second detector family. The central result is that standard mAP alone can hide operational differences that appear when false alarms on clean wood are measured explicitly.

## What Is Included

- Training and evaluation scripts for the YOLOv8s study and the Faster R-CNN robustness experiment.
- Variant configs for baseline, P1 CLAHE, P2 illumination normalization, P3 unsharp masking, A1 defect-preserving crop, A2 texture-aware colour jitter, and P4+A4 combined.
- Split and tiling manifests needed to reconstruct the benchmark datasets from the original raw datasets.
- Detector-separated CSV/JSON files behind the revised paper tables in `results/tables/`.
- Final figure PDFs in `figures/` and mirrored figure assets in `results/figures/`.
- Reproducibility commands in `REPRODUCE.md`.

Raw images, materialized datasets, checkpoints, and large per-seed prediction JSON files are not committed to GitHub.

## Datasets

The raw datasets must be obtained from their original sources:

- VNWoodKnot: Data in Brief, DOI `10.1016/j.dib.2025.112039`.
- VSB/Kodytek large-scale wood surface defects: F1000Research, DOI `10.12688/f1000research.52903`.

The VSB clean-wood set contains 1,992 defect-free source images identified by empty `*_anno.txt` files. Each source yields three 1024-pixel tiles. The actual tile origins are recorded in the released manifest because the final overlap depends on source width. See `data/README.md`.

## Environment

The paper runs used:

- Python 3.12
- PyTorch 2.6.0 with CUDA 12.4 on the original Vast.ai instance for training
- Ultralytics 8.4.60
- 2x NVIDIA RTX 3090, 24 GB VRAM each
- Seeds 42, 43, 44
- Image size 1024
- YOLOv8s batch size 40; Faster R-CNN batch size 4
- 50 epochs

Install:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For CUDA-enabled PyTorch, install the wheel matching your driver from the official PyTorch index before installing the remaining requirements.

## Quick Reproduction Without Retraining

The release package can be checked directly from the released CSV/JSON artifacts:

```bash
python scripts/release_integrity_check.py
```

The final lightweight tables are organized by detector:

- `results/tables/yolov8s/`: seven-variant, three-seed results on VNWoodKnot and VSB.
- `results/tables/fasterrcnn/`: three-variant, three-seed VNWoodKnot robustness results.
- `results/_deprecated/pre_revision_tables/`: superseded artifacts retained only for audit history.

If the large per-seed prediction archive is downloaded separately, threshold-level analyses can be rerun without retraining:

```bash
python analysis/retained_metrics.py
python analysis/vsb_negative_aware.py --skip-inference --bootstrap-samples 10000
```

See `REPRODUCE.md` for the full paper-artifact map.

## Training From Scratch

Full multiseed training is optional and GPU-intensive. The final YOLOv8s generation used the corrected 24-job queue after 18 unaffected checkpoints had been staged from the prior generation:

```bash
python scripts/run_all_experiments.py \
  --job-set corrected24 --dataset all --batch-size 40 \
  --epochs 50 --imgsz 1024 --gpus 0,1 \
  --rebuilt-root /path/to/datasets_rebuilt \
  --results-root /path/to/generation
```

Together with the 18 unaffected checkpoints, this produces the complete 42-run YOLOv8s generation: 21 VNWoodKnot jobs and 21 VSB rare-first jobs across seeds 42, 43, and 44. The launcher pins each job to one GPU with `CUDA_VISIBLE_DEVICES`.

The second-detector experiment is launched separately:

```bash
python scripts/run_fasterrcnn_experiments.py \
  --rebuilt-root /path/to/datasets_frcnn_rebuilt \
  --results-root /path/to/fasterrcnn_generation \
  --gpus 0,1 --batch-size 4 --epochs 50 --workers 4
```

It runs Baseline, A1 crop, and A2 colour jitter on VNWoodKnot for seeds 42, 43, and 44. Absolute metrics are compared only within each detector block.

## Main Analysis Entry Points

- `analysis/retained_metrics.py`: retained recall/AP50 at zero-FP operating points.
- `analysis/vsb_negative_aware.py`: VSB clean-wood negative-aware analysis.
- `analysis/inference_cost.py`: latency, model size, and preprocessing overhead.
- `analysis/plot_ap50_vs_tolerance.py`: AP50-vs-FP-tolerance figure.
- `analysis/fasterrcnn_negative_aware.py`: Faster R-CNN validation-selected negative-aware evaluation.
- `scripts/finalize_fasterrcnn_results.py`: Faster R-CNN standard-metric aggregation and artifact audit.
- `scripts/generate_revision_figures.py`: final revision figures from the frozen YOLOv8s generation.
- `scripts/threshold_analysis.py`: shared threshold-sweep and AP calculations.
- `scripts/evaluate_corrected_common.py`: fair common-evaluation mapping.

## Integrity Check

Run before release:

```bash
python scripts/release_integrity_check.py
```

The check verifies result cardinalities, key manuscript values for both detector blocks, VSB split and clean-set counts, source-disjoint clean partitions, checksums for the packaged tables, the deprecated 6,252-denominator note, and absence of personal absolute paths or credential-like strings in release files.

## Citation

Please cite the paper and this repository. A repository citation template is provided in `CITATION.cff`.
