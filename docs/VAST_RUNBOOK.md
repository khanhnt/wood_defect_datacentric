# Vast.ai Runbook

This is the exact command sequence for a fresh RTX 3090 24 GB Vast.ai instance.

## 1. Clone

```bash
cd /workspace
git clone <YOUR_GITHUB_REPO_URL> wood_defect_datacentric
cd /workspace/wood_defect_datacentric
```

## 2. Environment

```bash
conda create -n wooddc python=3.10 -y
conda activate wooddc
pip install --upgrade pip
pip install -r requirements.txt
```

If conda is unavailable:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Configure Paths

```bash
cp .env.example .env
nano .env
```

Set dataset YAML paths to:

```bash
WOOD_DC_VSB_BASELINE_DATASET_YAML=/workspace/data/main_dataset/benchmarks/vsb7_3600_rare_first_yolo/dataset.yaml
WOOD_DC_VN_BASELINE_DATASET_YAML=/workspace/data/vnwoodknot/benchmarks/vnwoodknot_live_dead_2class_yolo/dataset.yaml
RESULTS_ROOT=/workspace/wood_defect_datacentric/results
DEVICE=0
BATCH_SIZE=16
IMG_SIZE=1024
```

## 4. Copy Datasets

Use `gdown`, `rclone`, or `rsync/scp` as described in `docs/google_drive_dataset_setup.md`.

Confirm:

```bash
test -f /workspace/data/main_dataset/benchmarks/vsb7_3600_rare_first_yolo/dataset.yaml
test -f /workspace/data/vnwoodknot/benchmarks/vnwoodknot_live_dead_2class_yolo/dataset.yaml
```

## 5. Server Probe

```bash
./scripts/setup_server.sh
```

Stop if GPU, CUDA, PyTorch, Ultralytics, disk, or folder checks fail.

## 6. Dataset Readiness

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/check_server_ready.py
```

Stop if VSB/VNWoodKnot YAMLs, splits, image-label matching, class names, or VNWoodKnot negative checks fail.

## 7. Previews

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/preview_preprocessing.py \
  --dataset vnwoodknot \
  --split test \
  --num-samples 4 \
  --output-dir results/server_previews/preprocessing

PYTHONDONTWRITEBYTECODE=1 python scripts/preview_augmentation.py \
  --dataset vnwoodknot \
  --split test \
  --num-samples 4 \
  --output-dir results/server_previews/augmentation
```

Inspect preview images before running data-centric variants.

## 8. Baseline Dry-Runs

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/server_setup_sanity.py \
  --output-dir results/server_setup_prompt_6_5_vast \
  --write-launcher-dry-run
```

This safely dry-runs:

- `vsb_yolov8s_baseline_e50`
- `vn_t0_yolov8s_baseline_e50`

The dry-run metadata is kept under `results/server_setup_prompt_6_5_vast/` so it does not occupy real training run folders.

## 9. Start tmux

```bash
tmux new -s wooddc
```

## 10. Run Batch 1

```bash
./scripts/run_batch1_baselines.sh
```

Detach with `Ctrl-b d`; reattach with:

```bash
tmux attach -t wooddc
```

## 11. Aggregate After Batch 1

```bash
./scripts/aggregate_after_batch.sh
```

Review:

- `results/batch_summaries/*.csv`
- `results/batch_logs/batch1_baselines/*.log`
- `results/runs/*/validation_metrics.json`
- `results/runs/*/ultralytics/train/weights/best.pt`

Continue only if baseline metrics and logs are reasonable.

## 12. Later Batches

```bash
./scripts/run_batch2_preprocessing.sh
./scripts/aggregate_after_batch.sh

./scripts/run_batch3_augmentation.sh
./scripts/aggregate_after_batch.sh

./scripts/run_batch4_combined.sh
./scripts/aggregate_after_batch.sh
```

Do not run all batches blindly. Check metrics, logs, disk space, and checkpoint health after each batch.
