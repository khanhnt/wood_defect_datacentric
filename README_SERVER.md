# Vast.ai Server README

This repository is the independent IJACSA data-centric wood surface defect project. It keeps the detector fixed as YOLOv8s and studies preprocessing, augmentation, and negative-aware VNWoodKnot evaluation. It is intentionally separate from the accepted CMC detector-comparison repository.

Do not commit datasets, weights, checkpoints, archives, or generated results to GitHub.

## 1. Clone On Vast.ai

```bash
cd /workspace
git clone <YOUR_GITHUB_REPO_URL> wood_defect_datacentric
cd /workspace/wood_defect_datacentric
```

## 2. Create Environment

Use either conda:

```bash
conda create -n wooddc python=3.10 -y
conda activate wooddc
pip install -r requirements.txt
```

Or venv:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 3. Configure `.env`

```bash
cp .env.example .env
nano .env
```

Set at least:

- `PROJECT_ROOT=/workspace/wood_defect_datacentric`
- `DATA_ROOT=/workspace/data`
- `VSB_ROOT=/workspace/data/main_dataset/benchmarks/vsb7_3600_rare_first_yolo`
- `VNWOODKNOT_ROOT=/workspace/data/vnwoodknot/benchmarks/vnwoodknot_live_dead_2class_yolo`
- `WOOD_DC_VSB_BASELINE_DATASET_YAML=/workspace/data/main_dataset/benchmarks/vsb7_3600_rare_first_yolo/dataset.yaml`
- `WOOD_DC_VN_BASELINE_DATASET_YAML=/workspace/data/vnwoodknot/benchmarks/vnwoodknot_live_dead_2class_yolo/dataset.yaml`
- `RESULTS_ROOT=/workspace/wood_defect_datacentric/results`
- `DEVICE=0`
- `BATCH_SIZE=16`
- `IMG_SIZE=1024`

## 4. Copy Datasets From Google Drive

Use one of the methods in `docs/google_drive_dataset_setup.md`: `gdown`, `rclone`, or `rsync/scp` from a local machine that already has the files.

Expected baseline YOLO dataset YAMLs:

```text
/workspace/data/main_dataset/benchmarks/vsb7_3600_rare_first_yolo/dataset.yaml
/workspace/data/vnwoodknot/benchmarks/vnwoodknot_live_dead_2class_yolo/dataset.yaml
```

## 5. Verify Server Environment

```bash
./scripts/setup_server.sh
```

This checks GPU visibility, PyTorch CUDA, Ultralytics, disk space, and required folders. It does not start training.

## 6. Generate YOLO Datasets From Manifests If Needed

If the copied data contains only `images/`, `manifest.jsonl`, and `metadata.json`, generate YOLO folders before running training.

VNWoodKnot usually has manifest split labels:

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

For VSB, prefer an existing curated split. If the manifest has no split labels and you need a server-ready YOLO folder, create a deterministic random split:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/materialize_yolo_from_manifest.py \
  --manifest /workspace/data/main_dataset/manifest.jsonl \
  --images-root /workspace/data/main_dataset/images \
  --output-root /workspace/data/main_dataset/benchmarks/vsb7_3600_rare_first_yolo \
  --dataset-name vsb7_3600_rare_first_yolo \
  --classes live_knot dead_knot resin knot_with_crack crack marrow knot_missing \
  --split-strategy random \
  --seed 42 \
  --link-mode symlink
```

Review `materialization_report.json` before training. Records with unknown classes or invalid boxes are skipped by default to avoid creating false-negative labels.

## 7. Verify Datasets

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/check_server_ready.py
```

This checks:

- VSB and VNWoodKnot dataset YAML paths.
- Train/val/test split paths.
- Image/label matching.
- Label formatting and class IDs.
- Expected class names.
- VNWoodKnot empty/background labels and manifest `knot_free` retention.
- Results folder writability.

## 8. Preview Data-Centric Transforms

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

Review the generated panels before training data-centric variants.

## 9. Run Baseline Dry-Runs

Run the server sanity helper so dry-run metadata is kept away from real training run folders:

```bash
PYTHONDONTWRITEBYTECODE=1 python scripts/server_setup_sanity.py \
  --output-dir results/server_setup_sanity_vast \
  --write-launcher-dry-run
```

Both baseline dry-runs must be `ok=true` before Batch 1.

## 10. Run Batch 1 Baselines

Use tmux so training survives SSH disconnects:

```bash
tmux new -s wooddc
./scripts/run_batch1_baselines.sh
```

Detach with `Ctrl-b d`. Reattach with:

```bash
tmux attach -t wooddc
```

## 11. Aggregate After Batch 1

```bash
./scripts/aggregate_after_batch.sh
```

Review baseline mAP, precision, recall, logs, and checkpoint files. Continue only if baselines are reasonable and no dataset/path issue appears.

## 12. Continue Controlled Batches

Run later batches only after confirming Batch 1:

```bash
./scripts/run_batch2_preprocessing.sh
./scripts/aggregate_after_batch.sh

./scripts/run_batch3_augmentation.sh
./scripts/aggregate_after_batch.sh

./scripts/run_batch4_combined.sh
./scripts/aggregate_after_batch.sh
```

Optional copy-paste remains experimental and is not included in the default batch sequence.
