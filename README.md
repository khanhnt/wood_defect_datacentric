# Wood Defect Data-Centric Experiments

Independent experiment repository for the wood surface defect detection study.

The accepted CMC paper focused on detector-family comparison, protocol breadth, and source-to-target transfer. This repository intentionally uses a different research question: keep the detector fixed as YOLOv8s, then evaluate whether data-centric changes improve defect detection and reduce false positives on negative wood images.

## Research Focus

- Fixed detector: YOLOv8s.
- Data-centric variants: defect-preserving preprocessing and augmentation.
- Negative-aware evaluation: VNWoodKnot `knot_free` false positives and threshold sensitivity.
- Datasets: VSB curated benchmark and VNWoodKnot, reused with the existing train/val/test splits.

## Structure

```text
configs/          Experiment, preprocessing, augmentation, and project configs
data/processed/   Small manifest metadata copied for independent audits
datasets/         Dataset adapters and statistics helpers
preprocessing/    Safe image preprocessing variants
augmentation/     Defect-preserving augmentation variants
evaluation/       Negative-aware object detection evaluation
scripts/          CLI entry points and dry-run launchers
results/          Generated outputs; ignored by git except .gitkeep
docs/             Audit notes, variant notes, and handoff context
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

The `.env` file is optional for local audit scripts, but useful on Vast.ai for overriding YOLO `dataset.yaml` paths.

## Smoke Test

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/smoke_test.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/smoke_test.py --strict-optional
```

The default smoke test checks core imports and expected folders. `--strict-optional` also enforces CV/evaluation dependencies after `pip install -r requirements.txt`. It does not train models.

## Dataset Audit

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/dataset_stats.py
PYTHONDONTWRITEBYTECODE=1 python3 scripts/verify_splits.py
```

The audit scripts are read-only. They preserve configured splits, retain negative/background-only images, and write reports under `results/` and `docs/`.

## Preview Preprocessing and Augmentation

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/preview_preprocessing.py --dataset vnwoodknot --split test --num-samples 4
PYTHONDONTWRITEBYTECODE=1 python3 scripts/preview_augmentation.py --dataset vnwoodknot --split test --num-samples 4
```

These previews do not alter original datasets. They write visual samples under `results/preprocessing_preview/` and `results/augmentation_preview/`.

## YOLOv8s Training Pipeline

Experiment configs live in `configs/experiments/`; the run list is `configs/experiment_matrix.csv`. The launcher is safe by default and only trains when `--execute` is explicitly provided.

Dry-run all experiments, including optional copy-paste:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/launch_yolo_experiment.py   --matrix configs/experiment_matrix.csv   --dry-run   --include-optional
```

Dry-run one experiment strictly:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/launch_yolo_experiment.py   --matrix configs/experiment_matrix.csv   --experiment-id vn_t0_yolov8s_baseline_e50   --dry-run   --strict
```

Run one experiment only after the server sanity check passes and the YOLO `dataset.yaml` exists:

```bash
python3 scripts/launch_yolo_experiment.py   --experiment-config configs/experiments/vn_t0_yolov8s_baseline_e50.yaml   --execute
```

Run folders are written to `results/runs/<experiment_id>/` and store the resolved config, command, training log, validation status, validation metrics placeholder or parsed metrics, and expected checkpoint path.

## Phase 1 Multi-GPU Multiseed Runs

The Vast.ai two-GPU workflow for batch-size probing and the 36-job multiseed
queue is documented in `docs/PHASE1_GPU_OPTIMIZATION_RUNBOOK.md`.

Key entry points:

```bash
python scripts/batch_size_test.py
python scripts/run_all_experiments.py --batch-size 32 --gpus 0,1 --dry-run
bash scripts/setup_fresh_run.sh --timestamp
```

## Negative-Aware Evaluation

Run threshold-sensitive VNWoodKnot evaluation from an existing prediction JSONL:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 scripts/run_negative_eval.py   --predictions-jsonl /path/to/predictions.jsonl   --experiment-name vn_t0_negative_eval   --split test
```

Run from a YOLO checkpoint when predictions are not already exported:

```bash
python3 scripts/run_negative_eval.py   --checkpoint results/runs/vn_t0_yolov8s_baseline_e50/ultralytics/train/weights/best.pt   --experiment-name vn_t0_yolov8s_baseline_e50_negative_eval   --split test   --save-predictions
```

Outputs are written under `results/negative_eval/` and summarized in `docs/negative_aware_evaluation.md`.

## Next Step

Next planned step is Prompt 6.5: set up and verify the Vast.ai server with an RTX 3090 24GB instance. Do not start full training until server setup, data paths, CUDA, YOLO import, and dry-run validation pass.
