# Phase 1 GPU Optimization and Parallel Launcher Runbook

This runbook is for the Vast.ai server. Do not run the training commands on the
local Codex development machine.

## Scope

Phase 1 adds infrastructure for the multiseed fixed-YOLOv8s study:

- Batch-size probe on VNWoodKnot baseline.
- Fresh results directory setup.
- Two-GPU isolated job queue.
- Seeds: 42, 43, 44.
- Image size: 1024.
- Epochs: 50.
- Fixed detector: YOLOv8s.

The launcher uses `CUDA_VISIBLE_DEVICES` for single-GPU isolation. It does not
use DataParallel or DistributedDataParallel.

## Job Queue

The full queue contains 36 jobs:

- VNWoodKnot: 5 variants x 3 seeds = 15 jobs.
- VSB rare-first: 7 variants x 3 seeds = 21 jobs.

VNWoodKnot variants:

- `baseline`
- `p2_illumination`
- `a1_crop`
- `a2_colorjitter`
- `p4_a4_combined`

VSB rare-first variants:

- `baseline`
- `p1_clahe`
- `p2_illumination`
- `p3_unsharp`
- `a1_crop`
- `a2_colorjitter`
- `p4_a4_combined`

Priority order is VNWoodKnot first, then VSB rare-first.

## Vast Server Commands

Run these from the repo root on Vast.ai:

```bash
cd /workspace/wood_defect_datacentric
git pull
source .venv/bin/activate
```

Archive the previous `results/` directory and create the fresh Phase 1 layout:

```bash
bash scripts/setup_fresh_run.sh --timestamp
```

Run the batch-size probe on GPU 0:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/batch_size_test.py
cat results/gpu_optimization/batch_size_test.csv
```

Pick the largest successful batch whose peak VRAM is at most 22 GB. If the
recommendation is `32`, use `--batch-size 32` below. If it recommends a smaller
value, use that value instead.

Dry-run the full queue:

```bash
python scripts/run_all_experiments.py \
  --batch-size 32 \
  --gpus 0,1 \
  --dataset all \
  --dry-run
```

Launch the real run inside `tmux` or `screen`:

```bash
tmux new -s wood_phase1
python scripts/run_all_experiments.py \
  --batch-size 32 \
  --gpus 0,1 \
  --dataset all
```

If interrupted, rerun with `--resume`:

```bash
python scripts/run_all_experiments.py \
  --batch-size 32 \
  --gpus 0,1 \
  --dataset all \
  --resume
```

Run only VNWoodKnot or only VSB if needed:

```bash
python scripts/run_all_experiments.py --batch-size 32 --gpus 0,1 --dataset vnwoodknot
python scripts/run_all_experiments.py --batch-size 32 --gpus 0,1 --dataset vsb
```

## Outputs

Batch-size probe:

```text
results/gpu_optimization/batch_size_test.csv
results/gpu_optimization/batch_size_24_probe.log
results/gpu_optimization/batch_size_32_probe.log
results/gpu_optimization/batch_size_40_probe.log
```

Queue logs:

```text
results/gpu_optimization/run_log.csv
results/gpu_optimization/job_logs/
results/gpu_optimization/generated_configs/
results/gpu_optimization/materialization_logs/
```

Training outputs:

```text
results/multiseed/vnwoodknot/per_seed/runs/<variant>_seed<seed>/
results/multiseed/vsb_rarefirst/per_seed/runs/<variant>_seed<seed>/
```

Best checkpoint paths:

```text
results/multiseed/vnwoodknot/per_seed/runs/<variant>_seed<seed>/ultralytics/train/weights/best.pt
results/multiseed/vsb_rarefirst/per_seed/runs/<variant>_seed<seed>/ultralytics/train/weights/best.pt
```

## Notes

- The runner retries a failed/OOM job once using `batch_size - 8`.
- Baseline jobs use the existing YOLO dataset YAMLs.
- Preprocessing materialized datasets are shared across seeds.
- Augmentation materialized datasets are seed-specific.
- A3 copy-paste is excluded from the main 36-job queue.
