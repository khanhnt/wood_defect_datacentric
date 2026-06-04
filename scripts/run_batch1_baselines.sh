#!/usr/bin/env bash
set -Eeuo pipefail

BATCH_NAME="batch1_baselines"
EXPERIMENT_IDS=(
  "vsb_yolov8s_baseline_e50"
  "vn_t0_yolov8s_baseline_e50"
)
EXPERIMENT_CONFIGS=(
  "configs/experiments/vsb_yolov8s_baseline_e50.yaml"
  "configs/experiments/vn_t0_yolov8s_baseline_e50.yaml"
)

# shellcheck source=scripts/batch_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/batch_common.sh"
init_batch
run_batch
