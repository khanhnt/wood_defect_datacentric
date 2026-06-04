#!/usr/bin/env bash
set -Eeuo pipefail

BATCH_NAME="batch3_augmentation"
EXPERIMENT_IDS=(
  "vsb_yolov8s_a1_crop_e50"
  "vsb_yolov8s_a2_colorjitter_e50"
  "vn_yolov8s_a1_crop_e50"
)
EXPERIMENT_CONFIGS=(
  "configs/experiments/vsb_yolov8s_a1_crop_e50.yaml"
  "configs/experiments/vsb_yolov8s_a2_colorjitter_e50.yaml"
  "configs/experiments/vn_yolov8s_a1_crop_e50.yaml"
)

# shellcheck source=scripts/batch_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/batch_common.sh"
init_batch
run_batch
