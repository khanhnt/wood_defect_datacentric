#!/usr/bin/env bash
set -Eeuo pipefail

BATCH_NAME="batch2_preprocessing"
EXPERIMENT_IDS=(
  "vsb_yolov8s_p1_clahe_e50"
  "vsb_yolov8s_p2_illumination_e50"
  "vsb_yolov8s_p3_unsharp_e50"
  "vn_yolov8s_p2_illumination_e50"
)
EXPERIMENT_CONFIGS=(
  "configs/experiments/vsb_yolov8s_p1_clahe_e50.yaml"
  "configs/experiments/vsb_yolov8s_p2_illumination_e50.yaml"
  "configs/experiments/vsb_yolov8s_p3_unsharp_e50.yaml"
  "configs/experiments/vn_yolov8s_p2_illumination_e50.yaml"
)

# shellcheck source=scripts/batch_common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/batch_common.sh"
init_batch
run_batch
