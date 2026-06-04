#!/usr/bin/env bash

init_batch() {
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
  cd "${PROJECT_ROOT}"

  if [[ -f ".env" ]]; then
    set -a
    # shellcheck disable=SC1091
    source ".env"
    set +a
  fi

  RESULTS_ROOT="${RESULTS_ROOT:-${PROJECT_ROOT}/results}"
  RUNS_ROOT="${PROJECT_ROOT}/results/runs"
  LOG_ROOT="${RESULTS_ROOT}/batch_logs/${BATCH_NAME}"
  RUN_TRACKER="${RUN_TRACKER:-${RESULTS_ROOT}/run_tracker.csv}"
  mkdir -p "${LOG_ROOT}"
}

print_gpu_state() {
  echo "== nvidia-smi =="
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    echo "ERROR: nvidia-smi not found." >&2
    return 1
  fi
}

print_experiment_list() {
  echo "== ${BATCH_NAME} experiments =="
  local index=0
  while [[ ${index} -lt ${#EXPERIMENT_IDS[@]} ]]; do
    echo "- ${EXPERIMENT_IDS[${index}]} :: ${EXPERIMENT_CONFIGS[${index}]}"
    index=$((index + 1))
  done
}

ensure_not_overwriting() {
  local experiment_id="$1"
  local run_dir="${RUNS_ROOT}/${experiment_id}"
  if [[ -e "${run_dir}" ]]; then
    echo "ERROR: refusing to overwrite existing run directory: ${run_dir}" >&2
    echo "Move or archive that directory before rerunning this experiment." >&2
    return 1
  fi
}

update_tracker_if_available() {
  local experiment_id="$1"
  local status="$2"
  local log_path="$3"
  if [[ -f "${RUN_TRACKER}" ]]; then
    printf '%s,%s,%s,%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${experiment_id}" "${status}" "${log_path}" >> "${RUN_TRACKER}"
  fi
}

run_experiment() {
  local experiment_id="$1"
  local config_path="$2"
  local timestamp
  local log_path
  timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
  log_path="${LOG_ROOT}/${experiment_id}_${timestamp}.log"

  ensure_not_overwriting "${experiment_id}"
  echo
  echo "== Running ${experiment_id} =="
  echo "config=${config_path}"
  echo "log=${log_path}"

  if PYTHONDONTWRITEBYTECODE=1 python scripts/launch_yolo_experiment.py \
      --experiment-config "${config_path}" \
      --execute 2>&1 | tee "${log_path}"; then
    update_tracker_if_available "${experiment_id}" "complete" "${log_path}"
  else
    update_tracker_if_available "${experiment_id}" "failed" "${log_path}"
    echo "ERROR: ${experiment_id} failed. Stopping batch." >&2
    return 1
  fi
}

run_batch() {
  print_experiment_list
  print_gpu_state

  local index=0
  while [[ ${index} -lt ${#EXPERIMENT_IDS[@]} ]]; do
    run_experiment "${EXPERIMENT_IDS[${index}]}" "${EXPERIMENT_CONFIGS[${index}]}"
    index=$((index + 1))
  done

  echo
  echo "${BATCH_NAME} complete."
}
