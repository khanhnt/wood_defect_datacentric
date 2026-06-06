#!/usr/bin/env bash
set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/setup_fresh_run.sh [--timestamp] [--force-empty]

Archives the current results/ directory and creates the Phase 1 fresh results
layout. Run this on the Vast.ai server before launching the multiseed queue.

Options:
  --timestamp    Append HHMMSS to the archive folder if the date-only name exists.
  --force-empty  Continue when results/ does not exist.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

USE_TIMESTAMP=0
FORCE_EMPTY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --timestamp)
      USE_TIMESTAMP=1
      shift
      ;;
    --force-empty)
      FORCE_EMPTY=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

DATE_TAG="$(date +%Y%m%d)"
ARCHIVE_DIR="results_old_bs16_${DATE_TAG}"
if [[ "${USE_TIMESTAMP}" -eq 1 && -e "${ARCHIVE_DIR}" ]]; then
  ARCHIVE_DIR="results_old_bs16_${DATE_TAG}_$(date +%H%M%S)"
fi

if [[ -e "${ARCHIVE_DIR}" ]]; then
  echo "ERROR: archive already exists: ${ARCHIVE_DIR}" >&2
  echo "Use --timestamp or move the existing archive first." >&2
  exit 1
fi

if [[ -d "results" ]]; then
  echo "Archiving results/ -> ${ARCHIVE_DIR}/"
  mv "results" "${ARCHIVE_DIR}"
else
  if [[ "${FORCE_EMPTY}" -ne 1 ]]; then
    echo "ERROR: results/ does not exist. Use --force-empty to create a fresh layout anyway." >&2
    exit 1
  fi
  echo "No results/ directory found; creating a fresh layout."
fi

mkdir -p \
  results/gpu_optimization/job_logs \
  results/gpu_optimization/generated_configs \
  results/gpu_optimization/materialization_logs \
  results/multiseed/vsb_rarefirst/per_seed \
  results/multiseed/vnwoodknot/per_seed \
  results/negative_aware/predictions \
  results/negative_aware/logs \
  results/negative_aware/threshold_sweep \
  results/negative_aware/bootstrap \
  results/negative_aware/plots

touch results/.gitkeep

echo "Fresh results layout ready:"
find results -maxdepth 3 -type d | sort
