#!/usr/bin/env bash
set -Eeuo pipefail

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
GENERATED_DATA_ROOT="${GENERATED_DATA_ROOT:-/workspace/data/wood_defect_datacentric/generated_yolo}"

echo "== Wood defect data-centric server setup check =="
echo "project_root=${PROJECT_ROOT}"
echo "results_root=${RESULTS_ROOT}"
echo "generated_data_root=${GENERATED_DATA_ROOT}"

echo
echo "== GPU =="
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: nvidia-smi is not available. This is not a usable NVIDIA training server." >&2
  exit 1
fi
nvidia-smi

echo
echo "== Python =="
python -V

echo
echo "== PyTorch CUDA =="
python - <<'PY'
import sys

try:
    import torch
except Exception as exc:
    print(f"ERROR: PyTorch import failed: {exc}", file=sys.stderr)
    raise SystemExit(1)

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
print(f"cuda_version={torch.version.cuda}")
print(f"cuda_device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    print("ERROR: PyTorch CUDA is not available.", file=sys.stderr)
    raise SystemExit(1)
for index in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(index)
    print(f"cuda_device[{index}]={props.name}, vram_gb={props.total_memory / (1024 ** 3):.2f}")
PY

echo
echo "== Ultralytics =="
python - <<'PY'
import sys

try:
    import ultralytics
except Exception as exc:
    print(f"ERROR: Ultralytics import failed: {exc}", file=sys.stderr)
    raise SystemExit(1)

print(f"ultralytics={ultralytics.__version__}")
PY

if ! command -v yolo >/dev/null 2>&1; then
  echo "ERROR: yolo CLI is not available on PATH." >&2
  exit 1
fi
yolo version || true

echo
echo "== Disk =="
df -h "${PROJECT_ROOT}" "${RESULTS_ROOT%/*}" 2>/dev/null || df -h "${PROJECT_ROOT}"

echo
echo "== Folders =="
mkdir -p "${RESULTS_ROOT}" "${RESULTS_ROOT}/logs" "${GENERATED_DATA_ROOT}"
test -w "${RESULTS_ROOT}"
test -w "${GENERATED_DATA_ROOT}"
echo "Created/verified writable folders."

echo
echo "Server setup probe passed. No training was started."
