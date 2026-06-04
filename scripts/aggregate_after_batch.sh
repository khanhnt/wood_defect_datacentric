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
RUNS_ROOT="${PROJECT_ROOT}/results/runs"
SUMMARY_DIR="${RESULTS_ROOT}/batch_summaries"
SUMMARY_CSV="${SUMMARY_DIR}/run_summary_$(date -u +%Y%m%dT%H%M%SZ).csv"
mkdir -p "${SUMMARY_DIR}"

echo "== Aggregate after batch =="
echo "runs_root=${RUNS_ROOT}"
echo "summary_csv=${SUMMARY_CSV}"

python - "$RUNS_ROOT" "$SUMMARY_CSV" <<'PY'
import csv
import json
import sys
from pathlib import Path

runs_root = Path(sys.argv[1])
summary_csv = Path(sys.argv[2])
fieldnames = [
    "experiment_id",
    "status",
    "ok",
    "best_checkpoint_exists",
    "best_checkpoint_path",
    "metrics_status",
    "map50",
    "precision",
    "recall",
]
rows = []

for run_dir in sorted(path for path in runs_root.glob("*") if path.is_dir()):
    experiment_id = run_dir.name
    validation_path = run_dir / "validation_status.json"
    metrics_path = run_dir / "validation_metrics.json"
    checkpoint_path = run_dir / "ultralytics" / "train" / "weights" / "best.pt"
    validation = {}
    metrics = {}
    if validation_path.exists():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    best = metrics.get("best_by_map50") or {}
    rows.append(
        {
            "experiment_id": experiment_id,
            "status": "present",
            "ok": validation.get("ok", ""),
            "best_checkpoint_exists": checkpoint_path.exists(),
            "best_checkpoint_path": str(checkpoint_path),
            "metrics_status": metrics.get("status", ""),
            "map50": best.get("metrics/mAP50(B)", best.get(" metrics/mAP50(B)", "")),
            "precision": best.get("metrics/precision(B)", best.get(" metrics/precision(B)", "")),
            "recall": best.get("metrics/recall(B)", best.get(" metrics/recall(B)", "")),
        }
    )

summary_csv.parent.mkdir(parents=True, exist_ok=True)
with summary_csv.open("w", encoding="utf-8", newline="") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote: {summary_csv}")
print(f"Runs summarized: {len(rows)}")
PY

echo "Aggregation complete. Review the CSV before starting the next batch."
