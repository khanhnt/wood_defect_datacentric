#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

GEN="${GEN:-/workspace/generations/access_r1_g2}"
DATA="${DATA:-/workspace/data/datasets_rebuilt}"
GPU_LIST="${GPU_LIST:-0,1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEPRECATED_ROOT="$GEN/deprecated_checkpoints/multiseed"

required_files=(
  "$GEN/fair_eval/fair_metrics.csv"
  "$GEN/deprecated_audit/fair_eval/fair_metrics.csv"
  "$GEN/provenance/checkpoint_registry.csv"
  "$GEN/deprecated_checkpoints/deprecated_checkpoint_registry.csv"
  "$DATA/canonical/vnwoodknot/dataset.yaml"
  "$DATA/canonical/vsb_rarefirst/dataset.yaml"
)
for path in "${required_files[@]}"; do
  if [[ ! -s "$path" ]]; then
    echo "ERROR: required file is missing or empty: $path" >&2
    exit 1
  fi
done

if ! grep -q 'ultralytics_detection_validator' scripts/threshold_sweep_inference.py; then
  echo "ERROR: checkout does not contain the validation-path exporter." >&2
  exit 1
fi

echo "[preflight] validating the 126 primary exports from Sections 8.1-8.3"
"$PYTHON_BIN" - "$GEN" <<'PY'
import json
import sys
from collections import Counter
from pathlib import Path

root = Path(sys.argv[1]) / "predictions"
files = sorted(root.glob("*/*/*_predictions.json"))
paths = Counter()
counts = Counter()
bad = []
for path in files:
    payload = json.loads(path.read_text(encoding="utf-8"))
    paths[payload.get("inference_path")] += 1
    counts[(payload.get("dataset"), payload.get("split"), int(payload.get("num_images", -1)))] += 1
    if payload.get("inference_path") != "ultralytics_detection_validator":
        bad.append(str(path))

print("primary_json:", len(files))
print("inference_paths:", dict(paths))
print("dataset_split_counts:", dict(counts))
if len(files) != 126 or bad:
    print("ERROR: Sections 8.1-8.3 are incomplete or contain legacy exports.", file=sys.stderr)
    for path in bad[:10]:
        print(path, file=sys.stderr)
    raise SystemExit(1)
PY

echo "[8.4] exporting deprecated checkpoints through DetectionValidator"
for dataset in vnwoodknot vsb_rarefirst; do
  if [[ "$dataset" == "vnwoodknot" ]]; then
    RUNS="$DEPRECATED_ROOT/vnwoodknot/per_seed/runs"
    RAW="$DATA/canonical/vnwoodknot/dataset.yaml"
    P4="$DATA/variants/vnwoodknot/preprocessing/P4_combined_safe/dataset.yaml"
  else
    RUNS="$DEPRECATED_ROOT/vsb_rarefirst/per_seed/runs"
    RAW="$DATA/canonical/vsb_rarefirst/dataset.yaml"
    P4="$DATA/variants/vsb_rarefirst/preprocessing/P4_combined_safe/dataset.yaml"
  fi

  for split in val test; do
    "$PYTHON_BIN" scripts/threshold_sweep_inference.py \
      --dataset-name "$dataset" \
      --split "$split" \
      --checkpoint-root "$RUNS" \
      --output-dir "$GEN/deprecated_audit/predictions/$dataset/$split" \
      --gpus "$GPU_LIST" \
      --variants a1_crop a2_colorjitter p4_a4_combined \
      --seeds 42 43 44 \
      --conf 0.001 \
      --iou 0.7 \
      --imgsz 1024 \
      --batch 32 \
      --max-det 300 \
      --variant-data-yaml "a1_crop=$RAW" \
      --variant-data-yaml "a2_colorjitter=$RAW" \
      --variant-data-yaml "p4_a4_combined=$P4" \
      --overwrite
  done
done

echo "[8.4] exporting deprecated VSB checkpoints on source-disjoint strict-clean views"
for split in val test; do
  "$PYTHON_BIN" scripts/threshold_sweep_inference.py \
    --dataset-name vsb_strict_clean \
    --split "$split" \
    --checkpoint-root "$DEPRECATED_ROOT/vsb_rarefirst/per_seed/runs" \
    --output-dir "$GEN/deprecated_audit/predictions/vsb_strict_clean/$split" \
    --gpus "$GPU_LIST" \
    --variants a1_crop a2_colorjitter p4_a4_combined \
    --seeds 42 43 44 \
    --conf 0.001 \
    --iou 0.7 \
    --imgsz 1024 \
    --batch 32 \
    --max-det 300 \
    --variant-data-yaml "a1_crop=$DATA/eval_views/vsb_strict_clean/a1_crop/dataset.yaml" \
    --variant-data-yaml "a2_colorjitter=$DATA/eval_views/vsb_strict_clean/a2_colorjitter/dataset.yaml" \
    --variant-data-yaml "p4_a4_combined=$DATA/eval_views/vsb_strict_clean/p4_a4_combined/dataset.yaml" \
    --overwrite
done

echo "[8.4] validating deprecated export count and metadata"
"$PYTHON_BIN" - "$GEN" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1]) / "deprecated_audit" / "predictions"
files = sorted(root.glob("*/*/*_predictions.json"))
bad = []
for path in files:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("inference_path") != "ultralytics_detection_validator":
        bad.append(str(path))
print("deprecated_json:", len(files))
print("invalid_exporter:", len(bad))
if len(files) != 54 or bad:
    raise SystemExit("Deprecated prediction export verification failed.")
PY

echo "[9.1] reproducing primary fair-evaluation AP"
"$PYTHON_BIN" scripts/verify_prediction_map_reproduction.py \
  --predictions-root "$GEN/predictions" \
  --fair-summary "$GEN/fair_eval/fair_metrics.csv" \
  --checkpoint-registry "$GEN/provenance/checkpoint_registry.csv" \
  --output-csv "$GEN/fair_eval/prediction_ap_reproduction.csv" \
  --diagnostics-csv "$GEN/fair_eval/prediction_ap_matching_diagnostics.csv" \
  --exact-tolerance 0.002 \
  --review-tolerance 0.005

echo "[9.1] reproducing deprecated-checkpoint fair-evaluation AP"
"$PYTHON_BIN" scripts/verify_prediction_map_reproduction.py \
  --predictions-root "$GEN/deprecated_audit/predictions" \
  --fair-summary "$GEN/deprecated_audit/fair_eval/fair_metrics.csv" \
  --checkpoint-registry "$GEN/deprecated_checkpoints/deprecated_checkpoint_registry.csv" \
  --output-csv "$GEN/deprecated_audit/fair_eval/prediction_ap_reproduction.csv" \
  --diagnostics-csv "$GEN/deprecated_audit/fair_eval/prediction_ap_matching_diagnostics.csv" \
  --exact-tolerance 0.002 \
  --review-tolerance 0.005

echo "[9.2] writing provenance and checksums"
"$PYTHON_BIN" scripts/write_generation_provenance.py \
  --generation-root "$GEN" \
  --fair-summary "$GEN/fair_eval/fair_metrics.csv" \
  --checkpoint-registry "$GEN/provenance/checkpoint_registry.csv" \
  --prediction-root "$GEN/predictions" \
  --deprecated-checkpoint-registry "$GEN/deprecated_checkpoints/deprecated_checkpoint_registry.csv" \
  --deprecated-fair-summary "$GEN/deprecated_audit/fair_eval/fair_metrics.csv" \
  --deprecated-prediction-root "$GEN/deprecated_audit/predictions" \
  --pretrained-weights yolov8s.pt \
  --vn-manifest data/vnwoodknot_split/manifest.jsonl \
  --vsb-manifest data/vsb_rarefirst_split/manifest.jsonl \
  --vsb-clean-manifest "$DATA/eval_views/vsb_strict_clean/source_partition_manifest.csv" \
  --extra-manifest data/vsb_clean_manifest/clean_tile_manifest.csv \
  --extra-manifest "$DATA/canonical/vsb_strict_clean/clean_materialized_samples.csv" \
  --extra-manifest "$DATA/eval_views/vsb_strict_clean/tile_partition_manifest.csv" \
  --extra-manifest "$DATA/eval_views/vsb_strict_clean/partition_report.json"

echo "[summary]"
"$PYTHON_BIN" - "$GEN" <<'PY'
import csv
import sys
from collections import Counter
from pathlib import Path

generation = Path(sys.argv[1])
checks = [
    ("primary", generation / "fair_eval" / "prediction_ap_reproduction.csv", 84),
    ("deprecated", generation / "deprecated_audit" / "fair_eval" / "prediction_ap_reproduction.csv", 36),
]
all_exact = True
for label, path, expected in checks:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    statuses = Counter(row["status"] for row in rows)
    maximum = max((float(row["abs_residual"]) for row in rows), default=float("nan"))
    print(f"{label}: rows={len(rows)}/{expected}, statuses={dict(statuses)}, max_abs_residual={maximum:.9f}")
    all_exact &= len(rows) == expected and statuses == {"EXACT_PASS": expected}

print("primary_prediction_json:", len(list((generation / "predictions").glob("*/*/*_predictions.json"))))
print(
    "deprecated_prediction_json:",
    len(list((generation / "deprecated_audit" / "predictions").glob("*/*/*_predictions.json"))),
)
print("FINAL AUDIT:", "PASS" if all_exact else "REVIEW REQUIRED")
PY

echo "Completed remaining Sections 8.4-9.2. Generation: $GEN"
