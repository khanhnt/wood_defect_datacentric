# Faster R-CNN robustness check

This protocol is a nine-run VNWoodKnot experiment: Baseline, A1 crop, and A2 colour jitter at seeds 42, 43, and 44. It uses a COCO-pretrained `fasterrcnn_mobilenet_v3_large_fpn`, 1024-pixel inputs, 50 epochs, and checkpoint selection by validation mAP50-95. The materialized A1/A2 data are used for training only; all variants use byte-identical canonical validation and test images.

## Server preflight

```bash
cd /workspace/wood_defect_datacentric
source /workspace/wood_env/bin/activate

export DATA=/workspace/data/datasets_rebuilt
export FRCNN_GEN=/workspace/generations/access_r1_g3_fasterrcnn

python scripts/verify_fasterrcnn_protocol.py \
  --rebuilt-root "$DATA" \
  --results-root "$FRCNN_GEN" \
  --output-dir "$FRCNN_GEN/provenance"

python scripts/run_fasterrcnn_experiments.py \
  --rebuilt-root "$DATA" \
  --results-root "$FRCNN_GEN" \
  --gpus 0,1 \
  --batch-size 4 \
  --epochs 50 \
  --workers 4 \
  --dry-run
```

The verification gate must report `PASS (27/27)`. Its CSV proves image/label parity and byte-identical common validation/test inputs.

## One-epoch timing and memory smoke test

Run this before committing to the full matrix. It also downloads the official torchvision COCO weights once, avoiding a concurrent first-download race.

```bash
python scripts/run_fasterrcnn_experiments.py \
  --rebuilt-root "$DATA" \
  --results-root "${FRCNN_GEN}_smoke" \
  --gpus 0 \
  --variants baseline \
  --seeds 42 \
  --batch-size 4 \
  --epochs 1 \
  --workers 4

cat "${FRCNN_GEN}_smoke/fasterrcnn/vnwoodknot/per_seed/runs/baseline_seed42/training_summary.json"
```

Keep batch 4 if peak VRAM remains below 22 GB. If it exceeds 22 GB or OOMs, use batch 2 for every full run and record that single protocol-wide change. Do not choose batch independently by variant.

The launcher therefore does not reduce batch size per job by default. A failed main-matrix job exits nonzero and must be investigated; if batch 4 is rejected by the smoke test, restart the entire matrix at batch 2.

## Full training

```bash
nohup python scripts/run_fasterrcnn_experiments.py \
  --rebuilt-root "$DATA" \
  --results-root "$FRCNN_GEN" \
  --gpus 0,1 \
  --batch-size 4 \
  --epochs 50 \
  --workers 4 \
  > "$FRCNN_GEN/launcher.log" 2>&1 &

echo $! > "$FRCNN_GEN/launcher.pid"
tail -f "$FRCNN_GEN/launcher.log"
```

Each completed run must contain `weights/best.pt`, `weights/last.pt`, `results.csv`, `config_used.yaml`, `environment.json`, and `training_summary.json`.

## Fair prediction exports

```bash
for SPLIT in val test; do
  python scripts/export_fasterrcnn_predictions.py \
    --rebuilt-root "$DATA" \
    --results-root "$FRCNN_GEN" \
    --output-dir "$FRCNN_GEN/predictions/vnwoodknot/$SPLIT" \
    --split "$SPLIT" \
    --gpus 0,1 \
    --batch-size 4 \
    --workers 4
done
```

The exports retain detections at confidence 0.001, use NMS IoU 0.7 and at most 300 detections per image, and carry the Ultralytics-8.4.60-compatible ten-threshold TP mask required by the existing offline threshold analyses.

## Final audit and aggregation

```bash
python scripts/finalize_fasterrcnn_results.py \
  --rebuilt-root "$DATA" \
  --results-root "$FRCNN_GEN" \
  --predictions-root "$FRCNN_GEN/predictions/vnwoodknot" \
  --output-dir "$FRCNN_GEN/fasterrcnn/analysis"
```

Do not release the server unless `finalization_report.json` says `PASS`, with nine checkpoint runs, 18 metric rows, and 18 exact-mask prediction exports. The resulting `summary.csv` is the manuscript-ready mean and sample standard deviation over seeds.
