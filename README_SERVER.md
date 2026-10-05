# GPU Server Notes

These notes cover the reported YOLOv8s and Faster R-CNN experiment environments. Use `RUNBOOK.md` for the complete generation workflow and `REPRODUCE.md` for the paper-artifact map. Do not commit raw datasets, materialized image trees, checkpoints, prediction exports, credentials, or generated run directories.

## Environment

The frozen runs used Python 3.12, PyTorch 2.6.0 with CUDA 12.4, Ultralytics 8.4.60, OpenCV 4.10.0, and two NVIDIA RTX 3090 GPUs. The runtime gate records exact package and driver versions:

```bash
python scripts/verify_generation_runtime.py \
  --expected-gpus 2 \
  --output /path/to/generation/provenance/runtime_preflight.json
```

The gate must report `PASS` before training.

## Data

Obtain VNWoodKnot and VSB from the sources in `data/README.md`. Reconstruct canonical datasets only from the tracked manifests:

```bash
python scripts/materialize_yolo_from_manifest.py \
  --manifest data/vnwoodknot_split/manifest.jsonl \
  --images-root /path/to/VNWoodKnot/images \
  --output-root /path/to/datasets_rebuilt/canonical/vnwoodknot \
  --dataset-name vnwoodknot \
  --classes live_knot dead_knot \
  --split-strategy manifest \
  --link-mode copy \
  --exclude-image-id train/2/img_3671

python scripts/materialize_yolo_from_manifest.py \
  --manifest data/vsb_rarefirst_split/manifest.jsonl \
  --images-root /path/to/VSB/images \
  --output-root /path/to/datasets_rebuilt/canonical/vsb_rarefirst \
  --dataset-name vsb_rarefirst \
  --classes live_knot dead_knot resin knot_with_crack crack marrow knot_missing \
  --split-strategy manifest \
  --link-mode copy
```

Build preprocessing variants for all splits and augmentation variants for `train` only. The verification gate must pass before any training starts:

```bash
python scripts/verify_rebuilt_datasets.py \
  --root /path/to/datasets_rebuilt \
  --datasets vnwoodknot vsb_rarefirst \
  --output-csv /path/to/datasets_rebuilt/reports/verification_gate.csv \
  --output-md /path/to/datasets_rebuilt/reports/verification_gate.md
```

Expected canonical counts are VNWoodKnot 1,059/226/229 and VSB rare-first 7,679/977/972 for train/validation/test.

## Training

Run a dry-run before the full queue:

```bash
python scripts/run_all_experiments.py \
  --job-set corrected24 --dataset all --batch-size 40 \
  --epochs 50 --imgsz 1024 --workers 4 --gpus 0,1 \
  --rebuilt-root /path/to/datasets_rebuilt \
  --results-root /path/to/generation \
  --dry-run
```

Remove `--dry-run` only after checking all 24 job rows. The complete YOLOv8s generation comprises these 24 runs plus the 18 unaffected registered checkpoints, for 42 runs total. Both `best.pt` and `last.pt` are required for newly trained runs.

The Faster R-CNN protocol and nine-run command are documented in `docs/FASTERRCNN_ROBUSTNESS_RUNBOOK.md`.

## Before Releasing A Server

1. Verify the expected checkpoint and prediction counts.
2. Run AP reproduction and require every primary cell to pass its exact tolerance.
3. Write provenance and `SHA256SUMS`.
4. Copy the complete generation off the server and validate the checksums at the destination.
5. Build the external archive with `scripts/package_release_archive.py --strict`.

Do not terminate the server until the destination checksum check passes.
