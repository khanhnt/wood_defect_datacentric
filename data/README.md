# Data Preparation

This repository does not redistribute raw wood-surface images. Use the original dataset sources and the manifests in this folder to reconstruct the YOLO datasets used in the paper.

## Sources

- VNWoodKnot: Data in Brief, DOI `10.1016/j.dib.2025.112039`.
- VSB/Kodytek large-scale wood surface defects: F1000Research, DOI `10.12688/f1000research.52903.x`.

The VSB clean-wood set used for the negative-aware extension consists of the 1,992 source images whose annotation files are empty (`*_anno.txt`). These images are tiled with the same settings as the rare-first benchmark: tile size 1024 px, horizontal overlap 128 px, yielding 5,976 clean tiles.

## Included Manifests

- `vnwoodknot_split/manifest.jsonl`: portable VNWoodKnot split manifest.
- `vsb_rarefirst_split/manifest.jsonl`: full tile-level VSB rare-first split manifest.
- `vsb_rarefirst_split/source_manifest_sanitized.jsonl`: portable VSB source-image metadata with relative paths.
- `vsb_rarefirst_split/test_tile_manifest.csv`: held-out VSB rare-first test tiles used by released prediction files.
- `vsb_rarefirst_split/split_summary.json`: split counts used in the paper.
- `vsb_clean_manifest/source_ids.txt`: 1,992 VSB defect-free source image IDs.
- `vsb_clean_manifest/clean_tile_manifest.csv`: 5,976 VSB clean tiles.

Raw images, YOLO materializations, preprocessing outputs, checkpoints, and prediction JSON files are intentionally excluded from GitHub.

## Reconstruct YOLO Datasets

VNWoodKnot:

```bash
python data/prepare_vnwoodknot.py \
  --manifest data/vnwoodknot_split/manifest.jsonl \
  --images-root /path/to/VNWoodKnot \
  --output-root /path/to/vnwoodknot_live_dead_2class_yolo \
  --link-mode symlink
```

VSB rare-first:

```bash
python data/prepare_vsb.py \
  --manifest data/vsb_rarefirst_split/manifest.jsonl \
  --images-root /path/to/VSB/tiles_or_images \
  --output-root /path/to/vsb7_3600_rare_first_yolo \
  --link-mode symlink
```

VSB clean wood:

```bash
python analysis/vsb_negative_aware.py \
  --prepare-only \
  --clean-images-root /path/to/VSB_defect_free_images \
  --clean-ids-file configs/datasets/vsb_clean_source_ids.txt \
  --overwrite-clean-set \
  --overwrite-eval-datasets
```
