# VSB Rare-First Split

This folder contains portable VSB metadata for the rare-first benchmark.

Included now:

- `manifest.jsonl`: full tile-level rare-first `train/val/test` manifest generated on the server and sanitized to relative paths.
- `source_manifest_sanitized.jsonl`: source-image metadata with portable relative paths.
- `test_tile_manifest.csv`: held-out rare-first test tiles recovered from the released prediction metadata.
- `split_summary.json`: known split counts used by the paper (`train=7679`, `val=977`, `test=972`).

Manifest sanity:

- Total tiles: 9,628.
- Train/val/test: 7,679 / 977 / 972.
- Boxes in train/val/test: 9,346 / 1,146 / 1,173.
- Paths are relative to the reconstructed YOLO dataset root.
