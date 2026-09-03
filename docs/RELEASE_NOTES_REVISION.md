# Revision Reproducibility Release

This release accompanies the revised IEEE Access manuscript "Beyond mAP: Negative-Aware Evaluation of Data-Centric Pipelines for Wood Knot Detection".

## Included

- Source code and complete YAML configurations for the seven-variant YOLOv8s study.
- Source code for the three-variant Faster R-CNN robustness experiment.
- Portable VNWoodKnot, VSB rare-first, and VSB strict-clean manifests.
- Final detector-separated CSV/JSON table artifacts under `results/tables/`.
- Per-seed standard metrics, validation-selected operating points, tolerance analyses, calibration summaries, and reviewer-requested audits.
- Final revision figures in vector PDF and 300-DPI PNG form.
- Runtime, checkpoint, prediction, and protocol provenance for the Faster R-CNN block.
- A release integrity checker covering both detector families.

## Protocol Corrections Reflected Here

- Training-time augmentation is materialized only for the training split; validation and test use common non-augmented evaluation images.
- Negative-aware thresholds are selected on clean validation material and applied unchanged to held-out clean test material.
- The 1,992 VSB defect-free source images are partitioned by source ID into disjoint 996-source validation and test halves, each yielding 2,988 tiles.
- The older 6,252-tile VSB denominator and pre-revision table bundle are explicitly marked as superseded.
- YOLOv8s and Faster R-CNN values are reported in separate detector blocks and are compared only within a block.

## Not Included

- Raw public-dataset images.
- Materialized training/evaluation image trees.
- Model checkpoints.
- Large per-seed prediction JSON exports.

The derived tables are self-contained for manuscript-value verification. Checkpoints and prediction exports may be deposited separately for full replay of training-free analyses.

## Release Check

```bash
python scripts/release_integrity_check.py
```

The release should be tagged only after this command and the test suite pass on a clean checkout.
