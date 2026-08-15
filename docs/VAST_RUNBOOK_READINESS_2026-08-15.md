# Vast Runbook Readiness, 2026-08-15

## Scope

This report records the infrastructure status before renting the two-GPU Vast server.
No training or GPU inference was run on the Mac.

## VSB clean threshold split

- Source population: 1,992 strict-clean source images, exactly three tiles per source.
- Partition: deterministic Python `random.Random(42)` shuffle of lexicographically
  sorted source IDs.
- Threshold-selection half: 996 sources / 2,988 tiles, exposed as `val`.
- Final-test half: 996 sources / 2,988 tiles, exposed as `test`.
- Source overlap: zero. All tiles from one source remain in one half.
- Ratio rationale: equal halves balance threshold-selection stability and final-test
  precision without source leakage.
- Frozen outputs: `source_partition_manifest.csv`, `tile_partition_manifest.csv`,
  `partition_report.json`, and a per-variant evaluation YAML map.

Mac integration result: all seven VSB variants produced 2,988 validation and 2,988 test
tiles with zero source overlap.

## Deprecated checkpoint audit

The 18 archived A1/A2/P4+A4 checkpoints from `results 2/` are staged below the separate
`deprecated_checkpoints/` namespace. Their registry records source and staged paths,
SHA-256 values, source `args.yaml` SHA-256, and the role
`DEPRECATED_augmented_validation_selection`. Staging additionally verifies seed,
batch 40, 50 epochs, and image size 1024.

The audit evaluates 18 checkpoints on non-augmented val/test data and exports 36
low-confidence prediction JSONs. `compare_deprecated_checkpoints.py` reports corrected
minus deprecated precision, recall, mAP50, and mAP50-95 per seed and as mean/std. These
rows never enter the primary 42-checkpoint tables.

Estimated incremental cost: 0.8-1.5 GPU-hours, approximately 25-50 minutes on two RTX
3090 GPUs. No retraining is involved.

## AP reproduction policy

`verify_prediction_map_reproduction.py` first verifies checkpoint hash, dataset-YAML
hash, and image count. It then reports per-image differences between the
Ultralytics-style IoU-priority matcher and the confidence-ordered greedy matcher.

- Up to 0.002 absolute mAP50 residual: exact reproduction.
- Above 0.002 and up to 0.005: method review; acceptable only with matching provenance
  and a documented matching/interpolation explanation.
- Above 0.005, or any provenance mismatch: investigate before using the row in the
  manuscript.

The paid run does not stop on a method-review row. It completes provenance, checksums,
and result transfer so diagnosis can continue offline.

## Script and feature inventory

| Component | Status |
|---|---|
| `scripts/verify_generation_runtime.py` | Implemented; compile/CLI-tested; full gate is Vast-only |
| `scripts/generation_checkpoint_registry.py` | Implemented; real Mac staging/registry test |
| `scripts/build_generation_eval_map.py` | Implemented; real rebuilt-tree test |
| `scripts/verify_prediction_map_reproduction.py` | Implemented; matcher unit-tested; full AP check is post-inference |
| `scripts/write_generation_provenance.py` | Implemented; synthetic main+deprecated end-to-end test |
| `scripts/generation_status.py` | Implemented; empty pre-training log tested |
| `scripts/relocate_dataset_yamls.py` | Implemented; 33 real YAML copies tested |
| `scripts/build_minimal_transfer_manifest.py` | Implemented; real 306,138-file manifest test |
| `scripts/split_vsb_clean_sources.py` | Implemented; full Mac integration-tested |
| `scripts/stage_deprecated_checkpoints.py` | Implemented; 18/18 Mac integration-tested |
| `scripts/compare_deprecated_checkpoints.py` | Implemented; synthetic three-seed integration test |
| `run_all_experiments.py --job-set corrected24` | Implemented; queue unit-tested and 24-job dry run |
| `threshold_sweep_inference.py --split` | Implemented; mapped validation dry run |

Missing requested scripts/features: none.

Tests completed on Mac:

- Python compile check for the changed workflow scripts.
- Ten unit tests passed.
- Eleven new script CLIs passed `--help` smoke tests.
- VSB clean source split passed on the real rebuilt trees for all seven variants.
- Deprecated staging passed for all 18 archived checkpoints with matching source and
  staged SHA-256 values.

GPU-dependent full evaluation and prediction export remain to be run on Vast because
the new generation does not yet exist. Provenance assembly itself passed a synthetic
end-to-end Mac test; its final artifact set still depends on those Vast outputs.

## Transfer order

`revised/data/clean_data` is not part of STOP 2. The 24-job training queue starts after
the 84/84 VN/VSB training-data gate. The Mac uploads the 16 GiB clean source directory
in parallel. Strict-clean materialization and its separate 15/15 gate run only before
clean inference.

## Mac rclone authorization smoke

Before renting, reconnect the Drive remote and perform a real file read, not only a
metadata listing:

```bash
rclone config reconnect gdrive:
export RCLONE_SMOKE="$(mktemp -d)"
rclone copyto \
  "gdrive:2.Work/1.PTIT/1.Ca_nhan/2.Research/2026/workspace_20260624/data/vnwoodknot/images/test/knot_free/IMG_4832.jpg" \
  "$RCLONE_SMOKE/IMG_4832.jpg" -P
test -s "$RCLONE_SMOKE/IMG_4832.jpg"
cmp "$RCLONE_SMOKE/IMG_4832.jpg" \
  revised/data/vnwoodknot/images/test/knot_free/IMG_4832.jpg
```

Success requires both `test` and `cmp` to exit zero. An HTTP 403 means the OAuth grant
must be repaired before renting Vast.
