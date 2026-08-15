# Runbook Component Inventory, 2026-08-15

## Verdict

All requested scripts, flags, and experiment configurations exist. The six CPU tools
requested for real Mac execution were run against `revised/datasets_rebuilt/` and
`results 2/`. No training or GPU inference was run.

One defect was found during testing: `write_generation_provenance.py` passed
`torch.__version__` directly to PyYAML. Some PyTorch builds expose that value as a
`TorchVersion` string subclass, which `yaml.safe_dump()` cannot represent. The value is
now converted to a plain string. A synthetic end-to-end provenance assembly then passed
with four artifact records.

## Script Inventory

| Component | Exists | Mac test performed | Result | Commit state before this report |
|---|---:|---|---|---|
| `verify_generation_runtime.py` | Yes | Compile and `--help` | PASS. Full gate intentionally remains a Vast test: this Mac is Python 3.9 without the pinned CUDA/Ultralytics runtime. | Tracked at `713b330` |
| `generation_checkpoint_registry.py` | Yes | Real `results 2/` staging and registry build | 18 archived survivors staged; 18 PASS and 24 expected pre-training missing rows. `--allow-missing` returned success. | Tracked at `713b330` |
| `stage_deprecated_checkpoints.py` | Yes | Real `results 2/` audit/stage | PASS 18/18; seed, batch 40, 50 epochs, 1024 image size, args hash, and checkpoint hash verified. | Tracked at `713b330` |
| `build_generation_eval_map.py` | Yes | Real rebuilt datasets | PASS; 14 fair-evaluation rows and 7 strict-clean rows. | Tracked at `713b330` |
| `split_vsb_clean_sources.py` | Yes | Full real rebuilt-tree integration | PASS; 996/996 source split, zero source overlap, 2,988 tiles in each half for all seven variants. | Tracked at `713b330` |
| `verify_prediction_map_reproduction.py` | Yes | Compile, CLI, and matcher regression test | PASS for saved-export array parsing, IoU, Ultralytics-style matching, and alternate greedy matching. Full AP integration remains a Vast/post-inference test because Ultralytics 8.4.60 and generation predictions are not on this Mac. | Tracked at `713b330` |
| `compare_deprecated_checkpoints.py` | Yes | Synthetic three-seed corrected/deprecated table | PASS; generated per-seed and mean/std tables and recovered the injected -0.0500 mAP50 delta. | Tracked at `713b330` |
| `write_generation_provenance.py` | Yes | Synthetic main+deprecated end-to-end assembly | PASS after the `TorchVersion` fix; wrote CSV, JSON, manifests, YAML copies, environment YAML, and checksums for four records. | Tracked at `713b330`; fix is part of the new commit |
| `generation_status.py` | Yes | Empty pre-training run log | PASS; reported 0/24 and estimated remaining VN/VSB time instead of failing. | Tracked at `713b330` |
| `relocate_dataset_yamls.py` | Yes | YAML-only copy of the real rebuilt tree | PASS; relocated 33 dataset YAML files without touching rebuilt data. | Tracked at `713b330` |
| `build_minimal_transfer_manifest.py` | Yes | Real rebuilt datasets | PASS; selected 306,138 files, 79.25 GiB logical and 65.26 GiB physical with hard-link deduplication. | Tracked at `713b330` |

All eleven CLIs also passed compile and `--help` checks. The repository unit suite now
passes 10/10 tests.

## Requested Real Mac Outputs

### Generation checkpoint registry

The pre-training registry contains 42 planned runs:

- 18 `PASS` archived survivors staged from `results 2/`.
- 24 `FAIL` rows are the deliberately absent corrected runs that will be trained on
  Vast. This is the expected pre-training state, not an implementation failure.

### Deprecated checkpoint staging

`DEPRECATED CHECKPOINT REGISTRY: PASS (18/18)`.

All checkpoints are tagged `DEPRECATED_augmented_validation_selection`; none are
eligible for the primary corrected tables.

### VSB clean source partition

The frozen report is also stored at
`docs/runbook_test_evidence/vsb_clean_partition_report.json`.

```json
{
  "status": "PASS",
  "seed": 42,
  "ratio": "996/996 (50/50 by source ID)",
  "selection_source_count": 996,
  "final_test_source_count": 996,
  "source_overlap": 0,
  "tiles_per_source": 3,
  "selection_tiles_per_variant": 2988,
  "final_test_tiles_per_variant": 2988
}
```

The split is a deterministic `random.Random(42)` shuffle of sorted source IDs. It is
50/50 by source rather than tile, so all three tiles from a source remain in one half.

### Evaluation maps

- Fair map: 14 rows = two datasets x seven variants.
- Strict-clean map: 7 VSB variant rows.
- Baseline, A1, and A2 use canonical non-augmented evaluation data.
- P1, P2, P3, and P4+A4 use their preprocessing-only evaluation data.

### Minimal transfer manifest

- Files: 306,138.
- Logical size: 79.25 GiB.
- Physical size with hard-link deduplication: 65.26 GiB.

### YAML relocation

A YAML-only copy of the actual rebuilt tree was relocated successfully: 33/33 files.
The real `revised/datasets_rebuilt/` tree was not changed.

## Flag Inventory

| Flag/feature | Exists | Mac test | Result |
|---|---:|---|---|
| `run_all_experiments.py --job-set corrected24` | Yes | Actual dry run | PASS; exactly 24 jobs: 15 VN and 9 VSB. |
| `threshold_sweep_inference.py --split` | Yes | Actual dry run with `--split val` | PASS. |
| `threshold_sweep_inference.py --variant-data-yaml` | Yes | Actual baseline mapping in the same dry run | PASS. |
| `evaluate_corrected_common.py --eval-map-csv` | Yes | Actual `--prepare-only` run | PASS; resolved rebuilt VN/VSB paths and wrote the map. |
| `verify_rebuilt_datasets.py --datasets` | Yes | Actual VN+VSB gate | PASS 84/84. |
| `vsb_negative_aware.py --link-mode` | Yes | Parser/CLI plus rebuilt strict-clean reports | PASS; link mode is accepted and recorded. No rematerialization was done for this inventory. |
| `materialize_yolo_from_manifest.py --exclude-image-id` | Yes | Existing rebuilt canonical report | PASS; explicitly excluded `train/2/img_3671`, yielding VN train=1,059. |

## VN P1/P2/P3 Configuration Audit

The semantic comparison used P2 as the reference. P1 and P3 differ from P2 only in
the four intended identity fields:

1. `experiment_id`
2. human-readable `description`
3. `dataset.data_yaml` and its environment variable
4. `transforms.preprocessing`

There are no differences in split protocol, augmentation (`A0_default`), model,
epochs, image size, batch expression, workers, seed, patience, device, optimizer,
pretrained/deterministic/single-class settings, or output root.

### P1

```yaml
experiment_id: vn_yolov8s_p1_clahe_e50
description: VNWoodKnot preprocessing candidate using CLAHE on the luminance channel.
dataset:
  key: vnwoodknot
  data_yaml: ${WOOD_DC_VN_P1_DATASET_YAML:-/workspace/data/wood_defect_datacentric/generated_yolo/vnwoodknot/P1_CLAHE_luminance/dataset.yaml}
  split_protocol: existing_vnwoodknot_train_val_test_with_knot_free_retained
transforms:
  preprocessing: P1_CLAHE_luminance
  augmentation: A0_default
  materialized_dataset_required: true
training:
  model: yolov8s
  epochs: 50
  imgsz: ${IMG_SIZE:-1024}
  batch: ${BATCH_SIZE:-32}
  workers: ${WORKERS:-4}
  seed: 42
  patience: 30
  device: "${DEVICE:-0}"
  optimizer: auto
  pretrained: true
  deterministic: true
  single_cls: false
outputs:
  output_root: results
```

### P2 reference

```yaml
experiment_id: vn_yolov8s_p2_illumination_e50
description: VNWoodKnot selected preprocessing candidate using conservative luminance normalization.
dataset:
  key: vnwoodknot
  data_yaml: ${WOOD_DC_VN_P2_DATASET_YAML:-/workspace/data/wood_defect_datacentric/generated_yolo/vnwoodknot/P2_illumination_normalization/dataset.yaml}
  split_protocol: existing_vnwoodknot_train_val_test_with_knot_free_retained
transforms:
  preprocessing: P2_illumination_normalization
  augmentation: A0_default
  materialized_dataset_required: true
training:
  model: yolov8s
  epochs: 50
  imgsz: ${IMG_SIZE:-1024}
  batch: ${BATCH_SIZE:-32}
  workers: ${WORKERS:-4}
  seed: 42
  patience: 30
  device: "${DEVICE:-0}"
  optimizer: auto
  pretrained: true
  deterministic: true
  single_cls: false
outputs:
  output_root: results
```

### P3

```yaml
experiment_id: vn_yolov8s_p3_unsharp_e50
description: VNWoodKnot preprocessing candidate using mild unsharp masking.
dataset:
  key: vnwoodknot
  data_yaml: ${WOOD_DC_VN_P3_DATASET_YAML:-/workspace/data/wood_defect_datacentric/generated_yolo/vnwoodknot/P3_mild_unsharp/dataset.yaml}
  split_protocol: existing_vnwoodknot_train_val_test_with_knot_free_retained
transforms:
  preprocessing: P3_mild_unsharp
  augmentation: A0_default
  materialized_dataset_required: true
training:
  model: yolov8s
  epochs: 50
  imgsz: ${IMG_SIZE:-1024}
  batch: ${BATCH_SIZE:-32}
  workers: ${WORKERS:-4}
  seed: 42
  patience: 30
  device: "${DEVICE:-0}"
  optimizer: auto
  pretrained: true
  deterministic: true
  single_cls: false
outputs:
  output_root: results
```

## Remaining Vast-Only Gates

Two items are implemented and Mac-tested at the level possible without the paid
runtime, but cannot be marked end-to-end complete yet:

1. `verify_generation_runtime.py`: must verify Python 3.12, pinned packages, CUDA,
   driver, and two GPUs on the rented host.
2. `verify_prediction_map_reproduction.py`: the matcher is regression-tested, but the
   full AP comparison requires Ultralytics 8.4.60 and the newly exported prediction
   generation.

These are explicit runbook gates, not missing code.
