# Deprecated augmentation checkpoints

The 18 checkpoints listed below are retained for audit only. Their materialized A-variant datasets applied A1/A2/A4 to validation and test images as well as training images. They must not be used in the revised primary comparison because augmentation is a training-only intervention.

Replacement runs must use the rebuilt train-only datasets under `revised/datasets_rebuilt/variants/<dataset>/augmentation/seed<seed>/`. Validation and test are hardlinked from the non-augmented canonical input appropriate to the pipeline. New runs must retain both `best.pt` and `last.pt`.

| Dataset | Variant | Seeds | Archived run paths | Status |
|---|---|---|---|---|
| VNWoodKnot | A1 crop | 42, 43, 44 | `results 2/multiseed/vnwoodknot/per_seed/runs/a1_crop_seed<seed>/` | DEPRECATED |
| VNWoodKnot | A2 colour jitter | 42, 43, 44 | `results 2/multiseed/vnwoodknot/per_seed/runs/a2_colorjitter_seed<seed>/` | DEPRECATED |
| VNWoodKnot | P4+A4 combined | 42, 43, 44 | `results 2/multiseed/vnwoodknot/per_seed/runs/p4_a4_combined_seed<seed>/` | DEPRECATED |
| VSB rare-first | A1 crop | 42, 43, 44 | `results 2/multiseed/vsb_rarefirst/per_seed/runs/a1_crop_seed<seed>/` | DEPRECATED |
| VSB rare-first | A2 colour jitter | 42, 43, 44 | `results 2/multiseed/vsb_rarefirst/per_seed/runs/a2_colorjitter_seed<seed>/` | DEPRECATED |
| VSB rare-first | P4+A4 combined | 42, 43, 44 | `results 2/multiseed/vsb_rarefirst/per_seed/runs/p4_a4_combined_seed<seed>/` | DEPRECATED |

The files are intentionally not deleted. They support a separate audit comparison quantifying the effect of applying augmentation to evaluation splits.
