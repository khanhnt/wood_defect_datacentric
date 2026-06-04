# Handoff Context for Fresh Codex Chat

## Research Goal

This is the independent IJACSA/thesis data-centric wood surface defect detection project. The detector is fixed to YOLOv8s. The research question is whether careful preprocessing, defect-preserving augmentation, and negative-aware evaluation improve a practical wood defect pipeline.

## Difference from the Accepted CMC Paper

The accepted CMC paper emphasized detector-family comparison, protocol breadth, and source-to-target transfer/domain shift. This project should avoid repeating that contribution. Do not frame the new work as another detector comparison. Use YOLOv8s as the controlled detector and focus on data-centric interventions plus VNWoodKnot negative-image behavior.

## Implemented Through Prompt 6

- Project skeleton with `configs/`, `datasets/`, `preprocessing/`, `augmentation/`, `training/`, `evaluation/`, `visualization/`, `scripts/`, `results/`, and `docs/`.
- Dataset adapters and split verification for VSB curated benchmark and VNWoodKnot.
- Dataset statistics reports for image counts, class box counts, negative/background-only images, bbox area summaries, and VNWoodKnot `knot_free` retention.
- Preprocessing variants: `P0_baseline`, `P1_CLAHE_luminance`, `P2_illumination_normalization`, `P3_mild_unsharp`, and `P4_combined_safe`.
- Augmentation variants: `A0_default`, `A1_defect_preserving_crop`, `A2_texture_aware_color_jitter`, `A3_copy_paste_defects`, and `A4_combined_best`.
- YOLOv8s training launcher with dry-run mode, resume mode, structured run folders, and `configs/experiment_matrix.csv`.
- Negative-aware VNWoodKnot evaluation with threshold sensitivity, knot-free false-positive image rate, mean predictions per knot-free image, confidence summaries, class-specific false positives, CSV outputs, plots, and Markdown summary.

## Project Structure

```text
configs/          Project, experiment, preprocessing, and augmentation configs
data/processed/   Small manifest metadata copied into this independent repo
datasets/         Manifest adapters and statistics helpers
preprocessing/    Safe preprocessing implementations and materialization helpers
augmentation/     Defect-preserving augmentation implementations and materialization helpers
evaluation/       Negative-aware evaluation code
scripts/          CLI entry points for audits, previews, training dry-runs, and evaluation
results/          Generated local outputs; git-ignored except .gitkeep
docs/             Audit notes, method notes, and this handoff
```

## Dataset Expectations

- Metadata manifests are expected under `data/processed/`.
- Full image datasets are not copied into this repository.
- Local audit scripts can run from the manifests, but image existence depends on whether the original local dataset paths are mounted.
- VSB controlled training should use the curated YOLO dataset YAML, usually `/workspace/data/main_dataset/benchmarks/vsb7_3600_rare_first_yolo/dataset.yaml` on Vast.ai.
- VNWoodKnot controlled training should use `/workspace/data/vnwoodknot/benchmarks/vnwoodknot_live_dead_2class_yolo/dataset.yaml` on Vast.ai.
- Generated preprocessing/augmentation YOLO folders should be placed under `/workspace/data/wood_defect_datacentric/generated_yolo/...` unless overridden by environment variables.
- VNWoodKnot `knot_free` images must remain in train/val/test and must be used for negative-aware evaluation.

## Known Audit Notes

- The local VSB manifest may include records beyond the final seven-class curated YOLO split and may reference image paths unavailable on a given machine. Treat this as audit context, not a reason to regenerate splits without approval.
- The Vast.ai YOLO dataset YAML paths are expected to be unavailable on a Mac unless the `/workspace/data/...` layout is mounted.
- Training should remain fixed to YOLOv8s for this paper.

## Next Step: Prompt 6.5 Vast.ai Server Setup

Next planned step is server setup and sanity verification on Vast.ai with an RTX 3090 24GB instance.

Minimum checks before any full training:

1. Confirm CUDA and GPU visibility with `nvidia-smi`.
2. Create/activate the Python environment and install requirements.
3. Confirm `python3 scripts/smoke_test.py` passes.
4. Confirm dataset YAML paths exist on `/workspace/data/...`.
5. Run `scripts/launch_yolo_experiment.py --dry-run` and verify expected configs.
6. Run a tiny YOLO smoke/probe command only if explicitly requested.

Do not start full training until server verification passes.
