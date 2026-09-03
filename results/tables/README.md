# Revision result artifacts

This directory contains the lightweight numerical artifacts used by the revised
IEEE Access manuscript. Results are separated by detector family so that absolute
values and protocols are not mixed across model blocks.

## YOLOv8s

The authoritative YOLOv8s artifacts are in `yolov8s/` and were derived from the
frozen `access_r1_g2` generation.

- `fair_metrics_summary.csv` and `fair_metrics_per_seed.csv`: standard fair-test
  precision, recall, mAP50, and mAP50-95 on the common non-augmented evaluation
  splits.
- `validation_threshold_selection.csv`: thresholds selected on validation clean
  images for each false-positive tolerance.
- `locked_test_operating_points_*.csv`: test results obtained by applying the
  validation-selected thresholds unchanged.
- `locked_test_sensitivity_summary.csv`: the full tolerance sensitivity analysis.
- `clean_fp_sweep_*.csv`: clean-image false-positive rates and FPPI inputs.
- `calibration_*.csv`, `clean_max_confidence_*.csv`, and
  `reliability_bins.csv`: calibration and clean-confidence analyses.
- `reviewer_audits/`: exact-binomial intervals, source-level VSB analyses,
  convergence diagnostics, retained-AP characterization, and deprecated-checkpoint
  comparisons used in the revision and response letter.
- `latex/`: generated table rows retained for direct manuscript cross-checking.

## Faster R-CNN

The second-detector robustness artifacts are in `fasterrcnn/` and were produced by
the `access_r1_g3_fasterrcnn` generation.

- `standard/`: per-seed and aggregate validation/test metrics plus checkpoint and
  prediction audits.
- `negative_aware/`: validation-selected thresholds, test operating metrics,
  tolerance sweeps, and threshold-stability audits.
- `provenance/`: runtime, code-commit, verification-gate, and job-manifest records.

## Superseded artifacts

Files previously stored directly under `results/tables/` were moved to
`results/_deprecated/pre_revision_tables/`. They are preserved for audit history but
must not be used to reproduce the revised manuscript.
