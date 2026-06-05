# VNWoodKnot Experiment Summary

Four YOLOv8s fixed-detector VNWoodKnot experiments have completed: baseline, P2 illumination normalization, A1 defect-preserving crop, and P4+A4 combined safe preprocessing/augmentation.

## Validation Summary

| Variant | Best epoch | Val P | Val R | Val mAP50 | Val mAP50-95 |
|---|---:|---:|---:|---:|---:|
| Baseline | 50 | 0.810 | 0.823 | 0.813 | 0.458 |
| P2 illumination | 45 | 0.772 | 0.793 | 0.837 | 0.476 |
| A1 crop | 45 | 0.861 | 0.904 | 0.917 | 0.494 |
| P4+A4 combined | 42 | 0.860 | 0.859 | 0.911 | 0.505 |

## Negative-Aware Test Summary At Threshold 0.25

| Variant | Test AP50 | Precision | Recall | FP knot-free images | FP rate | Mean FP/knot-free |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.779 | 0.614 | 0.871 | 1/75 | 0.013 | 0.013 |
| P2 illumination | 0.760 | 0.581 | 0.858 | 0/75 | 0.000 | 0.000 |
| A1 crop | 0.664 | 0.559 | 0.761 | 4/75 | 0.053 | 0.067 |
| P4+A4 combined | 0.728 | 0.612 | 0.794 | 0/75 | 0.000 | 0.000 |

## Negative-Aware Test Summary At Threshold 0.50

| Variant | Test AP50 | Precision | Recall | FP knot-free images | FP rate | Mean FP/knot-free |
|---|---:|---:|---:|---:|---:|---:|
| Baseline | 0.507 | 0.847 | 0.535 | 0/75 | 0.000 | 0.000 |
| P2 illumination | 0.550 | 0.856 | 0.574 | 0/75 | 0.000 | 0.000 |
| A1 crop | 0.504 | 0.816 | 0.542 | 0/75 | 0.000 | 0.000 |
| P4+A4 combined | 0.621 | 0.824 | 0.665 | 0/75 | 0.000 | 0.000 |

## Current Interpretation

- P4+A4 and A1 give the strongest validation mAP, but validation alone is not enough for this study.
- P2 is the cleanest operating point at confidence threshold 0.25: zero false-positive knot-free images with only a modest AP/recall trade-off relative to baseline.
- A1 improves validation metrics but substantially worsens negative-aware behavior at low and moderate thresholds, showing why negative-only evaluation is necessary.
- P4+A4 is attractive at stricter threshold 0.50, where it has the best AP50/recall among zero-FP variants.

## Files

- Aggregate summary CSV: `results/summaries/vn_experiment_comparison_summary.csv`
- Long threshold CSV: `results/summaries/vn_negative_eval_all_thresholds_long.csv`
