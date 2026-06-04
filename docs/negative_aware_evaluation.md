# Negative-Aware Evaluation

This evaluation reports AP-style detection metrics together with false-positive behavior on VNWoodKnot `knot_free` images. Thresholds are reported transparently and are not tuned to hide failure cases.

- Experiment: `t0_yolov8s_vnwoodknot_target_only_e50_negative_eval`
- Manifest: `/Users/ntkhanh/PycharmProjects/wood_defect_datacentric/data/processed/vnwoodknot_manifest.jsonl`
- Prediction source: `artifacts/vnwoodknot/tables/t0_yolov8s_vnwoodknot_target_only_e50_eval_test_predictions.jsonl`
- Input prediction records: 229
- Aligned image records: 229
- Unmatched prediction records: 0

## Threshold Summary

| Conf | AP50 | mAP50-95 | Precision | Recall | FP knot-free images | FP image rate | Pred/knot-free | Mean FP conf |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.7721 | 0.3884 | 0.5696 | 0.8710 | 5 | 0.0667 | 0.0667 | 0.1326 |
| 0.25 | 0.7280 | 0.3641 | 0.7911 | 0.8065 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 0.50 | 0.3284 | 0.2148 | 0.9091 | 0.3226 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 0.75 | 0.1382 | 0.1069 | 1.0000 | 0.1355 | 0 | 0.0000 | 0.0000 | 0.0000 |

## Interpretation Notes

- `false_positive_images_knot_free` counts negative-only `knot_free` images with at least one prediction above the threshold.
- `mean_predictions_per_knot_free_image` penalizes multiple hallucinated boxes on a single negative image.
- AP50 and mAP50-95 are computed on the full evaluated split at the same confidence threshold for threshold-sensitivity analysis.
- Lower thresholds are expected to increase recall and may increase false positives on `knot_free` wood texture.
