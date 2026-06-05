# Negative-Aware Evaluation

This evaluation reports AP-style detection metrics together with false-positive behavior on VNWoodKnot `knot_free` images. Thresholds are reported transparently and are not tuned to hide failure cases.

- Experiment: `vn_yolov8s_p4_a4_combined_e50_negative_eval`
- Manifest: `/workspace/data/vnwoodknot/benchmarks/vnwoodknot_live_dead_2class_yolo/vnwoodknot_existing_images_manifest.jsonl`
- Prediction source: `results/runs/vn_yolov8s_p4_a4_combined_e50/ultralytics/train/weights/best.pt`
- Input prediction records: 229
- Aligned image records: 229
- Unmatched prediction records: 0

## Threshold Summary

| Conf | AP50 | mAP50-95 | Precision | Recall | FP knot-free images | FP image rate | Pred/knot-free | Mean FP conf |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.10 | 0.7567 | 0.3850 | 0.4577 | 0.8387 | 2 | 0.0267 | 0.0267 | 0.1157 |
| 0.25 | 0.7282 | 0.3655 | 0.6119 | 0.7935 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 0.50 | 0.6206 | 0.3215 | 0.8240 | 0.6645 | 0 | 0.0000 | 0.0000 | 0.0000 |
| 0.75 | 0.1053 | 0.0843 | 1.0000 | 0.1032 | 0 | 0.0000 | 0.0000 | 0.0000 |

## Interpretation Notes

- `false_positive_images_knot_free` counts negative-only `knot_free` images with at least one prediction above the threshold.
- `mean_predictions_per_knot_free_image` penalizes multiple hallucinated boxes on a single negative image.
- AP50 and mAP50-95 are computed on the full evaluated split at the same confidence threshold for threshold-sensitivity analysis.
- Lower thresholds are expected to increase recall and may increase false positives on `knot_free` wood texture.
