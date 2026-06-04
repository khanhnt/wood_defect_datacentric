#!/usr/bin/env python3
"""Run negative-aware VNWoodKnot evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent

if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from wood_defect_datacentric.evaluation.negative_aware_eval import (  # noqa: E402
    align_predictions,
    drop_nested_rows,
    evaluate_thresholds,
    load_predictions_jsonl,
    load_vnwoodknot_records,
    load_yolo_txt_predictions,
    predict_with_yolo_checkpoint,
    save_predictions_jsonl,
    write_csv,
    write_markdown_report,
    write_plots,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--predictions-jsonl", type=Path, help="Prediction JSONL with image_id, boxes, labels, scores.")
    source.add_argument("--predictions-dir", type=Path, help="YOLO txt prediction directory with class cx cy w h conf rows.")
    source.add_argument("--checkpoint", type=Path, help="YOLO checkpoint to run predictions before evaluation.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "processed" / "vnwoodknot_manifest.jsonl")
    parser.add_argument("--split", default="test")
    parser.add_argument("--experiment-name", default="negative_eval_sample")
    parser.add_argument("--class-names", nargs="+", default=["live_knot", "dead_knot"])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.10, 0.25, 0.50, 0.75])
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "results" / "negative_eval")
    parser.add_argument("--docs-output", type=Path, default=PROJECT_ROOT / "docs" / "negative_aware_evaluation.md")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--save-predictions", action="store_true", help="Save checkpoint-generated predictions JSONL.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = load_vnwoodknot_records(args.manifest, split=args.split, class_names=args.class_names)
    low_conf = min(float(threshold) for threshold in args.thresholds)

    if args.predictions_jsonl:
        predictions = load_predictions_jsonl(args.predictions_jsonl)
        prediction_source = str(args.predictions_jsonl)
    elif args.predictions_dir:
        predictions = load_yolo_txt_predictions(args.predictions_dir, records)
        prediction_source = str(args.predictions_dir)
    else:
        predictions = predict_with_yolo_checkpoint(
            args.checkpoint,
            records,
            imgsz=args.imgsz,
            device=args.device,
            batch=args.batch,
            max_detections=args.max_detections,
            low_conf=low_conf,
        )
        prediction_source = str(args.checkpoint)
        if args.save_predictions:
            save_predictions_jsonl(predictions, args.output_dir / f"{args.experiment_name}_predictions.jsonl")

    aligned_predictions, alignment_report = align_predictions(records, predictions)
    summary_rows, class_fp_rows, negative_image_rows = evaluate_thresholds(
        records,
        aligned_predictions,
        class_names=args.class_names,
        thresholds=args.thresholds,
        iou_threshold=args.iou_threshold,
    )

    summary_csv = args.output_dir / f"{args.experiment_name}_threshold_metrics.csv"
    class_fp_csv = args.output_dir / f"{args.experiment_name}_class_fp_counts.csv"
    negative_image_csv = args.output_dir / f"{args.experiment_name}_knot_free_image_counts.csv"
    write_csv(drop_nested_rows(summary_rows), summary_csv)
    write_csv(class_fp_rows, class_fp_csv)
    write_csv(negative_image_rows, negative_image_csv)
    write_markdown_report(
        output_path=args.docs_output,
        experiment_name=args.experiment_name,
        manifest_path=args.manifest,
        prediction_source=prediction_source,
        summary_rows=summary_rows,
        alignment_report=alignment_report,
    )
    plot_paths = write_plots(summary_rows, args.output_dir, args.experiment_name)

    print(f"Wrote: {summary_csv}")
    print(f"Wrote: {class_fp_csv}")
    print(f"Wrote: {negative_image_csv}")
    print(f"Wrote: {args.docs_output}")
    for path in plot_paths:
        print(f"Wrote: {path}")


if __name__ == "__main__":
    main()

