#!/usr/bin/env python3
"""Adapt Faster R-CNN detections to the frozen threshold-sweep JSON schema."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)


def xywh_to_xyxy(values: Sequence[float]) -> np.ndarray:
    x, y, width, height = (float(value) for value in values[:4])
    return np.asarray([x, y, x + width, y + height], dtype=np.float32)


def xyxy_to_xywh(values: Sequence[float]) -> list[float]:
    x1, y1, x2, y2 = (float(value) for value in values[:4])
    return [x1, y1, x2 - x1, y2 - y1]


def box_iou(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float32).reshape(-1, 4)
    predictions = np.asarray(predictions, dtype=np.float32).reshape(-1, 4)
    if not len(labels) or not len(predictions):
        return np.zeros((len(labels), len(predictions)), dtype=np.float32)
    top_left = np.maximum(labels[:, None, :2], predictions[None, :, :2])
    bottom_right = np.minimum(labels[:, None, 2:], predictions[None, :, 2:])
    widths_heights = np.clip(bottom_right - top_left, 0, None)
    intersection = widths_heights[..., 0] * widths_heights[..., 1]
    label_area = np.prod(labels[:, 2:] - labels[:, :2], axis=1)[:, None]
    prediction_area = np.prod(predictions[:, 2:] - predictions[:, :2], axis=1)[None, :]
    return intersection / np.maximum(label_area + prediction_area - intersection, 1e-16)


def ultralytics_validator_match(
    pred_classes: np.ndarray,
    true_classes: np.ndarray,
    iou: np.ndarray,
) -> np.ndarray:
    """Reproduce BaseValidator.match_predictions(use_scipy=False) from 8.4.60."""
    pred_classes = np.asarray(pred_classes)
    true_classes = np.asarray(true_classes)
    iou = np.asarray(iou, dtype=np.float32)
    correct = np.zeros((pred_classes.shape[0], len(IOU_THRESHOLDS)), dtype=bool)
    class_iou = iou * (true_classes[:, None] == pred_classes[None, :])
    for column, threshold in enumerate(IOU_THRESHOLDS):
        matches = np.asarray(np.nonzero(class_iou >= threshold)).T
        if not matches.shape[0]:
            continue
        if matches.shape[0] > 1:
            matches = matches[class_iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), column] = True
    return correct


def encode_tp_masks(correct: np.ndarray) -> list[int]:
    correct = np.asarray(correct, dtype=bool).reshape(-1, len(IOU_THRESHOLDS))
    return [
        sum(int(value) << bit for bit, value in enumerate(row))
        for row in correct
    ]


def adapt_fasterrcnn_image(
    *,
    image: str,
    canonical_id: str,
    image_path: str,
    width: int,
    height: int,
    gt_boxes_xyxy: np.ndarray,
    gt_class_ids: np.ndarray,
    pred_boxes_xyxy: np.ndarray,
    pred_scores: np.ndarray,
    pred_class_ids: np.ndarray,
    class_names: Sequence[str],
    model_label_to_class_id: Mapping[int, int] | None = None,
    min_confidence: float = 0.001,
) -> dict[str, Any]:
    """Convert one torchvision Faster R-CNN result to the frozen export schema."""
    gt_boxes_xyxy = np.asarray(gt_boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    gt_class_ids = np.asarray(gt_class_ids, dtype=np.int64).reshape(-1)
    pred_boxes_xyxy = np.asarray(pred_boxes_xyxy, dtype=np.float32).reshape(-1, 4)
    pred_scores = np.asarray(pred_scores, dtype=np.float32).reshape(-1)
    pred_class_ids = np.asarray(pred_class_ids, dtype=np.int64).reshape(-1)
    if model_label_to_class_id is None:
        model_label_to_class_id = {
            model_label: model_label - 1
            for model_label in range(1, len(class_names) + 1)
        }

    if not (len(gt_boxes_xyxy) == len(gt_class_ids)):
        raise ValueError("Ground-truth box and class counts differ.")
    if not (len(pred_boxes_xyxy) == len(pred_scores) == len(pred_class_ids)):
        raise ValueError("Prediction box, score, and class counts differ.")

    retained = pred_scores >= float(min_confidence)
    pred_boxes_xyxy = pred_boxes_xyxy[retained]
    pred_scores = pred_scores[retained]
    pred_class_ids = pred_class_ids[retained]
    try:
        pred_class_ids = np.asarray(
            [model_label_to_class_id[int(value)] for value in pred_class_ids],
            dtype=np.int64,
        )
    except KeyError as exc:
        raise ValueError(f"Unmapped Faster R-CNN model label: {exc.args[0]}") from exc
    if np.any(pred_class_ids < 0) or np.any(pred_class_ids >= len(class_names)):
        raise ValueError("Mapped prediction class ID is outside class_names.")

    iou = box_iou(gt_boxes_xyxy, pred_boxes_xyxy)
    correct = ultralytics_validator_match(pred_class_ids, gt_class_ids, iou)
    masks = encode_tp_masks(correct)
    predictions = []
    for prediction_index, (box, score, class_id, mask) in enumerate(
        zip(pred_boxes_xyxy, pred_scores, pred_class_ids, masks)
    ):
        class_id = int(class_id)
        row_ious = iou[:, prediction_index]
        predictions.append(
            {
                "bbox": xyxy_to_xywh(box),
                "conf": float(score),
                "class": class_names[class_id],
                "class_id": class_id,
                "max_iou_gt": float(np.max(row_ious)) if row_ious.size else 0.0,
                "validator_tp_mask": int(mask),
            }
        )

    gt_boxes = [
        [*xyxy_to_xywh(box), class_names[int(class_id)]]
        for box, class_id in zip(gt_boxes_xyxy, gt_class_ids)
    ]
    return {
        "image": image,
        "canonical_id": canonical_id,
        "image_path": image_path,
        "width": int(width),
        "height": int(height),
        "is_knot_free": not len(gt_boxes),
        "gt_boxes": gt_boxes,
        "predictions": predictions,
    }


def validate_saved_export(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    class_ids = {name: index for index, name in enumerate(payload["class_names"])}
    prediction_count = 0
    bit_count = 0
    exact_prediction_masks = 0
    exact_bits = 0
    exact_tp50 = 0
    clipped_predictions = 0
    clipped_mismatches = 0
    mismatch_examples: list[dict[str, Any]] = []

    for image_record in payload["images"]:
        gt_rows = image_record.get("gt_boxes", [])
        prediction_rows = image_record.get("predictions", [])
        gt_boxes = np.asarray([xywh_to_xyxy(row) for row in gt_rows], dtype=np.float32).reshape(-1, 4)
        gt_classes = np.asarray([class_ids[str(row[4])] for row in gt_rows], dtype=np.int64)
        pred_boxes = np.asarray(
            [xywh_to_xyxy(row["bbox"]) for row in prediction_rows], dtype=np.float32
        ).reshape(-1, 4)
        pred_classes = np.asarray([int(row["class_id"]) for row in prediction_rows], dtype=np.int64)
        recomputed = encode_tp_masks(
            ultralytics_validator_match(pred_classes, gt_classes, box_iou(gt_boxes, pred_boxes))
        )
        stored = [int(row["validator_tp_mask"]) for row in prediction_rows]
        clipped = [
            float(row["bbox"][0]) <= 1e-4
            or float(row["bbox"][1]) <= 1e-4
            or float(row["bbox"][0]) + float(row["bbox"][2]) >= float(image_record["width"]) - 1e-4
            or float(row["bbox"][1]) + float(row["bbox"][3]) >= float(image_record["height"]) - 1e-4
            for row in prediction_rows
        ]
        prediction_count += len(stored)
        bit_count += len(stored) * len(IOU_THRESHOLDS)
        exact_prediction_masks += sum(left == right for left, right in zip(stored, recomputed))
        exact_tp50 += sum((left & 1) == (right & 1) for left, right in zip(stored, recomputed))
        clipped_predictions += sum(clipped)
        clipped_mismatches += sum(
            is_clipped and left != right
            for is_clipped, left, right in zip(clipped, stored, recomputed)
        )
        exact_bits += sum(
            ((left >> bit) & 1) == ((right >> bit) & 1)
            for left, right in zip(stored, recomputed)
            for bit in range(len(IOU_THRESHOLDS))
        )
        if stored != recomputed and len(mismatch_examples) < 10:
            for index, (left, right) in enumerate(zip(stored, recomputed)):
                if left != right:
                    mismatch_examples.append(
                        {
                            "image": image_record.get("canonical_id", image_record.get("image")),
                            "prediction_index": index,
                            "stored": left,
                            "recomputed": right,
                        }
                    )
                    break

    return {
        "path": str(path.resolve()),
        "dataset": payload.get("dataset"),
        "variant": payload.get("variant"),
        "seed": payload.get("seed"),
        "split": payload.get("split"),
        "num_images": len(payload["images"]),
        "num_predictions": prediction_count,
        "exact_prediction_masks": exact_prediction_masks,
        "prediction_mask_exact_rate": exact_prediction_masks / prediction_count if prediction_count else 1.0,
        "exact_bits": exact_bits,
        "bit_exact_rate": exact_bits / bit_count if bit_count else 1.0,
        "exact_tp50": exact_tp50,
        "tp50_exact_rate": exact_tp50 / prediction_count if prediction_count else 1.0,
        "clipped_predictions": clipped_predictions,
        "clipped_mismatches": clipped_mismatches,
        "mismatch_examples": mismatch_examples,
        "status": "PASS" if exact_prediction_masks == prediction_count else "FAIL",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-yolo-export", type=Path, nargs="+", required=True)
    parser.add_argument("--output-report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports = [validate_saved_export(path.expanduser().resolve()) for path in args.validate_yolo_export]
    summary = {
        "schema_note": "Frozen gt_boxes and prediction bbox fields use [x, y, width, height].",
        "matching_source": "Ultralytics 8.4.60 BaseValidator.match_predictions(use_scipy=False)",
        "iou_thresholds": IOU_THRESHOLDS.tolist(),
        "files": reports,
        "num_files": len(reports),
        "num_images": sum(int(report["num_images"]) for report in reports),
        "num_predictions": sum(int(report["num_predictions"]) for report in reports),
        "exact_prediction_masks": sum(int(report["exact_prediction_masks"]) for report in reports),
        "exact_tp50": sum(int(report["exact_tp50"]) for report in reports),
        "clipped_predictions": sum(int(report["clipped_predictions"]) for report in reports),
        "clipped_mismatches": sum(int(report["clipped_mismatches"]) for report in reports),
    }
    total = int(summary["num_predictions"])
    exact = int(summary["exact_prediction_masks"])
    summary["prediction_mask_exact_rate"] = exact / total if total else 1.0
    summary["tp50_exact_rate"] = int(summary["exact_tp50"]) / total if total else 1.0
    summary["status"] = "PASS" if all(report["status"] == "PASS" for report in reports) else "FAIL"
    if args.output_report:
        output_path = args.output_report.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote: {output_path}")
    print(
        f"ADAPTER VALIDATION: {summary['status']} files={summary['num_files']} "
        f"images={summary['num_images']} predictions={total} exact={exact}/{total} "
        f"rate={summary['prediction_mask_exact_rate']:.9f}"
    )
    if summary["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
