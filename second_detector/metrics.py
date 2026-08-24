"""Ultralytics-8.4.60-compatible matching and AP aggregation."""

from __future__ import annotations

from typing import Any

import numpy as np


IOU_THRESHOLDS = np.linspace(0.50, 0.95, 10)


def match_predictions(pred_classes: np.ndarray, true_classes: np.ndarray, iou: np.ndarray) -> np.ndarray:
    pred_classes = np.asarray(pred_classes)
    true_classes = np.asarray(true_classes)
    class_iou = np.asarray(iou, dtype=np.float32) * (true_classes[:, None] == pred_classes[None, :])
    correct = np.zeros((len(pred_classes), len(IOU_THRESHOLDS)), dtype=bool)
    for column, threshold in enumerate(IOU_THRESHOLDS):
        matches = np.asarray(np.nonzero(class_iou >= threshold)).T
        if not len(matches):
            continue
        if len(matches) > 1:
            matches = matches[class_iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), column] = True
    return correct


def box_iou_numpy(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float32).reshape(-1, 4)
    predictions = np.asarray(predictions, dtype=np.float32).reshape(-1, 4)
    if not len(labels) or not len(predictions):
        return np.zeros((len(labels), len(predictions)), dtype=np.float32)
    top_left = np.maximum(labels[:, None, :2], predictions[None, :, :2])
    bottom_right = np.minimum(labels[:, None, 2:], predictions[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0, None)
    intersection = wh[..., 0] * wh[..., 1]
    label_area = np.prod(np.clip(labels[:, 2:] - labels[:, :2], 0, None), axis=1)[:, None]
    pred_area = np.prod(np.clip(predictions[:, 2:] - predictions[:, :2], 0, None), axis=1)[None, :]
    return intersection / np.maximum(label_area + pred_area - intersection, 1e-16)


def encode_tp_masks(correct: np.ndarray) -> list[int]:
    return [sum(int(value) << bit for bit, value in enumerate(row)) for row in np.asarray(correct, dtype=bool)]


def summarize_stats(stats: list[dict[str, np.ndarray]], class_names: list[str]) -> dict[str, Any]:
    from ultralytics.utils.metrics import ap_per_class

    correct = np.concatenate([row["correct"] for row in stats], axis=0) if stats else np.zeros((0, 10), bool)
    confidence = np.concatenate([row["confidence"] for row in stats]) if stats else np.zeros(0)
    pred_class = np.concatenate([row["pred_class"] for row in stats]) if stats else np.zeros(0)
    target_class = np.concatenate([row["target_class"] for row in stats]) if stats else np.zeros(0)
    result = ap_per_class(
        correct,
        confidence,
        pred_class,
        target_class,
        plot=False,
        names={index: name for index, name in enumerate(class_names)},
    )
    _, _, precision, recall, _, ap, unique_classes, *_ = result
    per_class = []
    for index, class_id in enumerate(unique_classes):
        per_class.append(
            {
                "class_id": int(class_id),
                "class_name": class_names[int(class_id)],
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "mAP50": float(ap[index, 0]),
                "mAP50_95": float(ap[index].mean()),
                "instances": int((target_class == class_id).sum()),
            }
        )
    return {
        "precision": float(np.mean(precision)) if len(precision) else 0.0,
        "recall": float(np.mean(recall)) if len(recall) else 0.0,
        "mAP50": float(ap[:, 0].mean()) if len(ap) else 0.0,
        "mAP50_95": float(ap.mean()) if len(ap) else 0.0,
        "n_predictions": int(len(confidence)),
        "n_instances": int(len(target_class)),
        "per_class": per_class,
    }
