"""Negative-aware object-detection evaluation for VNWoodKnot.

The evaluator is read-only with respect to model predictions. It can score
existing prediction JSONL files, YOLO txt prediction folders, or run a YOLO
checkpoint when requested by the wrapper script.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
import csv
import json
import math
import os
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import numpy as np
from PIL import Image


SPLIT_ALIASES = {"validation": "val", "valid": "val"}
IMAGE_STEM_RE = re.compile(r"(img[_-]?\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class EvalRecord:
    image_id: str
    canonical_id: str
    image_path: Path
    split: str
    source_category: str | None
    width: int
    height: int
    boxes: np.ndarray
    labels: np.ndarray

    @property
    def is_negative(self) -> bool:
        return self.source_category == "knot_free" or len(self.labels) == 0


@dataclass(frozen=True)
class Prediction:
    image_id: str
    canonical_id: str
    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray


def normalize_split(split: Any) -> str:
    if split is None:
        return "unspecified"
    value = str(split).strip().lower()
    return SPLIT_ALIASES.get(value, value)


def canonical_image_id(value: str | Path) -> str:
    text = str(value).replace("\\", "/")
    match = IMAGE_STEM_RE.search(text)
    if match:
        return match.group(1).lower().replace("-", "_")
    if "__" in text:
        return text.split("__")[-1].lower()
    return Path(text).stem.lower()


def load_vnwoodknot_records(
    manifest_path: Path,
    *,
    split: str,
    class_names: Sequence[str],
) -> list[EvalRecord]:
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    wanted_split = normalize_split(split)
    records: list[EvalRecord] = []
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            raw = json.loads(line)
            if normalize_split(raw.get("split")) != wanted_split:
                continue
            width = int(raw.get("width") or 0)
            height = int(raw.get("height") or 0)
            boxes: list[list[float]] = []
            labels: list[int] = []
            for annotation in raw.get("annotations") or []:
                class_name = str(annotation.get("class_name"))
                if class_name not in class_to_id:
                    continue
                x1, y1, x2, y2 = [float(v) for v in annotation["bbox_xyxy_norm"]]
                boxes.append([x1 * width, y1 * height, x2 * width, y2 * height])
                labels.append(class_to_id[class_name])
            image_id = str(raw["image_id"])
            records.append(
                EvalRecord(
                    image_id=image_id,
                    canonical_id=canonical_image_id(image_id),
                    image_path=Path(str(raw.get("image_path", ""))).expanduser(),
                    split=wanted_split,
                    source_category=raw.get("source_category"),
                    width=width,
                    height=height,
                    boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                    labels=np.asarray(labels, dtype=np.int64),
                )
            )
    if not records:
        raise ValueError(f"No records found for split={split!r} in {manifest_path}")
    _assert_unique_record_keys(records)
    return records


def _assert_unique_record_keys(records: Sequence[EvalRecord]) -> None:
    counts = Counter(record.canonical_id for record in records)
    duplicates = [key for key, count in counts.items() if count > 1]
    if duplicates:
        raise ValueError(f"Duplicate canonical image IDs found: {duplicates[:10]}")


def load_predictions_jsonl(path: Path) -> list[Prediction]:
    predictions: list[Prediction] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            raw = json.loads(line)
            image_id = str(raw["image_id"])
            boxes = np.asarray(raw.get("boxes") or [], dtype=np.float32).reshape(-1, 4)
            labels = np.asarray(raw.get("labels") or [], dtype=np.int64)
            scores = np.asarray(raw.get("scores") or [], dtype=np.float32)
            predictions.append(
                Prediction(
                    image_id=image_id,
                    canonical_id=canonical_image_id(image_id),
                    boxes=boxes,
                    labels=labels,
                    scores=scores,
                )
            )
    return predictions


def load_yolo_txt_predictions(predictions_dir: Path, records: Sequence[EvalRecord]) -> list[Prediction]:
    record_by_key = {record.canonical_id: record for record in records}
    predictions: list[Prediction] = []
    for txt_path in sorted(predictions_dir.glob("*.txt")):
        key = canonical_image_id(txt_path.stem)
        record = record_by_key.get(key)
        if record is None:
            continue
        boxes: list[list[float]] = []
        labels: list[int] = []
        scores: list[float] = []
        with txt_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                parts = line.strip().split()
                if len(parts) < 6:
                    continue
                class_id = int(float(parts[0]))
                cx, cy, bw, bh, score = [float(v) for v in parts[1:6]]
                x1 = (cx - bw / 2.0) * record.width
                y1 = (cy - bh / 2.0) * record.height
                x2 = (cx + bw / 2.0) * record.width
                y2 = (cy + bh / 2.0) * record.height
                boxes.append([x1, y1, x2, y2])
                labels.append(class_id)
                scores.append(score)
        predictions.append(
            Prediction(
                image_id=txt_path.stem,
                canonical_id=key,
                boxes=np.asarray(boxes, dtype=np.float32).reshape(-1, 4),
                labels=np.asarray(labels, dtype=np.int64),
                scores=np.asarray(scores, dtype=np.float32),
            )
        )
    return predictions


def predict_with_yolo_checkpoint(
    checkpoint: Path,
    records: Sequence[EvalRecord],
    *,
    imgsz: int,
    device: str,
    batch: int,
    max_detections: int,
    low_conf: float,
) -> list[Prediction]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required for checkpoint-based negative-aware evaluation.") from exc

    model = YOLO(str(checkpoint))
    predictions: list[Prediction] = []
    batch = max(1, int(batch))
    for start in range(0, len(records), batch):
        record_batch = records[start : start + batch]
        path_batch = [str(record.image_path) for record in record_batch]
        results = model.predict(
            source=path_batch,
            stream=True,
            imgsz=int(imgsz),
            conf=float(low_conf),
            max_det=int(max_detections),
            device=str(device),
            batch=len(path_batch),
            verbose=False,
        )
        for record, result in zip(record_batch, results):
            boxes = result.boxes
            if boxes is None:
                xyxy = np.zeros((0, 4), dtype=np.float32)
                scores = np.zeros((0,), dtype=np.float32)
                labels = np.zeros((0,), dtype=np.int64)
            else:
                xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32).reshape(-1, 4)
                scores = boxes.conf.detach().cpu().numpy().astype(np.float32)
                labels = boxes.cls.detach().cpu().numpy().astype(np.int64)
            predictions.append(
                Prediction(
                    image_id=record.image_id,
                    canonical_id=record.canonical_id,
                    boxes=xyxy,
                    labels=labels,
                    scores=scores,
                )
            )
    return predictions


def align_predictions(
    records: Sequence[EvalRecord],
    predictions: Sequence[Prediction],
) -> tuple[list[Prediction], dict[str, int]]:
    record_keys = {record.canonical_id for record in records}
    pred_by_key: dict[str, Prediction] = {}
    duplicate_count = 0
    unmatched_count = 0
    for prediction in predictions:
        if prediction.canonical_id not in record_keys:
            unmatched_count += 1
            continue
        if prediction.canonical_id in pred_by_key:
            duplicate_count += 1
            pred_by_key[prediction.canonical_id] = merge_predictions(pred_by_key[prediction.canonical_id], prediction)
        else:
            pred_by_key[prediction.canonical_id] = prediction
    aligned: list[Prediction] = []
    for record in records:
        aligned.append(
            pred_by_key.get(
                record.canonical_id,
                Prediction(
                    image_id=record.image_id,
                    canonical_id=record.canonical_id,
                    boxes=np.zeros((0, 4), dtype=np.float32),
                    labels=np.zeros((0,), dtype=np.int64),
                    scores=np.zeros((0,), dtype=np.float32),
                ),
            )
        )
    return aligned, {
        "num_input_predictions": len(predictions),
        "num_aligned_predictions": len(aligned),
        "num_unmatched_prediction_records": unmatched_count,
        "num_duplicate_prediction_records": duplicate_count,
    }


def merge_predictions(first: Prediction, second: Prediction) -> Prediction:
    return Prediction(
        image_id=first.image_id,
        canonical_id=first.canonical_id,
        boxes=np.concatenate([first.boxes, second.boxes], axis=0),
        labels=np.concatenate([first.labels, second.labels], axis=0),
        scores=np.concatenate([first.scores, second.scores], axis=0),
    )


def evaluate_thresholds(
    records: Sequence[EvalRecord],
    predictions: Sequence[Prediction],
    *,
    class_names: Sequence[str],
    thresholds: Sequence[float],
    iou_threshold: float = 0.50,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary_rows: list[dict[str, Any]] = []
    class_fp_rows: list[dict[str, Any]] = []
    negative_image_rows: list[dict[str, Any]] = []
    negative_records = [record for record in records if record.source_category == "knot_free" or record.is_negative]

    for threshold in thresholds:
        filtered = filter_predictions(predictions, threshold)
        standard = compute_standard_metrics(
            records=records,
            predictions=filtered,
            class_names=class_names,
            score_threshold=threshold,
            iou_threshold=iou_threshold,
        )
        negative = compute_negative_metrics(
            negative_records=negative_records,
            predictions=filtered,
            class_names=class_names,
            threshold=threshold,
        )
        summary_rows.append({**standard, **negative})
        class_fp_rows.extend(negative["class_specific_rows"])
        negative_image_rows.extend(negative["per_negative_image_rows"])
    return summary_rows, class_fp_rows, negative_image_rows


def filter_predictions(predictions: Sequence[Prediction], threshold: float) -> list[Prediction]:
    filtered: list[Prediction] = []
    for prediction in predictions:
        keep = prediction.scores >= float(threshold)
        filtered.append(
            Prediction(
                image_id=prediction.image_id,
                canonical_id=prediction.canonical_id,
                boxes=prediction.boxes[keep],
                labels=prediction.labels[keep],
                scores=prediction.scores[keep],
            )
        )
    return filtered


def compute_standard_metrics(
    *,
    records: Sequence[EvalRecord],
    predictions: Sequence[Prediction],
    class_names: Sequence[str],
    score_threshold: float,
    iou_threshold: float,
) -> dict[str, Any]:
    iou_thresholds = [round(v, 2) for v in np.arange(0.50, 1.00, 0.05)]
    ap_by_class_iou: dict[tuple[int, float], float] = {}
    precision_parts: list[float] = []
    recall_parts: list[float] = []
    tp_total = 0
    fp_total = 0
    fn_total = 0
    per_class_rows = []

    for class_id, class_name in enumerate(class_names):
        for iou in iou_thresholds:
            result = evaluate_class(records, predictions, class_id=class_id, iou_threshold=iou)
            ap_by_class_iou[(class_id, iou)] = result["ap"]
            if math.isclose(iou, iou_threshold):
                tp_total += int(result["tp"])
                fp_total += int(result["fp"])
                fn_total += int(result["fn"])
                precision_parts.append(float(result["precision"]))
                recall_parts.append(float(result["recall"]))
                per_class_rows.append(
                    {
                        "class_name": class_name,
                        "tp": int(result["tp"]),
                        "fp": int(result["fp"]),
                        "fn": int(result["fn"]),
                        "ap50": result["ap"],
                    }
                )

    ap50_values = [
        ap
        for (class_id, iou), ap in ap_by_class_iou.items()
        if math.isclose(iou, 0.50) and not math.isnan(ap)
    ]
    ap5095_values = [ap for ap in ap_by_class_iou.values() if not math.isnan(ap)]
    precision = tp_total / max(tp_total + fp_total, 1)
    recall = tp_total / max(tp_total + fn_total, 1)
    return {
        "threshold": float(score_threshold),
        "AP50": round(float(np.mean(ap50_values)), 6) if ap50_values else 0.0,
        "mAP50_95": round(float(np.mean(ap5095_values)), 6) if ap5095_values else 0.0,
        "precision": round(float(precision), 6),
        "recall": round(float(recall), 6),
        "tp50": int(tp_total),
        "fp50": int(fp_total),
        "fn50": int(fn_total),
        "num_images": int(len(records)),
        "num_predictions": int(sum(len(prediction.scores) for prediction in predictions)),
        "num_targets": int(sum(len(record.labels) for record in records)),
        "per_class_standard": per_class_rows,
    }


def evaluate_class(
    records: Sequence[EvalRecord],
    predictions: Sequence[Prediction],
    *,
    class_id: int,
    iou_threshold: float,
) -> dict[str, float]:
    gt_by_image: dict[str, np.ndarray] = {}
    matched_by_image: dict[str, np.ndarray] = {}
    num_gt = 0
    for record in records:
        class_boxes = record.boxes[record.labels == class_id]
        gt_by_image[record.canonical_id] = class_boxes
        matched_by_image[record.canonical_id] = np.zeros((len(class_boxes),), dtype=bool)
        num_gt += len(class_boxes)

    pred_rows: list[tuple[str, float, np.ndarray]] = []
    for prediction in predictions:
        keep = prediction.labels == class_id
        for box, score in zip(prediction.boxes[keep], prediction.scores[keep]):
            pred_rows.append((prediction.canonical_id, float(score), box))
    pred_rows.sort(key=lambda item: item[1], reverse=True)

    if not pred_rows:
        return {
            "ap": float("nan") if num_gt == 0 else 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "tp": 0.0,
            "fp": 0.0,
            "fn": float(num_gt),
        }

    tp = np.zeros((len(pred_rows),), dtype=np.float32)
    fp = np.zeros((len(pred_rows),), dtype=np.float32)
    for index, (image_key, _, pred_box) in enumerate(pred_rows):
        gt_boxes = gt_by_image.get(image_key, np.zeros((0, 4), dtype=np.float32))
        if len(gt_boxes) == 0:
            fp[index] = 1.0
            continue
        ious = box_iou(pred_box[None, :], gt_boxes)[0]
        best_idx = int(np.argmax(ious))
        best_iou = float(ious[best_idx])
        if best_iou >= iou_threshold and not matched_by_image[image_key][best_idx]:
            matched_by_image[image_key][best_idx] = True
            tp[index] = 1.0
        else:
            fp[index] = 1.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    precision_curve = tp_cum / np.clip(tp_cum + fp_cum, 1e-8, None)
    recall_curve = tp_cum / max(float(num_gt), 1e-8)
    ap = average_precision(recall_curve, precision_curve) if num_gt > 0 else float("nan")
    tp_total = float(tp.sum())
    fp_total = float(fp.sum())
    fn_total = float(max(num_gt - tp_total, 0.0))
    return {
        "ap": float(ap),
        "precision": float(tp_total / max(tp_total + fp_total, 1e-8)),
        "recall": float(tp_total / max(float(num_gt), 1e-8)) if num_gt > 0 else 0.0,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
    }


def average_precision(recall: np.ndarray, precision: np.ndarray) -> float:
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for index in range(len(mpre) - 1, 0, -1):
        mpre[index - 1] = max(mpre[index - 1], mpre[index])
    change = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[change + 1] - mrec[change]) * mpre[change + 1]))


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    top_left = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], 0.0, None) * np.clip(boxes1[:, 3] - boxes1[:, 1], 0.0, None)
    area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], 0.0, None) * np.clip(boxes2[:, 3] - boxes2[:, 1], 0.0, None)
    union = np.clip(area1[:, None] + area2[None, :] - inter, 1e-8, None)
    return inter / union


def compute_negative_metrics(
    *,
    negative_records: Sequence[EvalRecord],
    predictions: Sequence[Prediction],
    class_names: Sequence[str],
    threshold: float,
) -> dict[str, Any]:
    prediction_by_key = {prediction.canonical_id: prediction for prediction in predictions}
    fp_image_count = 0
    pred_counts: list[int] = []
    false_confidences: list[float] = []
    class_counter: Counter[str] = Counter()
    per_image_rows: list[dict[str, Any]] = []

    for record in negative_records:
        prediction = prediction_by_key.get(record.canonical_id)
        if prediction is None:
            pred_count = 0
            scores = np.zeros((0,), dtype=np.float32)
            labels = np.zeros((0,), dtype=np.int64)
        else:
            pred_count = int(len(prediction.scores))
            scores = prediction.scores
            labels = prediction.labels
        pred_counts.append(pred_count)
        if pred_count > 0:
            fp_image_count += 1
            false_confidences.extend(float(score) for score in scores)
            for class_id in labels:
                class_name = class_names[int(class_id)] if 0 <= int(class_id) < len(class_names) else f"class_{int(class_id)}"
                class_counter[class_name] += 1
        per_image_rows.append(
            {
                "threshold": float(threshold),
                "image_id": record.image_id,
                "canonical_id": record.canonical_id,
                "source_category": record.source_category,
                "prediction_count": pred_count,
                "max_confidence": round(float(np.max(scores)), 6) if len(scores) else 0.0,
                "mean_confidence": round(float(np.mean(scores)), 6) if len(scores) else 0.0,
            }
        )

    num_negative = len(negative_records)
    class_rows = [
        {
            "threshold": float(threshold),
            "class_name": class_name,
            "false_positive_count_on_knot_free": int(count),
        }
        for class_name, count in sorted(class_counter.items())
    ]
    if not class_rows:
        class_rows.append(
            {
                "threshold": float(threshold),
                "class_name": "",
                "false_positive_count_on_knot_free": 0,
            }
        )
    return {
        "num_knot_free_images": int(num_negative),
        "false_positive_images_knot_free": int(fp_image_count),
        "false_positive_image_rate_knot_free": round(float(fp_image_count / max(num_negative, 1)), 6),
        "mean_predictions_per_knot_free_image": round(float(np.mean(pred_counts)), 6) if pred_counts else 0.0,
        "mean_confidence_false_predictions_knot_free": round(float(np.mean(false_confidences)), 6)
        if false_confidences
        else 0.0,
        "class_specific_rows": class_rows,
        "per_negative_image_rows": per_image_rows,
    }


def drop_nested_rows(summary_rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in summary_rows:
        rows.append({key: value for key, value in row.items() if key not in {"per_class_standard", "class_specific_rows", "per_negative_image_rows"}})
    return rows


def write_csv(rows: Sequence[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_report(
    *,
    output_path: Path,
    experiment_name: str,
    manifest_path: Path,
    prediction_source: str,
    summary_rows: Sequence[dict[str, Any]],
    alignment_report: dict[str, int],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    flat_rows = drop_nested_rows(summary_rows)
    lines = [
        "# Negative-Aware Evaluation",
        "",
        "This evaluation reports AP-style detection metrics together with false-positive behavior on VNWoodKnot `knot_free` images. Thresholds are reported transparently and are not tuned to hide failure cases.",
        "",
        f"- Experiment: `{experiment_name}`",
        f"- Manifest: `{manifest_path}`",
        f"- Prediction source: `{prediction_source}`",
        f"- Input prediction records: {alignment_report.get('num_input_predictions', 0)}",
        f"- Aligned image records: {alignment_report.get('num_aligned_predictions', 0)}",
        f"- Unmatched prediction records: {alignment_report.get('num_unmatched_prediction_records', 0)}",
        "",
        "## Threshold Summary",
        "",
        "| Conf | AP50 | mAP50-95 | Precision | Recall | FP knot-free images | FP image rate | Pred/knot-free | Mean FP conf |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in flat_rows:
        lines.append(
            "| "
            f"{row['threshold']:.2f} | {row['AP50']:.4f} | {row['mAP50_95']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | "
            f"{row['false_positive_images_knot_free']} | {row['false_positive_image_rate_knot_free']:.4f} | "
            f"{row['mean_predictions_per_knot_free_image']:.4f} | {row['mean_confidence_false_predictions_knot_free']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Notes",
            "",
            "- `false_positive_images_knot_free` counts negative-only `knot_free` images with at least one prediction above the threshold.",
            "- `mean_predictions_per_knot_free_image` penalizes multiple hallucinated boxes on a single negative image.",
            "- AP50 and mAP50-95 are computed on the full evaluated split at the same confidence threshold for threshold-sensitivity analysis.",
            "- Lower thresholds are expected to increase recall and may increase false positives on `knot_free` wood texture.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_plots(summary_rows: Sequence[dict[str, Any]], output_dir: Path, experiment_name: str) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_root = Path(os.environ.get("TMPDIR", "/tmp"))
    cache_dir = tmp_root / "wood_defect_datacentric_mpl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    thresholds = [float(row["threshold"]) for row in summary_rows]
    fp_per_image = [float(row["mean_predictions_per_knot_free_image"]) for row in summary_rows]
    precision = [float(row["precision"]) for row in summary_rows]
    recall = [float(row["recall"]) for row in summary_rows]
    ap50 = [float(row["AP50"]) for row in summary_rows]
    paths: list[Path] = []

    def save_current(name: str) -> None:
        path = output_dir / f"{experiment_name}_{name}.png"
        plt.tight_layout()
        plt.savefig(path, dpi=180)
        plt.close()
        paths.append(path)

    plt.figure(figsize=(5.5, 3.4))
    plt.plot(thresholds, fp_per_image, marker="o", color="#a23b2a")
    plt.xlabel("Confidence threshold")
    plt.ylabel("Mean predictions per knot-free image")
    plt.title("Threshold vs FP/image on knot-free")
    plt.grid(alpha=0.25)
    save_current("threshold_vs_fp_per_image")

    plt.figure(figsize=(5.5, 3.4))
    plt.plot(thresholds, precision, marker="o", label="Precision", color="#2f6f9f")
    plt.plot(thresholds, recall, marker="s", label="Recall", color="#3f8f45")
    plt.xlabel("Confidence threshold")
    plt.ylabel("Metric")
    plt.title("Threshold vs precision/recall")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    plt.legend()
    save_current("threshold_vs_precision_recall")

    plt.figure(figsize=(5.5, 3.4))
    plt.plot(thresholds, ap50, marker="o", color="#6b5b95")
    plt.xlabel("Confidence threshold")
    plt.ylabel("AP50")
    plt.title("Threshold vs AP50")
    plt.ylim(0, 1.02)
    plt.grid(alpha=0.25)
    save_current("threshold_vs_ap50")
    return paths


def save_predictions_jsonl(predictions: Sequence[Prediction], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for prediction in predictions:
            handle.write(
                json.dumps(
                    {
                        "image_id": prediction.image_id,
                        "canonical_id": prediction.canonical_id,
                        "boxes": prediction.boxes.tolist(),
                        "labels": prediction.labels.tolist(),
                        "scores": prediction.scores.tolist(),
                    }
                )
                + "\n"
            )
