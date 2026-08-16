#!/usr/bin/env python3
"""Audit saved-prediction AP against fair evaluation with provenance-aware diagnostics."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--fair-summary", type=Path, required=True)
    parser.add_argument("--checkpoint-registry", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--diagnostics-csv", type=Path, required=True)
    parser.add_argument("--exact-tolerance", type=float, default=0.002)
    parser.add_argument("--review-tolerance", type=float, default=0.005)
    parser.add_argument("--strict", action="store_true", help="Exit nonzero when any row requires investigation.")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def xywh_to_xyxy(values: list[float]) -> np.ndarray:
    x, y, w, h = (float(value) for value in values[:4])
    return np.asarray([x, y, x + w, y + h], dtype=np.float32)


def box_iou(labels: np.ndarray, predictions: np.ndarray) -> np.ndarray:
    if len(labels) == 0 or len(predictions) == 0:
        return np.zeros((len(labels), len(predictions)), dtype=np.float32)
    lt = np.maximum(labels[:, None, :2], predictions[None, :, :2])
    rb = np.minimum(labels[:, None, 2:], predictions[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    intersection = wh[..., 0] * wh[..., 1]
    label_area = np.prod(labels[:, 2:] - labels[:, :2], axis=1)[:, None]
    pred_area = np.prod(predictions[:, 2:] - predictions[:, :2], axis=1)[None, :]
    return intersection / np.maximum(label_area + pred_area - intersection, 1e-16)


def ultralytics_match(pred_classes: np.ndarray, true_classes: np.ndarray, iou: np.ndarray) -> np.ndarray:
    thresholds = np.linspace(0.5, 0.95, 10)
    correct = np.zeros((len(pred_classes), len(thresholds)), dtype=bool)
    class_iou = iou * (true_classes[:, None] == pred_classes[None, :])
    for column, threshold in enumerate(thresholds):
        matches = np.asarray(np.nonzero(class_iou >= threshold)).T
        if not len(matches):
            continue
        if len(matches) > 1:
            matches = matches[class_iou[matches[:, 0], matches[:, 1]].argsort()[::-1]]
            matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
            matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        correct[matches[:, 1].astype(int), column] = True
    return correct


def confidence_greedy_match(
    pred_classes: np.ndarray,
    true_classes: np.ndarray,
    iou: np.ndarray,
    confidences: np.ndarray,
) -> np.ndarray:
    thresholds = np.linspace(0.5, 0.95, 10)
    correct = np.zeros((len(pred_classes), len(thresholds)), dtype=bool)
    order = np.argsort(-confidences, kind="stable")
    for column, threshold in enumerate(thresholds):
        used: set[int] = set()
        for pred_index in order:
            candidates = [
                target_index
                for target_index in range(len(true_classes))
                if target_index not in used
                and true_classes[target_index] == pred_classes[pred_index]
                and iou[target_index, pred_index] >= threshold
            ]
            if not candidates:
                continue
            target = max(candidates, key=lambda index: float(iou[index, pred_index]))
            used.add(target)
            correct[pred_index, column] = True
    return correct


def image_arrays(image: dict, class_ids: dict[str, int]) -> tuple[np.ndarray, ...]:
    gt = image.get("gt_boxes", [])
    gt_boxes = np.asarray([xywh_to_xyxy(row) for row in gt], dtype=np.float32).reshape(-1, 4)
    gt_classes = np.asarray([class_ids[str(row[4])] for row in gt], dtype=np.float32)
    predictions = image.get("predictions", [])
    pred_boxes = np.asarray([xywh_to_xyxy(row["bbox"]) for row in predictions], dtype=np.float32).reshape(-1, 4)
    pred_classes = np.asarray([float(row["class_id"]) for row in predictions], dtype=np.float32)
    confidences = np.asarray([float(row["conf"]) for row in predictions], dtype=np.float32)
    return gt_boxes, gt_classes, pred_boxes, pred_classes, confidences


def validator_tp_rows(predictions: list[dict]) -> np.ndarray | None:
    if not all(isinstance(prediction.get("validator_tp_mask"), int) for prediction in predictions):
        return None
    return np.asarray(
        [
            [bool(int(prediction["validator_tp_mask"]) & (1 << bit)) for bit in range(10)]
            for prediction in predictions
        ],
        dtype=bool,
    ).reshape(-1, 10)


def build_stats(payload: dict, *, alternate: bool = False) -> tuple[np.ndarray, ...]:
    tp_rows, confidences, pred_classes, target_classes = [], [], [], []
    class_ids = {name: index for index, name in enumerate(payload["class_names"])}
    for image in payload["images"]:
        gt_boxes, gt_classes, pred_boxes, pred_cls, pred_conf = image_arrays(image, class_ids)
        iou = box_iou(gt_boxes, pred_boxes)
        predictions = image.get("predictions", [])
        saved_tp = validator_tp_rows(predictions)
        if alternate:
            matcher = confidence_greedy_match(pred_cls, gt_classes, iou, pred_conf)
        elif saved_tp is not None:
            matcher = saved_tp
        else:
            matcher = ultralytics_match(pred_cls, gt_classes, iou)
        tp_rows.append(matcher)
        confidences.append(pred_conf)
        pred_classes.append(pred_cls)
        target_classes.append(gt_classes)
    return tuple(
        np.concatenate(values, axis=0) if values else np.asarray([])
        for values in (tp_rows, confidences, pred_classes, target_classes)
    )


def per_image_differences(payload: dict, key: tuple[str, str, int, str]) -> list[dict[str, object]]:
    class_ids = {name: index for index, name in enumerate(payload["class_names"])}
    rows: list[dict[str, object]] = []
    for image in payload["images"]:
        gt_boxes, gt_classes, pred_boxes, pred_cls, pred_conf = image_arrays(image, class_ids)
        iou = box_iou(gt_boxes, pred_boxes)
        offline = ultralytics_match(pred_cls, gt_classes, iou)[:, 0]
        alternate = confidence_greedy_match(pred_cls, gt_classes, iou, pred_conf)[:, 0]
        saved_rows = validator_tp_rows(image.get("predictions", []))
        validator = saved_rows[:, 0] if saved_rows is not None else offline
        if np.array_equal(validator, offline) and np.array_equal(offline, alternate):
            continue
        rows.append(
            {
                "dataset": key[0],
                "variant": key[1],
                "seed": key[2],
                "split": key[3],
                "image": image.get("image", image.get("canonical_id", "")),
                "num_gt": len(gt_boxes),
                "num_predictions": len(pred_boxes),
                "validator_tp50": int(validator.sum()),
                "offline_ultralytics_style_tp50": int(offline.sum()),
                "confidence_greedy_tp50": int(alternate.sum()),
            }
        )
    return rows


def read_registry(path: Path) -> dict[tuple[str, str, int], str]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return {
            (row["dataset"], row["variant"], int(row["seed"])): row["best_sha256"]
            for row in csv.DictReader(handle)
            if row["status"] == "PASS"
        }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if args.review_tolerance < args.exact_tolerance:
        raise SystemExit("--review-tolerance must be >= --exact-tolerance")
    import ultralytics
    from ultralytics.utils.metrics import ap_per_class

    if ultralytics.__version__ != "8.4.60":
        raise SystemExit(f"Expected Ultralytics 8.4.60, found {ultralytics.__version__}")
    with args.fair_summary.expanduser().resolve().open("r", newline="", encoding="utf-8") as handle:
        fair_rows = {
            (row["dataset"], row["variant"], int(row["seed"]), row["split"]): row
            for row in csv.DictReader(handle)
        }
    registry = read_registry(args.checkpoint_registry.expanduser().resolve())

    rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    for path in sorted(args.predictions_root.expanduser().resolve().rglob("*_predictions.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        dataset = str(payload.get("dataset", ""))
        if dataset not in {"vnwoodknot", "vsb_rarefirst"}:
            continue
        key = (dataset, str(payload["variant"]), int(payload["seed"]), str(payload["split"]))
        if key not in fair_rows:
            continue
        fair = fair_rows[key]
        primary = ap_per_class(*build_stats(payload), plot=False)
        alternate = ap_per_class(*build_stats(payload, alternate=True), plot=False)
        recomputed = float(np.mean(primary[5][:, 0]))
        alternate_map = float(np.mean(alternate[5][:, 0]))
        reported = float(fair["mAP50"])
        residual = recomputed - reported
        validator_map = float((payload.get("validator_metrics") or {}).get("mAP50", "nan"))
        validator_residual = validator_map - reported

        expected_checkpoint_hash = registry.get(key[:3], "")
        payload_checkpoint_hash = str(payload.get("checkpoint_sha256", ""))
        checkpoint_match = bool(expected_checkpoint_hash and payload_checkpoint_hash == expected_checkpoint_hash)
        fair_yaml = Path(fair["data_yaml"]).expanduser().resolve()
        fair_yaml_hash = sha256(fair_yaml) if fair_yaml.exists() else ""
        yaml_match = bool(fair_yaml_hash and payload.get("dataset_yaml_sha256") == fair_yaml_hash)
        image_count_match = int(payload["num_images"]) == int(fair["n_images"])
        image_loader_match = payload.get("image_loader") == "ultralytics_yolo_dataset_opencv"
        inference_geometry_match = (
            payload.get("inference_path") == "ultralytics_detection_validator"
            and payload.get("validation_rect") is True
            and float(payload.get("validation_pad", -1)) == 0.5
            and payload.get("validation_scaleup") is False
        )
        tp_source_match = payload.get("tp_source") == "ultralytics_detection_validator_process_batch"
        differences = per_image_differences(payload, key)
        diagnostic_rows.extend(differences)
        provenance_consistent = (
            checkpoint_match
            and yaml_match
            and image_count_match
            and image_loader_match
            and inference_geometry_match
            and tp_source_match
        )

        if (
            abs(residual) <= args.exact_tolerance
            and abs(validator_residual) <= args.exact_tolerance
            and provenance_consistent
        ):
            status = "EXACT_PASS"
            diagnosis = "same_generation_and_within_exact_tolerance"
        elif (
            abs(residual) <= args.review_tolerance
            and abs(validator_residual) <= args.review_tolerance
            and provenance_consistent
        ):
            status = "METHOD_REVIEW"
            diagnosis = "matching_convention_candidate" if differences else "export_rounding_or_ap_interpolation_candidate"
        else:
            status = "INVESTIGATE"
            if not provenance_consistent:
                diagnosis = "generation_or_input_mismatch"
            elif abs(validator_residual) > args.review_tolerance:
                diagnosis = "validator_metric_differs_from_fair_evaluation"
            elif differences:
                diagnosis = "matching_difference_exceeds_review_tolerance"
            else:
                diagnosis = "unexplained_ap_estimator_difference"
        rows.append(
            {
                "dataset": key[0],
                "variant": key[1],
                "seed": key[2],
                "split": key[3],
                "prediction_path": str(path),
                "num_images": int(payload["num_images"]),
                "reported_mAP50": reported,
                "export_validator_mAP50": validator_map,
                "validator_residual": validator_residual,
                "recomputed_mAP50": recomputed,
                "confidence_greedy_mAP50": alternate_map,
                "residual": residual,
                "abs_residual": abs(residual),
                "checkpoint_hash_match": checkpoint_match,
                "dataset_yaml_hash_match": yaml_match,
                "image_count_match": image_count_match,
                "image_loader_match": image_loader_match,
                "inference_geometry_match": inference_geometry_match,
                "tp_source_match": tp_source_match,
                "matching_difference_images": len(differences),
                "diagnosis": diagnosis,
                "status": status,
            }
        )

    if not rows:
        raise SystemExit("No prediction exports matched fair-summary rows.")
    write_csv(args.output_csv.expanduser().resolve(), rows, list(rows[0]))
    diagnostic_fields = [
        "dataset", "variant", "seed", "split", "image", "num_gt", "num_predictions",
        "validator_tp50", "offline_ultralytics_style_tp50", "confidence_greedy_tp50",
    ]
    write_csv(args.diagnostics_csv.expanduser().resolve(), diagnostic_rows, diagnostic_fields)
    counts = {status: sum(row["status"] == status for row in rows) for status in ("EXACT_PASS", "METHOD_REVIEW", "INVESTIGATE")}
    maximum = max(float(row["abs_residual"]) for row in rows)
    print(f"Ultralytics: {ultralytics.__version__} ({Path(ultralytics.__file__).resolve()})")
    print(f"Wrote: {args.output_csv.expanduser().resolve()}")
    print(f"Wrote: {args.diagnostics_csv.expanduser().resolve()}")
    print(f"AP REPRODUCTION AUDIT: {counts}; max_abs_residual={maximum:.6f}")
    for row in rows:
        if row["status"] != "EXACT_PASS":
            print(
                f"{row['status']} {row['dataset']} {row['variant']} seed={row['seed']} {row['split']}: "
                f"residual={float(row['residual']):+.6f}; {row['diagnosis']}"
            )
    if args.strict and counts["INVESTIGATE"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
