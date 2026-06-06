#!/usr/bin/env python3
"""Render qualitative VNWoodKnot detection scenarios from real predictions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from threshold_analysis import box_iou, xywh_to_xyxy
from threshold_sweep_inference import (
    DEFAULT_CHECKPOINT_ROOT,
    DEFAULT_DATA_YAML,
    load_yolo_test_records,
    predict_records,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "qualitative"
PREDICTIONS_DIR = PROJECT_ROOT / "results" / "negative_aware" / "predictions"
VARIANTS = ("baseline", "p2_illumination", "a1_crop", "a2_colorjitter", "p4_a4_combined")
VARIANT_LABELS = {
    "baseline": "Baseline",
    "p2_illumination": "P2 illumination",
    "a1_crop": "A1 crop",
    "a2_colorjitter": "A2 color jitter",
    "p4_a4_combined": "P4+A4 combined",
}
TP_COLOR = "#009E73"
FP_COLOR = "#D55E00"
GT_COLOR = "#009E73"
TEXT_BG = {"facecolor": "black", "alpha": 0.62, "edgecolor": "none", "pad": 1.5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-yaml", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--predictions-dir", type=Path, default=PREDICTIONS_DIR)
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument(
        "--render-only",
        action="store_true",
        help="Reuse detection_scenarios_selection.json and existing prediction JSONs; never run candidate search or inference.",
    )
    parser.add_argument(
        "--selection-json",
        type=Path,
        default=None,
        help="Selection JSON to use with --render-only. Defaults to <output-dir>/detection_scenarios_selection.json.",
    )
    parser.add_argument("--force-inline", action="store_true", help="Ignore existing prediction JSONs and run checkpoint inference.")
    parser.add_argument("--no-save-inline-predictions", action="store_true")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.01)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--scenario1-image", default=None)
    parser.add_argument("--scenario2-image", default=None)
    parser.add_argument("--scenario3-image", default=None)
    parser.add_argument("--scenario4-image", default=None)
    parser.add_argument("--scenario1-threshold", type=float, default=0.25)
    parser.add_argument("--scenario2-baseline-threshold", type=float, default=0.60)
    parser.add_argument("--scenario2-p4-threshold", type=float, default=0.35)
    parser.add_argument("--scenario3-threshold", type=float, default=0.35)
    parser.add_argument("--scenario4-threshold", type=float, default=0.10)
    parser.add_argument("--candidate-log-size", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    payloads = load_payloads(args)
    warn_if_prediction_datasets_differ(payloads, args.data_yaml)
    if args.render_only:
        scenarios, candidate_log = build_scenarios_from_selection(payloads, args, output_dir)
    else:
        scenarios, candidate_log = build_scenarios(payloads, args)
    figure_paths = render_all_scenarios(scenarios, output_dir=output_dir, dpi=args.dpi)
    write_selection_logs(scenarios, candidate_log, output_dir)

    print("Chosen detection scenario images:")
    for index, scenario in enumerate(scenarios, start=1):
        print(f"Scenario {index}: {scenario['image_id']} | {scenario['title']}")
    for path in figure_paths:
        print(f"Wrote: {path}")
    print(f"Wrote: {output_dir / 'detection_scenarios_selection.txt'}")


def load_payloads(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    payloads = {}
    for variant in VARIANTS:
        payloads[variant] = load_or_generate_payload(variant, args)
    return payloads


def load_or_generate_payload(variant: str, args: argparse.Namespace) -> dict[str, Any]:
    predictions_dir = args.predictions_dir.expanduser().resolve()
    predictions_path = predictions_dir / f"{variant}_seed{args.seed}_predictions.json"
    if predictions_path.exists() and not args.force_inline:
        payload = json.loads(predictions_path.read_text(encoding="utf-8"))
        payload.setdefault("variant", variant)
        payload.setdefault("seed", args.seed)
        return payload
    if args.render_only:
        raise SystemExit(f"--render-only requires existing prediction JSON: {predictions_path}")

    data_yaml = args.data_yaml.expanduser().resolve()
    if not data_yaml.exists():
        raise SystemExit(f"Cannot run inline inference; dataset YAML not found: {data_yaml}")
    run_dir = args.checkpoint_root.expanduser().resolve() / f"{variant}_seed{args.seed}"
    checkpoint = run_dir / "ultralytics" / "train" / "weights" / "best.pt"
    if not checkpoint.exists():
        raise SystemExit(f"Cannot run inline inference; checkpoint not found: {checkpoint}")

    print(f"Running inline inference for {variant}_seed{args.seed} on {data_yaml}")
    records, class_names = load_yolo_test_records(data_yaml)
    predictions_by_key = predict_records(
        checkpoint=checkpoint,
        records=records,
        class_names=class_names,
        imgsz=args.imgsz,
        batch=args.batch,
        max_det=args.max_det,
        conf=args.conf,
        device=args.device,
    )
    payload = {
        "checkpoint": f"{variant}_seed{args.seed}",
        "variant": variant,
        "seed": args.seed,
        "checkpoint_path": str(checkpoint),
        "dataset_yaml": str(data_yaml),
        "split": "test",
        "base_confidence_threshold": float(args.conf),
        "class_names": list(class_names),
        "num_images": len(records),
        "num_knot_free_images": sum(1 for record in records if record.is_knot_free),
        "images": [
            {
                "image": record.image_name,
                "canonical_id": record.canonical_id,
                "image_path": str(record.image_path),
                "width": record.width,
                "height": record.height,
                "is_knot_free": record.is_knot_free,
                "gt_boxes": record.gt_boxes_json,
                "predictions": predictions_by_key.get(record.canonical_id, []),
            }
            for record in records
        ],
    }
    if not args.no_save_inline_predictions:
        predictions_dir.mkdir(parents=True, exist_ok=True)
        predictions_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote inline predictions: {predictions_path}")
    return payload


def build_scenarios(payloads: dict[str, dict[str, Any]], args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    lookups = {variant: image_lookup(payload) for variant, payload in payloads.items()}
    candidate_log: dict[str, list[dict[str, Any]]] = {}

    scenario1_id, candidate_log["scenario_1"] = choose_scenario1(lookups, args)
    scenario2_id, candidate_log["scenario_2"] = choose_scenario2(lookups, args)
    scenario3_id, candidate_log["scenario_3"] = choose_scenario3(lookups, args)
    scenario4_id, candidate_log["scenario_4"] = choose_scenario4(lookups, args)

    scenarios = [
        {
            "title": f"Clean wood at tau={args.scenario1_threshold:.2f}",
            "image_id": scenario1_id,
            "panels": [
                panel("a1_crop", lookups["a1_crop"][scenario1_id], args.scenario1_threshold, "A1 crop"),
                panel("p4_a4_combined", lookups["p4_a4_combined"][scenario1_id], args.scenario1_threshold, "P4+A4 combined"),
            ],
        },
        {
            "title": "Missed knots vs retained recall",
            "image_id": scenario2_id,
            "panels": [
                panel("baseline", lookups["baseline"][scenario2_id], args.scenario2_baseline_threshold, "Baseline"),
                panel("p4_a4_combined", lookups["p4_a4_combined"][scenario2_id], args.scenario2_p4_threshold, "P4+A4 combined"),
            ],
        },
        {
            "title": f"Correct detection at tau={args.scenario3_threshold:.2f}",
            "image_id": scenario3_id,
            "panels": [
                panel("p4_a4_combined", lookups["p4_a4_combined"][scenario3_id], args.scenario3_threshold, "P4+A4 combined"),
            ],
        },
        {
            "title": f"Clean-wood confidence at tau={args.scenario4_threshold:.2f}",
            "image_id": scenario4_id,
            "panels": [
                panel("a1_crop", lookups["a1_crop"][scenario4_id], args.scenario4_threshold, "A1 crop"),
                panel("a2_colorjitter", lookups["a2_colorjitter"][scenario4_id], args.scenario4_threshold, "A2 color jitter"),
                panel("p2_illumination", lookups["p2_illumination"][scenario4_id], args.scenario4_threshold, "P2 illumination"),
            ],
        },
    ]
    return scenarios, candidate_log


def build_scenarios_from_selection(
    payloads: dict[str, dict[str, Any]],
    args: argparse.Namespace,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    selection_path = args.selection_json.expanduser().resolve() if args.selection_json else output_dir / "detection_scenarios_selection.json"
    if not selection_path.exists():
        raise SystemExit(f"--render-only requires an existing selection JSON: {selection_path}")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    chosen = selection.get("chosen") or []
    if len(chosen) < 4:
        raise SystemExit(f"Selection JSON must contain four chosen scenarios: {selection_path}")

    ids_by_scenario = {}
    for item in chosen:
        scenario_number = int(item.get("scenario", 0))
        image_id = str(item.get("image_id") or "")
        if scenario_number and image_id:
            ids_by_scenario[scenario_number] = image_id
    missing = [number for number in range(1, 5) if number not in ids_by_scenario]
    if missing:
        raise SystemExit(f"Selection JSON missing scenario IDs: {missing}")

    lookups = {variant: image_lookup(payload) for variant, payload in payloads.items()}
    candidate_log = selection.get("candidates") or {}
    scenarios = build_scenarios_for_ids(
        lookups,
        args,
        scenario1_id=ids_by_scenario[1],
        scenario2_id=ids_by_scenario[2],
        scenario3_id=ids_by_scenario[3],
        scenario4_id=ids_by_scenario[4],
    )
    print(f"Render-only: reused selected images from {selection_path}")
    return scenarios, candidate_log


def build_scenarios_for_ids(
    lookups: dict[str, dict[str, dict[str, Any]]],
    args: argparse.Namespace,
    *,
    scenario1_id: str,
    scenario2_id: str,
    scenario3_id: str,
    scenario4_id: str,
) -> list[dict[str, Any]]:
    validate_selected_ids(lookups, scenario1_id, ("a1_crop", "p4_a4_combined"), "scenario 1")
    validate_selected_ids(lookups, scenario2_id, ("baseline", "p4_a4_combined"), "scenario 2")
    validate_selected_ids(lookups, scenario3_id, ("p4_a4_combined",), "scenario 3")
    validate_selected_ids(lookups, scenario4_id, ("a1_crop", "a2_colorjitter", "p2_illumination"), "scenario 4")
    return [
        {
            "title": f"Clean wood at tau={args.scenario1_threshold:.2f}",
            "image_id": scenario1_id,
            "panels": [
                panel("a1_crop", lookups["a1_crop"][scenario1_id], args.scenario1_threshold, "A1 crop"),
                panel("p4_a4_combined", lookups["p4_a4_combined"][scenario1_id], args.scenario1_threshold, "P4+A4 combined"),
            ],
        },
        {
            "title": "Missed knots vs retained recall",
            "image_id": scenario2_id,
            "panels": [
                panel("baseline", lookups["baseline"][scenario2_id], args.scenario2_baseline_threshold, "Baseline"),
                panel("p4_a4_combined", lookups["p4_a4_combined"][scenario2_id], args.scenario2_p4_threshold, "P4+A4 combined"),
            ],
        },
        {
            "title": f"Correct detection at tau={args.scenario3_threshold:.2f}",
            "image_id": scenario3_id,
            "panels": [
                panel("p4_a4_combined", lookups["p4_a4_combined"][scenario3_id], args.scenario3_threshold, "P4+A4 combined"),
            ],
        },
        {
            "title": f"Clean-wood confidence at tau={args.scenario4_threshold:.2f}",
            "image_id": scenario4_id,
            "panels": [
                panel("a1_crop", lookups["a1_crop"][scenario4_id], args.scenario4_threshold, "A1 crop"),
                panel("a2_colorjitter", lookups["a2_colorjitter"][scenario4_id], args.scenario4_threshold, "A2 color jitter"),
                panel("p2_illumination", lookups["p2_illumination"][scenario4_id], args.scenario4_threshold, "P2 illumination"),
            ],
        },
    ]


def validate_selected_ids(
    lookups: dict[str, dict[str, dict[str, Any]]],
    image_id: str,
    variants: tuple[str, ...],
    scenario_name: str,
) -> None:
    missing = [variant for variant in variants if image_id not in lookups[variant]]
    if missing:
        raise SystemExit(f"Selected {scenario_name} image is missing for variants {missing}: {image_id}")


def choose_scenario1(lookups: dict[str, dict[str, dict[str, Any]]], args: argparse.Namespace) -> tuple[str, list[dict[str, Any]]]:
    if args.scenario1_image:
        return require_common_image(args.scenario1_image, lookups, ("a1_crop", "p4_a4_combined")), []
    candidates = []
    for image_id in common_ids(lookups, ("a1_crop", "p4_a4_combined")):
        a1 = lookups["a1_crop"][image_id]
        p4 = lookups["p4_a4_combined"][image_id]
        if not is_knot_free(a1) or not is_knot_free(p4):
            continue
        a1_preds = threshold_predictions(a1, args.scenario1_threshold)
        p4_preds = threshold_predictions(p4, args.scenario1_threshold)
        if a1_preds and not p4_preds:
            candidates.append(
                {
                    "image_id": image_id,
                    "score": max_conf(a1_preds),
                    "a1_predictions": len(a1_preds),
                    "p4_predictions": len(p4_preds),
                }
            )
    return choose_best("scenario 1", candidates, score_key="score", log_size=args.candidate_log_size)


def choose_scenario2(lookups: dict[str, dict[str, dict[str, Any]]], args: argparse.Namespace) -> tuple[str, list[dict[str, Any]]]:
    if args.scenario2_image:
        return require_common_image(args.scenario2_image, lookups, ("baseline", "p4_a4_combined")), []
    candidates = []
    fallback = []
    for image_id in common_ids(lookups, ("baseline", "p4_a4_combined")):
        base = lookups["baseline"][image_id]
        p4 = lookups["p4_a4_combined"][image_id]
        if is_knot_free(base):
            continue
        gt = len(base.get("gt_boxes", []))
        if gt < 2:
            continue
        base_stats = detection_stats(base, args.scenario2_baseline_threshold)
        p4_stats = detection_stats(p4, args.scenario2_p4_threshold)
        score = (p4_stats["recall"] - base_stats["recall"]) + 0.02 * gt - 0.03 * p4_stats["fp"]
        row = {
            "image_id": image_id,
            "score": round(float(score), 6),
            "gt_boxes": gt,
            "baseline_recall": round(base_stats["recall"], 6),
            "p4_recall": round(p4_stats["recall"], 6),
            "baseline_tp": base_stats["tp"],
            "p4_tp": p4_stats["tp"],
        }
        fallback.append(row)
        if base_stats["recall"] <= 0.60 and p4_stats["recall"] >= 0.75 and p4_stats["recall"] > base_stats["recall"]:
            candidates.append(row)
    if not candidates:
        candidates = fallback
    return choose_best("scenario 2", candidates, score_key="score", log_size=args.candidate_log_size)


def choose_scenario3(lookups: dict[str, dict[str, dict[str, Any]]], args: argparse.Namespace) -> tuple[str, list[dict[str, Any]]]:
    if args.scenario3_image:
        return require_common_image(args.scenario3_image, lookups, ("p4_a4_combined",)), []
    candidates = []
    fallback = []
    for image_id, image in lookups["p4_a4_combined"].items():
        if is_knot_free(image):
            continue
        gt = len(image.get("gt_boxes", []))
        stats = detection_stats(image, args.scenario3_threshold)
        f1 = 2.0 * stats["precision"] * stats["recall"] / max(stats["precision"] + stats["recall"], 1e-8)
        score = f1 + 0.02 * gt - 0.03 * stats["fp"]
        row = {
            "image_id": image_id,
            "score": round(float(score), 6),
            "gt_boxes": gt,
            "precision": round(stats["precision"], 6),
            "recall": round(stats["recall"], 6),
            "tp": stats["tp"],
            "fp": stats["fp"],
        }
        fallback.append(row)
        if stats["precision"] >= 0.75 and stats["recall"] >= 0.75:
            candidates.append(row)
    if not candidates:
        candidates = fallback
    return choose_best("scenario 3", candidates, score_key="score", log_size=args.candidate_log_size)


def choose_scenario4(lookups: dict[str, dict[str, dict[str, Any]]], args: argparse.Namespace) -> tuple[str, list[dict[str, Any]]]:
    if args.scenario4_image:
        return require_common_image(args.scenario4_image, lookups, ("a1_crop", "a2_colorjitter", "p2_illumination")), []
    candidates = []
    for image_id in common_ids(lookups, ("a1_crop", "a2_colorjitter", "p2_illumination")):
        a1 = lookups["a1_crop"][image_id]
        if not is_knot_free(a1):
            continue
        a1_preds = threshold_predictions(a1, args.scenario4_threshold)
        a2_preds = threshold_predictions(lookups["a2_colorjitter"][image_id], args.scenario4_threshold)
        p2_preds = threshold_predictions(lookups["p2_illumination"][image_id], args.scenario4_threshold)
        if not a1_preds:
            continue
        other_max = max(max_conf(a2_preds), max_conf(p2_preds))
        score = max_conf(a1_preds) - other_max + (0.2 if not a2_preds and not p2_preds else 0.0)
        candidates.append(
            {
                "image_id": image_id,
                "score": round(float(score), 6),
                "a1_max_conf": round(max_conf(a1_preds), 6),
                "a2_predictions": len(a2_preds),
                "p2_predictions": len(p2_preds),
            }
        )
    return choose_best("scenario 4", candidates, score_key="score", log_size=args.candidate_log_size)


def render_all_scenarios(scenarios: list[dict[str, Any]], *, output_dir: Path, dpi: int) -> list[Path]:
    configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    paths: list[Path] = []
    for index, scenario in enumerate(scenarios, start=1):
        paths.extend(render_scenario_figure(scenario, output_dir / f"scenario_{index}", dpi=dpi))
    paths.extend(render_combined_figure(scenarios, output_dir / "detection_scenarios", dpi=dpi))
    plt.close("all")
    return paths


def render_scenario_figure(scenario: dict[str, Any], stem: Path, *, dpi: int) -> list[Path]:
    import matplotlib.pyplot as plt

    panels = scenario["panels"]
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.25), squeeze=False)
    reference = panels[0]["image"]
    for axis, panel_data in zip(axes.flat, panels):
        draw_panel(axis, panel_data, reference_image=reference)
    fig.suptitle(scenario["title"], fontsize=11, fontweight="bold", y=0.975)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=0.88, wspace=0.04)
    return save_figure(fig, stem, dpi=dpi)


def render_combined_figure(scenarios: list[dict[str, Any]], stem: Path, *, dpi: int) -> list[Path]:
    import matplotlib.pyplot as plt

    row_heights = [max(0.95, 0.98 + 0.08 * len(scenario["panels"])) for scenario in scenarios]
    fig = plt.figure(figsize=(7.0, 7.6))
    outer = fig.add_gridspec(
        len(scenarios),
        1,
        height_ratios=row_heights,
        left=0.01,
        right=0.99,
        bottom=0.01,
        top=0.985,
        hspace=0.10,
    )
    for row_index, scenario in enumerate(scenarios):
        panels = scenario["panels"]
        reference = panels[0]["image"]
        inner = outer[row_index].subgridspec(2, len(panels), height_ratios=[0.20, 1.0], hspace=0.06, wspace=0.05)
        caption_axis = fig.add_subplot(inner[0, :])
        caption_axis.set_axis_off()
        caption_axis.text(
            0.0,
            0.50,
            f"{chr(ord('A') + row_index)}. {scenario['title']}",
            transform=caption_axis.transAxes,
            fontsize=11,
            fontweight="bold",
            va="center",
            ha="left",
        )
        for col_index, panel_data in enumerate(panels):
            axis = fig.add_subplot(inner[1, col_index])
            draw_panel(axis, panel_data, reference_image=reference)
    return save_figure(fig, stem, dpi=dpi)


def draw_panel(axis, panel_data: dict[str, Any], *, reference_image: dict[str, Any]) -> None:
    import matplotlib.patches as patches

    image = panel_data["image"]
    display_image = load_image(reference_image)
    axis.imshow(display_image)
    axis.axis("off")
    axis.set_title(f"{panel_data['label']}  tau={panel_data['threshold']:.2f}", fontsize=9, pad=2)

    for gt in image.get("gt_boxes", []):
        x, y, w, h, class_name = gt
        rect = patches.Rectangle((float(x), float(y)), float(w), float(h), fill=False, edgecolor=GT_COLOR, linewidth=1.0, linestyle="--")
        axis.add_patch(rect)
        axis.text(
            float(x),
            max(float(y) - 2, 0),
            str(class_name),
            color="white",
            fontsize=6,
            va="bottom",
            bbox={"facecolor": GT_COLOR, "alpha": 0.78, "edgecolor": "none", "pad": 0.8},
        )

    matched = match_predictions(image, panel_data["threshold"])
    for pred in matched["predictions"]:
        x, y, w, h = [float(value) for value in pred["bbox"]]
        is_tp = pred["status"] == "tp"
        color = TP_COLOR if is_tp else FP_COLOR
        rect = patches.Rectangle((x, y), w, h, fill=False, edgecolor=color, linewidth=1.3)
        axis.add_patch(rect)
        axis.text(
            x,
            min(y + h + 2, image.get("height", display_image.height) - 2),
            f"{pred['class']} {float(pred['conf']):.2f}",
            color="white",
            fontsize=6,
            va="top",
            bbox={"facecolor": color, "alpha": 0.86, "edgecolor": "none", "pad": 0.8},
        )


def match_predictions(image: dict[str, Any], threshold: float, iou_threshold: float = 0.50) -> dict[str, Any]:
    gt_rows = [
        {"bbox_xyxy": np.asarray(xywh_to_xyxy(row[:4]), dtype=np.float32), "class": str(row[4]), "matched": False}
        for row in image.get("gt_boxes", [])
    ]
    pred_rows = sorted(threshold_predictions(image, threshold), key=lambda row: float(row.get("conf", 0.0)), reverse=True)
    output_preds = []
    for prediction in pred_rows:
        pred_box = np.asarray(xywh_to_xyxy(prediction["bbox"]), dtype=np.float32)
        same_class_indices = [idx for idx, gt in enumerate(gt_rows) if not gt["matched"] and gt["class"] == str(prediction.get("class"))]
        status = "fp"
        if same_class_indices:
            gt_boxes = np.asarray([gt_rows[idx]["bbox_xyxy"] for idx in same_class_indices], dtype=np.float32).reshape(-1, 4)
            ious = box_iou(pred_box[None, :], gt_boxes)[0]
            best_local = int(np.argmax(ious))
            if float(ious[best_local]) >= iou_threshold:
                gt_rows[same_class_indices[best_local]]["matched"] = True
                status = "tp"
        copied = dict(prediction)
        copied["status"] = status
        output_preds.append(copied)
    missed = [gt for gt in gt_rows if not gt["matched"]]
    return {"predictions": output_preds, "missed_gt": missed}


def detection_stats(image: dict[str, Any], threshold: float) -> dict[str, float | int]:
    matched = match_predictions(image, threshold)
    tp = sum(1 for pred in matched["predictions"] if pred["status"] == "tp")
    fp = sum(1 for pred in matched["predictions"] if pred["status"] == "fp")
    fn = len(matched["missed_gt"])
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": tp / max(tp + fp, 1),
        "recall": tp / max(tp + fn, 1),
    }


def panel(variant: str, image: dict[str, Any], threshold: float, label: str) -> dict[str, Any]:
    return {"variant": variant, "image": image, "threshold": float(threshold), "label": label}


def image_lookup(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup = {}
    for image in payload.get("images", []):
        image_id = canonical_image_id(image)
        image["canonical_id"] = image_id
        lookup[image_id] = image
    return lookup


def common_ids(lookups: dict[str, dict[str, dict[str, Any]]], variants: tuple[str, ...]) -> list[str]:
    ids = set(lookups[variants[0]])
    for variant in variants[1:]:
        ids &= set(lookups[variant])
    return sorted(ids)


def require_common_image(query: str, lookups: dict[str, dict[str, dict[str, Any]]], variants: tuple[str, ...]) -> str:
    matches = []
    for image_id in common_ids(lookups, variants):
        images = [lookups[variant][image_id] for variant in variants]
        if any(image_matches(image, query) for image in images):
            matches.append(image_id)
    if not matches:
        raise SystemExit(f"Override image not found across variants {variants}: {query}")
    if len(matches) > 1:
        raise SystemExit(f"Override image is ambiguous: {query}. Matches: {matches[:10]}")
    return matches[0]


def choose_best(name: str, candidates: list[dict[str, Any]], *, score_key: str, log_size: int) -> tuple[str, list[dict[str, Any]]]:
    if not candidates:
        raise SystemExit(f"No suitable candidate found for {name}. Review prediction JSONs or pass an override image.")
    ranked = sorted(candidates, key=lambda row: float(row[score_key]), reverse=True)
    return str(ranked[0]["image_id"]), ranked[:log_size]


def threshold_predictions(image: dict[str, Any], threshold: float) -> list[dict[str, Any]]:
    return [prediction for prediction in image.get("predictions", []) if float(prediction.get("conf", 0.0)) >= threshold]


def max_conf(predictions: list[dict[str, Any]]) -> float:
    if not predictions:
        return 0.0
    return max(float(prediction.get("conf", 0.0)) for prediction in predictions)


def is_knot_free(image: dict[str, Any]) -> bool:
    return bool(image.get("is_knot_free", False)) or len(image.get("gt_boxes", [])) == 0


def image_matches(image: dict[str, Any], query: str) -> bool:
    query_norm = normalize_identifier(query)
    candidates = {
        normalize_identifier(image.get("canonical_id", "")),
        normalize_identifier(image.get("image", "")),
        normalize_identifier(Path(str(image.get("image", ""))).name),
        normalize_identifier(Path(str(image.get("image", ""))).stem),
        normalize_identifier(image.get("image_path", "")),
        normalize_identifier(Path(str(image.get("image_path", ""))).name),
        normalize_identifier(Path(str(image.get("image_path", ""))).stem),
    }
    return query_norm in candidates or any(candidate.endswith(query_norm) for candidate in candidates)


def canonical_image_id(image: dict[str, Any]) -> str:
    return normalize_identifier(image.get("canonical_id") or image.get("image") or image.get("image_path") or "")


def normalize_identifier(value: Any) -> str:
    return str(value).replace("\\", "/").strip().lower()


def load_image(image: dict[str, Any]) -> Image.Image:
    path = Path(str(image.get("image_path", ""))).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"Image path is not accessible: {path}")
    return Image.open(path).convert("RGB")


def save_figure(fig: Any, stem: Path, *, dpi: int) -> list[Path]:
    paths = [stem.with_suffix(".pdf"), stem.with_suffix(".png")]
    fig.savefig(paths[0], bbox_inches="tight")
    fig.savefig(paths[1], dpi=dpi, bbox_inches="tight")
    return paths


def write_selection_logs(scenarios: list[dict[str, Any]], candidate_log: dict[str, list[dict[str, Any]]], output_dir: Path) -> None:
    txt_lines = ["Detection scenario selection", ""]
    json_payload = {"chosen": [], "candidates": candidate_log}
    for index, scenario in enumerate(scenarios, start=1):
        first_image = scenario["panels"][0]["image"]
        txt_lines.append(f"Scenario {index}: {scenario['title']}")
        txt_lines.append(f"  image_id: {scenario['image_id']}")
        txt_lines.append(f"  image: {first_image.get('image', '')}")
        txt_lines.append(f"  image_path: {first_image.get('image_path', '')}")
        json_payload["chosen"].append(
            {
                "scenario": index,
                "title": scenario["title"],
                "image_id": scenario["image_id"],
                "image": first_image.get("image", ""),
                "image_path": first_image.get("image_path", ""),
            }
        )
    txt_lines.append("")
    txt_lines.append("Top candidates:")
    for scenario_name, rows in candidate_log.items():
        txt_lines.append(f"- {scenario_name}")
        for row in rows:
            txt_lines.append(f"  {row}")
    (output_dir / "detection_scenarios_selection.txt").write_text("\n".join(txt_lines) + "\n", encoding="utf-8")
    (output_dir / "detection_scenarios_selection.json").write_text(json.dumps(json_payload, indent=2) + "\n", encoding="utf-8")


def warn_if_prediction_datasets_differ(payloads: dict[str, dict[str, Any]], requested_data_yaml: Path) -> None:
    datasets = {str(payload.get("dataset_yaml", "")) for payload in payloads.values() if payload.get("dataset_yaml")}
    if len(datasets) > 1:
        print("WARNING: prediction JSONs reference multiple dataset YAMLs.")
        for value in sorted(datasets):
            print(f"  - {value}")
        print("For strict same-image panels, regenerate/use predictions with --data-yaml pointing to the baseline VNWoodKnot test set.")
    requested = str(requested_data_yaml.expanduser())
    if datasets and requested not in datasets and requested_data_yaml.exists():
        print(f"WARNING: requested --data-yaml differs from loaded prediction JSON dataset YAMLs: {requested}")


def configure_matplotlib_cache() -> None:
    import tempfile

    cache_dir = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "wood_defect_datacentric_mpl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


if __name__ == "__main__":
    main()
