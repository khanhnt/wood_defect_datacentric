#!/usr/bin/env python3
"""Render representative VNWoodKnot test-set samples for the paper."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from threshold_sweep_inference import DEFAULT_DATA_YAML, TestRecord, load_yolo_test_records


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "results" / "qualitative"
PANEL_LETTERS = ("a", "b", "c", "d", "e", "f")
CLASS_COLORS = {
    "live_knot": "#0072B2",
    "dead_knot": "#D55E00",
    "knot_free": "#4D4D4D",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", type=Path, default=DEFAULT_DATA_YAML)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=42, help="Used only to break ties in auto-selection.")
    parser.add_argument(
        "--images",
        nargs="*",
        default=None,
        help="Optional six image names/paths in panel order: live, live, dead, dead, knot-free, knot-free.",
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--candidate-log-size", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = args.data_yaml.expanduser().resolve()
    if not data_yaml.exists():
        raise SystemExit(f"VNWoodKnot dataset YAML not found: {data_yaml}")

    records, class_names = load_yolo_test_records(data_yaml)
    if list(class_names) != ["live_knot", "dead_knot"]:
        raise SystemExit(f"Unexpected VNWoodKnot class names in {data_yaml}: {class_names}")
    selected, candidate_log = select_records(records, args)

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    render_dataset_samples(selected, output_dir=output_dir, dpi=args.dpi)
    write_selection_log(selected, candidate_log, output_dir / "dataset_samples_selection.txt")
    write_selection_json(selected, candidate_log, output_dir / "dataset_samples_selection.json")

    print("Chosen dataset sample images:")
    for index, record in enumerate(selected):
        print(f"({PANEL_LETTERS[index]}) {panel_label(index)} | {record.image_name}")
    print(f"Wrote: {output_dir / 'dataset_samples.pdf'}")
    print(f"Wrote: {output_dir / 'dataset_samples.png'}")
    print(f"Wrote: {output_dir / 'dataset_samples_selection.txt'}")


def select_records(records: list[TestRecord], args: argparse.Namespace) -> tuple[list[TestRecord], dict[str, list[dict[str, float | str]]]]:
    if args.images:
        if len(args.images) != 6:
            raise SystemExit("--images must provide exactly six image names/paths.")
        return [match_record(records, item) for item in args.images], {}

    rng = np.random.default_rng(args.seed)
    candidate_log: dict[str, list[dict[str, float | str]]] = {}
    live = rank_records(records, "live_knot", rng=rng)
    dead = rank_records(records, "dead_knot", rng=rng)
    free = rank_records(records, "knot_free", rng=rng)
    for key, ranked in (("live_knot", live), ("dead_knot", dead), ("knot_free", free)):
        candidate_log[key] = [
            {"image": record.image_name, "score": round(float(score), 6)}
            for score, record in ranked[: args.candidate_log_size]
        ]
    selected = [record for _, record in live[:2]] + [record for _, record in dead[:2]] + [record for _, record in free[:2]]
    if len(selected) != 6:
        raise SystemExit("Could not auto-select six dataset samples from the VNWoodKnot test set.")
    return selected, candidate_log


def rank_records(records: list[TestRecord], category: str, *, rng: np.random.Generator) -> list[tuple[float, TestRecord]]:
    ranked = []
    for record in records:
        if category == "knot_free":
            if not record.is_knot_free:
                continue
            score = image_quality_score(record) + float(rng.uniform(0, 1e-4))
        else:
            class_id = list(record.class_names).index(category)
            if not np.any(record.labels == class_id):
                continue
            box_score = best_box_score(record, class_id)
            if box_score <= 0.0:
                continue
            score = image_quality_score(record) + box_score + float(rng.uniform(0, 1e-4))
        ranked.append((score, record))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def image_quality_score(record: TestRecord) -> float:
    with Image.open(record.image_path) as image:
        arr = np.asarray(image.convert("L").resize((256, 256)), dtype=np.float32) / 255.0
    brightness = float(arr.mean())
    contrast = float(arr.std())
    gx = np.diff(arr, axis=1)
    gy = np.diff(arr, axis=0)
    sharpness = float(gx.var() + gy.var())
    exposure_score = 1.0 - min(abs(brightness - 0.55) / 0.55, 1.0)
    contrast_score = min(contrast / 0.22, 1.0)
    sharpness_score = min(sharpness / 0.012, 1.0)
    return 0.50 * exposure_score + 0.25 * contrast_score + 0.25 * sharpness_score


def best_box_score(record: TestRecord, class_id: int) -> float:
    best = 0.0
    boxes = record.boxes_xyxy[record.labels == class_id]
    for box in boxes:
        x1, y1, x2, y2 = [float(value) for value in box]
        width = max(x2 - x1, 0.0)
        height = max(y2 - y1, 0.0)
        area = (width * height) / max(record.width * record.height, 1)
        margin = min(x1 / record.width, y1 / record.height, (record.width - x2) / record.width, (record.height - y2) / record.height)
        if margin < 0.015 or area < 0.001 or area > 0.20:
            continue
        area_score = 1.0 - min(abs(np.log10(max(area, 1e-6)) - np.log10(0.025)) / 2.0, 1.0)
        margin_score = min(margin / 0.10, 1.0)
        best = max(best, 0.65 * area_score + 0.35 * margin_score)
    return best


def render_dataset_samples(records: list[TestRecord], *, output_dir: Path, dpi: int) -> None:
    configure_matplotlib_cache()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.8))
    for index, (axis, record) in enumerate(zip(axes.flat, records)):
        image = Image.open(record.image_path).convert("RGB")
        axis.imshow(image)
        axis.set_axis_off()
        draw_ground_truth(axis, record)
        title = f"({PANEL_LETTERS[index]}) {panel_label(index)}"
        axis.text(
            0.02,
            0.04,
            title,
            transform=axis.transAxes,
            color="white",
            fontsize=8,
            fontweight="bold",
            bbox={"facecolor": "black", "alpha": 0.58, "edgecolor": "none", "pad": 2.0},
        )
    fig.tight_layout(pad=0.15, w_pad=0.15, h_pad=0.15)
    fig.savefig(output_dir / "dataset_samples.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "dataset_samples.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def draw_ground_truth(axis, record: TestRecord) -> None:
    import matplotlib.patches as patches

    for box, class_id in zip(record.boxes_xyxy, record.labels):
        class_name = record.class_names[int(class_id)]
        color = CLASS_COLORS.get(class_name, "#009E73")
        x1, y1, x2, y2 = [float(value) for value in box]
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.2)
        axis.add_patch(rect)
        axis.text(
            x1,
            max(y1 - 3, 0),
            class_name,
            color="white",
            fontsize=7,
            va="bottom",
            ha="left",
            bbox={"facecolor": color, "alpha": 0.88, "edgecolor": "none", "pad": 1.2},
        )


def panel_label(index: int) -> str:
    if index < 2:
        return "live knot"
    if index < 4:
        return "dead knot"
    return "knot-free"


def match_record(records: list[TestRecord], query: str) -> TestRecord:
    normalized = normalize_identifier(query)
    matches = [record for record in records if record_matches(record, normalized)]
    if not matches:
        raise SystemExit(f"Could not find requested image in test split: {query}")
    if len(matches) > 1:
        names = ", ".join(record.image_name for record in matches[:5])
        raise SystemExit(f"Requested image is ambiguous: {query}. Matches: {names}")
    return matches[0]


def record_matches(record: TestRecord, normalized_query: str) -> bool:
    candidates = {
        normalize_identifier(record.image_name),
        normalize_identifier(record.image_path.name),
        normalize_identifier(record.image_path.stem),
        normalize_identifier(str(record.image_path)),
    }
    return normalized_query in candidates or any(candidate.endswith(normalized_query) for candidate in candidates)


def normalize_identifier(value: str | Path) -> str:
    return str(value).replace("\\", "/").strip().lower()


def write_selection_log(
    selected: list[TestRecord],
    candidate_log: dict[str, list[dict[str, float | str]]],
    path: Path,
) -> None:
    lines = ["Dataset sample selection", ""]
    lines.append("Chosen panels:")
    for index, record in enumerate(selected):
        lines.append(f"({PANEL_LETTERS[index]}) {panel_label(index)} | {record.image_name} | {record.image_path}")
    if candidate_log:
        lines.append("")
        lines.append("Top auto-selection candidates:")
        for category, rows in candidate_log.items():
            lines.append(f"- {category}")
            for row in rows:
                lines.append(f"  {row['score']}: {row['image']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_selection_json(
    selected: list[TestRecord],
    candidate_log: dict[str, list[dict[str, float | str]]],
    path: Path,
) -> None:
    payload = {
        "selected": [
            {"panel": f"({PANEL_LETTERS[index]})", "label": panel_label(index), "image": record.image_name, "path": str(record.image_path)}
            for index, record in enumerate(selected)
        ],
        "candidates": candidate_log,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def configure_matplotlib_cache() -> None:
    import tempfile

    cache_dir = Path(os.environ.get("TMPDIR", tempfile.gettempdir())) / "wood_defect_datacentric_mpl_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir / "mplconfig"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))


if __name__ == "__main__":
    main()
