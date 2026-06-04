"""Dataset statistics and split verification utilities."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import csv
import statistics
from typing import Iterable

from wood_defect_datacentric.datasets.adapters import (
    DatasetLoadResult,
    DatasetRecord,
    UNKNOWN_SPLIT,
)


EXPECTED_SPLITS = ("train", "val", "test")


@dataclass(frozen=True)
class SplitStats:
    dataset_key: str
    dataset_name: str
    split: str
    num_images: int
    num_positive_images: int
    num_negative_images: int
    num_boxes: int
    mean_bbox_area: float
    median_bbox_area: float
    missing_image_paths: int
    invalid_records: int
    invalid_boxes_declared: int
    knot_free_images: int
    knot_free_with_boxes: int


def group_records_by_split(records: Iterable[DatasetRecord]) -> dict[str, list[DatasetRecord]]:
    grouped: dict[str, list[DatasetRecord]] = defaultdict(list)
    for record in records:
        grouped[record.split].append(record)
    return dict(sorted(grouped.items()))


def compute_split_stats(load_result: DatasetLoadResult) -> list[SplitStats]:
    stats: list[SplitStats] = []
    for split, records in group_records_by_split(load_result.records).items():
        areas = [ann.area_norm for record in records for ann in record.annotations]
        stats.append(
            SplitStats(
                dataset_key=load_result.dataset_key,
                dataset_name=load_result.dataset_name,
                split=split,
                num_images=len(records),
                num_positive_images=sum(record.is_positive for record in records),
                num_negative_images=sum(record.is_negative for record in records),
                num_boxes=sum(len(record.annotations) for record in records),
                mean_bbox_area=statistics.fmean(areas) if areas else 0.0,
                median_bbox_area=statistics.median(areas) if areas else 0.0,
                missing_image_paths=sum("missing_image_path" in record.issues for record in records),
                invalid_records=sum(bool(record.issues) for record in records),
                invalid_boxes_declared=sum(record.declared_invalid_boxes for record in records),
                knot_free_images=sum(record.is_knot_free for record in records),
                knot_free_with_boxes=sum(record.is_knot_free and bool(record.annotations) for record in records),
            )
        )
    return stats


def compute_class_distribution(load_result: DatasetLoadResult) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    grouped = group_records_by_split(load_result.records)

    for split, records in grouped.items():
        box_counter: Counter[str] = Counter()
        image_counter: Counter[str] = Counter()
        total_boxes = 0

        for record in records:
            classes_in_image = set()
            for ann in record.annotations:
                box_counter[ann.class_name] += 1
                classes_in_image.add(ann.class_name)
                total_boxes += 1
            for class_name in classes_in_image:
                image_counter[class_name] += 1

        class_names = sorted(set(load_result.expected_classes) | set(box_counter))
        for class_name in class_names:
            box_count = box_counter[class_name]
            rows.append(
                {
                    "dataset_key": load_result.dataset_key,
                    "dataset_name": load_result.dataset_name,
                    "row_type": "class_distribution",
                    "split": split,
                    "class_name": class_name,
                    "num_images": "",
                    "num_positive_images": "",
                    "num_negative_images": "",
                    "num_boxes": box_count,
                    "num_images_with_class": image_counter[class_name],
                    "mean_bbox_area": "",
                    "median_bbox_area": "",
                    "class_box_share": round(box_count / total_boxes, 8) if total_boxes else 0.0,
                    "missing_image_paths": "",
                    "invalid_records": "",
                    "invalid_boxes_declared": "",
                    "knot_free_images": "",
                    "knot_free_with_boxes": "",
                    "notes": "",
                }
            )
    return rows


def build_csv_rows(load_result: DatasetLoadResult) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for stat in compute_split_stats(load_result):
        rows.append(
            {
                "dataset_key": stat.dataset_key,
                "dataset_name": stat.dataset_name,
                "row_type": "split_summary",
                "split": stat.split,
                "class_name": "",
                "num_images": stat.num_images,
                "num_positive_images": stat.num_positive_images,
                "num_negative_images": stat.num_negative_images,
                "num_boxes": stat.num_boxes,
                "num_images_with_class": "",
                "mean_bbox_area": round(stat.mean_bbox_area, 8),
                "median_bbox_area": round(stat.median_bbox_area, 8),
                "class_box_share": "",
                "missing_image_paths": stat.missing_image_paths,
                "invalid_records": stat.invalid_records,
                "invalid_boxes_declared": stat.invalid_boxes_declared,
                "knot_free_images": stat.knot_free_images,
                "knot_free_with_boxes": stat.knot_free_with_boxes,
                "notes": _split_notes(load_result, stat.split),
            }
        )
    rows.extend(compute_class_distribution(load_result))
    return rows


def write_stats_csv(load_result: DatasetLoadResult, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = build_csv_rows(load_result)
    fieldnames = [
        "dataset_key",
        "dataset_name",
        "row_type",
        "split",
        "class_name",
        "num_images",
        "num_positive_images",
        "num_negative_images",
        "num_boxes",
        "num_images_with_class",
        "mean_bbox_area",
        "median_bbox_area",
        "class_box_share",
        "missing_image_paths",
        "invalid_records",
        "invalid_boxes_declared",
        "knot_free_images",
        "knot_free_with_boxes",
        "notes",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def verify_splits(load_result: DatasetLoadResult) -> list[dict[str, str | int]]:
    rows: list[dict[str, str | int]] = []
    grouped = group_records_by_split(load_result.records)
    present = set(grouped)

    for split in EXPECTED_SPLITS:
        rows.append(
            {
                "dataset_key": load_result.dataset_key,
                "check": f"split_present:{split}",
                "status": "pass" if split in present else "warn",
                "count": len(grouped.get(split, [])),
                "message": "split is present" if split in present else "split is absent in current source",
            }
        )

    if UNKNOWN_SPLIT in present:
        rows.append(
            {
                "dataset_key": load_result.dataset_key,
                "check": "unspecified_split",
                "status": "warn",
                "count": len(grouped[UNKNOWN_SPLIT]),
                "message": "records without split labels are preserved as unspecified",
            }
        )

    all_records = list(load_result.records)
    missing_images = sum("missing_image_path" in record.issues for record in all_records)
    rows.append(
        {
            "dataset_key": load_result.dataset_key,
            "check": "image_paths_exist",
            "status": "pass" if missing_images == 0 else "warn",
            "count": missing_images,
            "message": "all image paths exist" if missing_images == 0 else "some image paths are not accessible locally",
        }
    )

    bad_boxes = sum(
        any(issue in record.issues for issue in ("non_positive_bbox", "bbox_outside_normalized_range", "non_positive_bbox_area"))
        for record in all_records
    )
    rows.append(
        {
            "dataset_key": load_result.dataset_key,
            "check": "bbox_validity",
            "status": "pass" if bad_boxes == 0 else "fail",
            "count": bad_boxes,
            "message": "all normalized boxes are valid" if bad_boxes == 0 else "invalid normalized boxes found",
        }
    )

    unexpected_classes = sorted(
        {
            issue.split(":", 1)[1]
            for record in all_records
            for issue in record.issues
            if issue.startswith("unexpected_class:")
        }
    )
    unexpected_class_boxes = sum(
        1
        for record in all_records
        for ann in record.annotations
        if ann.class_name in unexpected_classes
    )
    rows.append(
        {
            "dataset_key": load_result.dataset_key,
            "check": "configured_class_set",
            "status": "pass" if unexpected_class_boxes == 0 else "warn",
            "count": unexpected_class_boxes,
            "message": "all labels are in configured class set"
            if unexpected_class_boxes == 0
            else f"labels outside configured class set: {', '.join(unexpected_classes)}",
        }
    )

    if load_result.dataset_key == "vnwoodknot":
        for split in EXPECTED_SPLITS:
            count = sum(record.is_knot_free for record in grouped.get(split, []))
            rows.append(
                {
                    "dataset_key": load_result.dataset_key,
                    "check": f"knot_free_retained:{split}",
                    "status": "pass" if count > 0 else "fail",
                    "count": count,
                    "message": "knot_free negative images retained" if count > 0 else "missing knot_free negatives",
                }
            )
        knot_free_with_boxes = sum(record.is_knot_free and bool(record.annotations) for record in all_records)
        rows.append(
            {
                "dataset_key": load_result.dataset_key,
                "check": "knot_free_has_no_boxes",
                "status": "pass" if knot_free_with_boxes == 0 else "fail",
                "count": knot_free_with_boxes,
                "message": "knot_free records are background-only" if knot_free_with_boxes == 0 else "knot_free records with boxes found",
            }
        )

    return rows


def write_verification_csv(load_results: Iterable[DatasetLoadResult], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["dataset_key", "check", "status", "count", "message"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for load_result in load_results:
            writer.writerows(verify_splits(load_result))


def render_markdown_report(
    *,
    load_results: Iterable[DatasetLoadResult],
    output_paths: dict[str, Path],
) -> str:
    results = list(load_results)
    lines: list[str] = [
        "# Dataset Audit",
        "",
        "This report was generated by the read-only data-centric dataset adapters. It does not alter datasets, regenerate splits, drop negative images, or resize images offline.",
        "",
        "## Sources",
        "",
        "| Dataset | Source | Records | Classes |",
        "|---|---:|---:|---|",
    ]

    for result in results:
        lines.append(
            f"| {result.dataset_key} | `{result.source_path}` | {len(result.records)} | {', '.join(result.expected_classes)} |"
        )

    lines.extend(["", "## Split Summary", "", "| Dataset | Split | Images | Positive images | Negative images | Boxes | Mean box area | Median box area | Missing image paths | Knot-free images |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for result in results:
        for stat in compute_split_stats(result):
            lines.append(
                "| "
                f"{result.dataset_key} | {stat.split} | {stat.num_images} | {stat.num_positive_images} | "
                f"{stat.num_negative_images} | {stat.num_boxes} | {stat.mean_bbox_area:.6f} | "
                f"{stat.median_bbox_area:.6f} | {stat.missing_image_paths} | {stat.knot_free_images} |"
            )

    lines.extend(["", "## Class Distribution", ""])
    for result in results:
        lines.extend([f"### {result.dataset_key}", "", "| Split | Class | Boxes | Images with class | Box share |", "|---|---|---:|---:|---:|"])
        for row in compute_class_distribution(result):
            lines.append(
                f"| {row['split']} | {row['class_name']} | {row['num_boxes']} | {row['num_images_with_class']} | {float(row['class_box_share']):.4f} |"
            )
        lines.append("")

    lines.extend(["## Split Verification", "", "| Dataset | Check | Status | Count | Message |", "|---|---|---:|---:|---|"])
    for result in results:
        for row in verify_splits(result):
            lines.append(
                f"| {row['dataset_key']} | {row['check']} | {row['status']} | {row['count']} | {row['message']} |"
            )

    warnings = [warning for result in results for warning in result.warnings]
    if warnings:
        lines.extend(["", "## Warnings", ""])
        for warning in warnings:
            lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            f"- VSB CSV: `{output_paths['vsb_csv']}`",
            f"- VNWoodKnot CSV: `{output_paths['vn_csv']}`",
            f"- Split verification CSV: `{output_paths['verification_csv']}`",
            "",
            "## Notes",
            "",
            "- VSB records with missing split labels are preserved as `unspecified`; no split regeneration is performed.",
            "- VNWoodKnot `source_category=knot_free` records are explicitly marked as negative/background-only images.",
            "- Image existence checks report local filesystem availability only; missing paths may mean the external dataset volume is not mounted at the manifest path.",
        ]
    )
    return "\n".join(lines) + "\n"


def write_markdown_report(report_text: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report_text, encoding="utf-8")


def _split_notes(load_result: DatasetLoadResult, split: str) -> str:
    if split == UNKNOWN_SPLIT:
        return "split label missing in source; preserved without regeneration"
    if load_result.dataset_key == "vnwoodknot":
        return "validation split normalized to val for reporting" if split == "val" else ""
    return ""
