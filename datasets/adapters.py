"""Dataset adapters for data-centric wood-defect experiments.

The adapters are intentionally read-only. They preserve existing split labels
and expose normalized split names only for reporting consistency.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


UNKNOWN_SPLIT = "unspecified"
SPLIT_ALIASES = {
    "validation": "val",
    "valid": "val",
}


@dataclass(frozen=True)
class Annotation:
    class_name: str
    class_id: int | None
    bbox_xyxy_norm: tuple[float, float, float, float]
    area_norm: float
    source_label: str | None = None


@dataclass(frozen=True)
class DatasetRecord:
    dataset_key: str
    dataset_name: str
    image_id: str
    image_path: Path
    raw_split: str | None
    split: str
    source_category: str | None
    width: int | None
    height: int | None
    annotations: tuple[Annotation, ...]
    is_empty: bool
    empty_reason: str | None
    is_knot_free: bool = False
    issues: tuple[str, ...] = field(default_factory=tuple)
    declared_invalid_boxes: int = 0

    @property
    def is_negative(self) -> bool:
        return self.is_knot_free or self.is_empty or len(self.annotations) == 0

    @property
    def is_positive(self) -> bool:
        return not self.is_negative


@dataclass(frozen=True)
class DatasetLoadResult:
    dataset_key: str
    dataset_name: str
    source_path: Path
    records: tuple[DatasetRecord, ...]
    expected_classes: tuple[str, ...]
    warnings: tuple[str, ...]


def normalize_split(raw_split: Any) -> str:
    if raw_split is None:
        return UNKNOWN_SPLIT
    split = str(raw_split).strip()
    if not split:
        return UNKNOWN_SPLIT
    return SPLIT_ALIASES.get(split.lower(), split.lower())


def resolve_repo_path(path_value: str | Path, repo_root: Path) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    return (repo_root / path).resolve()


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}: {exc}") from exc


def _coerce_annotation(raw: dict[str, Any]) -> Annotation:
    bbox = raw.get("bbox_xyxy_norm")
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        bbox_tuple = (0.0, 0.0, 0.0, 0.0)
    else:
        bbox_tuple = tuple(float(value) for value in bbox)

    area = raw.get("bbox_area_norm")
    if area is None:
        x1, y1, x2, y2 = bbox_tuple
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    class_id = raw.get("class_id")
    if class_id is not None:
        class_id = int(class_id)

    return Annotation(
        class_name=str(raw.get("class_name", "")),
        class_id=class_id,
        bbox_xyxy_norm=bbox_tuple,
        area_norm=float(area),
        source_label=raw.get("source_label"),
    )


def _record_issues(
    record: DatasetRecord,
    expected_classes: set[str],
    check_image_exists: bool,
) -> tuple[str, ...]:
    issues: list[str] = []

    if check_image_exists and not record.image_path.exists():
        issues.append("missing_image_path")

    if record.width is None or record.height is None or record.width <= 0 or record.height <= 0:
        issues.append("invalid_image_size")

    if record.is_knot_free and record.annotations:
        issues.append("knot_free_has_annotations")

    if record.is_empty and record.annotations:
        issues.append("empty_record_has_annotations")

    for ann in record.annotations:
        if not ann.class_name:
            issues.append("missing_class_name")
        elif expected_classes and ann.class_name not in expected_classes:
            issues.append(f"unexpected_class:{ann.class_name}")

        x1, y1, x2, y2 = ann.bbox_xyxy_norm
        if x2 <= x1 or y2 <= y1:
            issues.append("non_positive_bbox")
        if min(x1, y1, x2, y2) < 0.0 or max(x1, y1, x2, y2) > 1.0:
            issues.append("bbox_outside_normalized_range")
        if ann.area_norm <= 0.0:
            issues.append("non_positive_bbox_area")

    return tuple(sorted(set(issues)))


def load_manifest_dataset(
    *,
    dataset_key: str,
    manifest_path: Path,
    expected_classes: Iterable[str],
    negative_source_category: str | None = None,
    check_image_exists: bool = True,
) -> DatasetLoadResult:
    expected_classes_tuple = tuple(expected_classes)
    expected_class_set = set(expected_classes_tuple)
    warnings: list[str] = []
    records: list[DatasetRecord] = []

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    for raw in read_jsonl(manifest_path):
        raw_split = raw.get("split")
        split = normalize_split(raw_split)
        annotations = tuple(_coerce_annotation(item) for item in raw.get("annotations", []))
        source_category = raw.get("source_category")
        is_knot_free = negative_source_category is not None and source_category == negative_source_category

        record = DatasetRecord(
            dataset_key=dataset_key,
            dataset_name=str(raw.get("dataset_name", dataset_key)),
            image_id=str(raw.get("image_id", "")),
            image_path=Path(str(raw.get("image_path", ""))).expanduser(),
            raw_split=None if raw_split is None else str(raw_split),
            split=split,
            source_category=None if source_category is None else str(source_category),
            width=_optional_int(raw.get("width")),
            height=_optional_int(raw.get("height")),
            annotations=annotations,
            is_empty=bool(raw.get("is_empty", len(annotations) == 0)),
            empty_reason=raw.get("empty_reason"),
            is_knot_free=is_knot_free,
            declared_invalid_boxes=int(raw.get("num_invalid_boxes") or 0),
        )
        issues = set(raw.get("issues") or [])
        issues.update(_record_issues(record, expected_class_set, check_image_exists))
        records.append(_replace_record_issues(record, issues))

    split_names = {record.split for record in records}
    if UNKNOWN_SPLIT in split_names:
        warnings.append(
            f"{dataset_key}: {UNKNOWN_SPLIT} split records found; existing train/val/test split cannot be fully verified from this source."
        )

    unexpected_classes = sorted(
        {
            ann.class_name
            for record in records
            for ann in record.annotations
            if expected_class_set and ann.class_name not in expected_class_set
        }
    )
    if unexpected_classes:
        warnings.append(
            f"{dataset_key}: labels outside configured class set found: {', '.join(unexpected_classes)}."
        )

    if negative_source_category:
        negative_records = [record for record in records if record.source_category == negative_source_category]
        negative_splits = sorted({record.split for record in negative_records})
        if not negative_records:
            warnings.append(f"{dataset_key}: no records found for negative source category {negative_source_category!r}.")
        elif not {"train", "val", "test"}.issubset(set(negative_splits)):
            warnings.append(
                f"{dataset_key}: negative source category {negative_source_category!r} is not present in all train/val/test splits."
            )

    dataset_name = records[0].dataset_name if records else dataset_key
    return DatasetLoadResult(
        dataset_key=dataset_key,
        dataset_name=dataset_name,
        source_path=manifest_path,
        records=tuple(records),
        expected_classes=expected_classes_tuple,
        warnings=tuple(warnings),
    )


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _replace_record_issues(record: DatasetRecord, issues: Iterable[str]) -> DatasetRecord:
    return DatasetRecord(
        dataset_key=record.dataset_key,
        dataset_name=record.dataset_name,
        image_id=record.image_id,
        image_path=record.image_path,
        raw_split=record.raw_split,
        split=record.split,
        source_category=record.source_category,
        width=record.width,
        height=record.height,
        annotations=record.annotations,
        is_empty=record.is_empty,
        empty_reason=record.empty_reason,
        is_knot_free=record.is_knot_free,
        issues=tuple(sorted(set(issues))),
        declared_invalid_boxes=record.declared_invalid_boxes,
    )
