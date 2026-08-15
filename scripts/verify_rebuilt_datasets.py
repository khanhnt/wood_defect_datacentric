#!/usr/bin/env python3
"""Verify freshly materialized canonical and variant YOLO datasets."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test")


EXPECTED = {
    "vnwoodknot": {
        "train": {"images": 1059, "labels": 1059, "empty_labels": 350, "boxes": 714},
        "val": {"images": 226, "labels": 226, "empty_labels": 75, "boxes": 151},
        "test": {"images": 229, "labels": 229, "empty_labels": 75, "boxes": 155},
    },
    "vsb_rarefirst": {
        "train": {"images": 7679, "labels": 7679, "empty_labels": 2297, "boxes": 9346},
        "val": {"images": 977, "labels": 977, "empty_labels": 276, "boxes": 1146},
        "test": {"images": 972, "labels": 972, "empty_labels": 276, "boxes": 1173},
    },
    "vsb_strict_clean": {
        "train": {"images": 0, "labels": 0, "empty_labels": 0, "boxes": 0},
        "val": {"images": 0, "labels": 0, "empty_labels": 0, "boxes": 0},
        "test": {"images": 5976, "labels": 5976, "empty_labels": 5976, "boxes": 0},
    },
}
EXPECTED_NAMES = {
    "vnwoodknot": ["live_knot", "dead_knot"],
    "vsb_rarefirst": [
        "live_knot",
        "dead_knot",
        "resin",
        "knot_with_crack",
        "crack",
        "marrow",
        "knot_missing",
    ],
    "vsb_strict_clean": [
        "live_knot",
        "dead_knot",
        "resin",
        "knot_with_crack",
        "crack",
        "marrow",
        "knot_missing",
    ],
}


@dataclass(frozen=True)
class DatasetIdentity:
    dataset: str
    variant: str
    seed: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("revised/datasets_rebuilt"))
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("revised/datasets_rebuilt/reports/verification_gate.csv"),
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=Path("revised/datasets_rebuilt/reports/verification_gate.md"),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=tuple(EXPECTED),
        default=list(EXPECTED),
        help="Verify only the selected datasets; useful before strict-clean pixels arrive.",
    )
    return parser.parse_args()


def normalize_names(raw: Any) -> list[str]:
    if isinstance(raw, list):
        return [str(value) for value in raw]
    if isinstance(raw, dict):
        return [str(raw[key]) for key in sorted(raw, key=lambda value: int(value))]
    raise ValueError("dataset.yaml must contain names as a list or mapping")


def identify_dataset(root: Path, yaml_path: Path) -> DatasetIdentity:
    rel = yaml_path.relative_to(root)
    parts = rel.parts
    if parts[0] == "canonical":
        return DatasetIdentity(parts[1], "canonical", "")
    if parts[0] == "variants" and parts[2] == "preprocessing":
        return DatasetIdentity(parts[1], parts[3], "")
    if parts[0] == "variants" and parts[2] == "augmentation":
        return DatasetIdentity(parts[1], parts[4], parts[3].removeprefix("seed"))
    raise ValueError(f"Unrecognized rebuilt-dataset path: {rel}")


def resolve_dataset_root(yaml_path: Path, config: dict[str, Any]) -> Path:
    value = Path(str(config.get("path", yaml_path.parent))).expanduser()
    if not value.is_absolute():
        value = yaml_path.parent / value
    return value.resolve()


def resolve_split(root: Path, config: dict[str, Any], split: str) -> Path:
    value = config.get(split, f"images/{split}")
    if isinstance(value, list):
        value = value[0] if value else f"images/{split}"
    path = Path(str(value)).expanduser()
    return path if path.is_absolute() else (root / path)


def label_path(image_path: Path, images_dir: Path, root: Path, split: str) -> Path:
    return (root / "labels" / split / image_path.relative_to(images_dir)).with_suffix(".txt")


def inspect_split(
    *,
    images_dir: Path,
    root: Path,
    split: str,
    class_count: int,
) -> dict[str, Any]:
    images = sorted(path for path in images_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS) if images_dir.exists() else []
    labels_dir = root / "labels" / split
    labels = sorted(labels_dir.rglob("*.txt")) if labels_dir.exists() else []
    expected_labels = {label_path(path, images_dir, root, split) for path in images}
    missing_labels = [path for path in expected_labels if not path.exists()]
    orphan_labels = [path for path in labels if path not in expected_labels]
    empty_labels = 0
    boxes = 0
    invalid_lines = 0
    for path in labels:
        lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if not lines:
            empty_labels += 1
        boxes += len(lines)
        for line in lines:
            fields = line.split()
            try:
                class_id = int(float(fields[0]))
                coords = [float(value) for value in fields[1:]]
                valid = (
                    len(fields) == 5
                    and 0 <= class_id < class_count
                    and len(coords) == 4
                    and all(0.0 <= value <= 1.0 for value in coords)
                    and coords[2] > 0.0
                    and coords[3] > 0.0
                )
            except (IndexError, ValueError):
                valid = False
            if not valid:
                invalid_lines += 1
    return {
        "images": len(images),
        "labels": len(labels),
        "empty_labels": empty_labels,
        "boxes": boxes,
        "missing_labels": len(missing_labels),
        "orphan_labels": len(orphan_labels),
        "invalid_label_lines": invalid_lines,
        "images_dir": str(images_dir),
    }


def hardlink_audit(yaml_path: Path, split: str, rebuilt_root: Path) -> tuple[str, int, int]:
    report_path = yaml_path.parent / "materialization_report.json"
    if not report_path.exists():
        return "n/a", 0, 0
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("passthrough_mode") != "hardlink" or split not in report.get("passthrough_splits", []):
        return "n/a", 0, 0
    source_yaml = Path(report["source_yaml"]).expanduser().resolve()
    if not source_yaml.exists():
        portable = str(report["source_yaml"]).replace("\\", "/")
        marker = "/revised/datasets_rebuilt/"
        if marker in portable:
            source_yaml = rebuilt_root / portable.split(marker, 1)[1]
    if not source_yaml.exists():
        return "FAIL", 0, 1
    source_config = yaml.safe_load(source_yaml.read_text(encoding="utf-8"))
    output_config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    source_root = resolve_dataset_root(source_yaml, source_config)
    output_root = resolve_dataset_root(yaml_path, output_config)
    source_images = resolve_split(source_root, source_config, split)
    output_images = resolve_split(output_root, output_config, split)
    checked = 0
    mismatches = 0
    if output_images.exists():
        for output_path in output_images.rglob("*"):
            if output_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            source_path = source_images / output_path.relative_to(output_images)
            checked += 1
            if not source_path.exists() or (source_path.stat().st_dev, source_path.stat().st_ino) != (
                output_path.stat().st_dev,
                output_path.stat().st_ino,
            ):
                mismatches += 1
    source_labels = source_root / "labels" / split
    output_labels = output_root / "labels" / split
    if output_labels.exists():
        for output_path in output_labels.rglob("*.txt"):
            source_path = source_labels / output_path.relative_to(output_labels)
            checked += 1
            if not source_path.exists() or (source_path.stat().st_dev, source_path.stat().st_ino) != (
                output_path.stat().st_dev,
                output_path.stat().st_ino,
            ):
                mismatches += 1
    return ("PASS" if mismatches == 0 else "FAIL"), checked, mismatches


def make_row(root: Path, yaml_path: Path, split: str) -> dict[str, Any]:
    identity = identify_dataset(root, yaml_path)
    config = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = normalize_names(config["names"])
    dataset_root = resolve_dataset_root(yaml_path, config)
    actual = inspect_split(
        images_dir=resolve_split(dataset_root, config, split),
        root=dataset_root,
        split=split,
        class_count=len(names),
    )
    expected = EXPECTED[identity.dataset][split]
    hardlink_status, hardlinks_checked, hardlink_mismatches = hardlink_audit(yaml_path, split, root)
    status = "PASS"
    reasons: list[str] = []
    if names != EXPECTED_NAMES[identity.dataset]:
        status = "FAIL"
        reasons.append(f"class_names:{names}!={EXPECTED_NAMES[identity.dataset]}")
    for key in ("images", "labels", "empty_labels", "boxes"):
        if actual[key] != expected[key]:
            status = "FAIL"
            reasons.append(f"{key}:{actual[key]}!={expected[key]}")
    for key in ("missing_labels", "orphan_labels", "invalid_label_lines"):
        if actual[key] != 0:
            status = "FAIL"
            reasons.append(f"{key}:{actual[key]}")
    if hardlink_status == "FAIL":
        status = "FAIL"
        reasons.append(f"hardlink_mismatches:{hardlink_mismatches}")
    return {
        "dataset": identity.dataset,
        "variant": identity.variant,
        "seed": identity.seed,
        "split": split,
        "dataset_yaml": str(yaml_path.resolve()),
        "class_names": "|".join(names),
        "expected_class_names": "|".join(EXPECTED_NAMES[identity.dataset]),
        **{f"expected_{key}": value for key, value in expected.items()},
        **actual,
        "hardlink_status": hardlink_status,
        "hardlinks_checked": hardlinks_checked,
        "hardlink_mismatches": hardlink_mismatches,
        "status": status,
        "failure_reason": ";".join(reasons),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    failed = [row for row in rows if row["status"] != "PASS"]
    lines = [
        "# Rebuilt Dataset Verification Gate",
        "",
        f"Overall: **{'PASS' if not failed else 'FAIL'}** ({len(rows) - len(failed)}/{len(rows)} rows passed)",
        "",
        "| Dataset | Variant | Seed | Split | Images | Labels | Empty | Boxes | Hardlink | Status |",
        "|---|---|---:|---|---:|---:|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['variant']} | {row['seed'] or '-'} | {row['split']} | "
            f"{row['images']}/{row['expected_images']} | {row['labels']}/{row['expected_labels']} | "
            f"{row['empty_labels']}/{row['expected_empty_labels']} | {row['boxes']}/{row['expected_boxes']} | "
            f"{row['hardlink_status']} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Manuscript Acceptance Counts",
            "",
            "- VNWoodKnot: 1,059/226/229 images; test = 154 defective + 75 clean, 155 boxes.",
            "- VSB rare-first: 7,679/977/972 tiles.",
            "- VSB strict-clean: 1,992 sources materialized as 5,976 empty-label tiles.",
            "",
        ]
    )
    if failed:
        lines.extend(["## Failures", ""])
        lines.extend(f"- {row['dataset']} {row['variant']} seed={row['seed']} {row['split']}: {row['failure_reason']}" for row in failed)
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = args.root.expanduser().resolve()
    yaml_paths = sorted(root.glob("canonical/*/dataset.yaml"))
    yaml_paths.extend(sorted(root.glob("variants/*/preprocessing/*/dataset.yaml")))
    yaml_paths.extend(sorted(root.glob("variants/*/augmentation/seed*/*/dataset.yaml")))
    selected = set(args.datasets)
    yaml_paths = [path for path in yaml_paths if identify_dataset(root, path).dataset in selected]
    if not yaml_paths:
        raise SystemExit(f"No rebuilt dataset YAML files found under {root}")
    rows = [make_row(root, yaml_path, split) for yaml_path in yaml_paths for split in SPLITS]
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows)
    failed = [row for row in rows if row["status"] != "PASS"]
    print(f"Wrote: {args.output_csv}")
    print(f"Wrote: {args.output_md}")
    print(f"VERIFICATION GATE: {'PASS' if not failed else 'FAIL'} ({len(rows) - len(failed)}/{len(rows)})")
    if failed:
        for row in failed:
            print(
                f"FAIL {row['dataset']} {row['variant']} seed={row['seed'] or '-'} "
                f"split={row['split']}: {row['failure_reason']}"
            )
        raise SystemExit(1)


if __name__ == "__main__":
    main()
