#!/usr/bin/env python3
"""Verify that a Vast.ai server is ready for controlled YOLOv8s runs.

This script is read-only with respect to datasets. It validates configured
YOLO dataset YAMLs, image/label matching, split presence, class names,
VNWoodKnot negative/background labels, and result-folder writability.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent

if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from wood_defect_datacentric.datasets.adapters import load_manifest_dataset, resolve_repo_path  # noqa: E402
from wood_defect_datacentric.scripts.dataset_stats import load_config  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXPECTED_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--project-config", type=Path, default=PROJECT_ROOT / "configs" / "project.yaml")
    parser.add_argument("--server-config", type=Path, default=PROJECT_ROOT / "configs" / "server.yaml")
    parser.add_argument("--output-json", type=Path, help="Optional explicit report path.")
    parser.add_argument("--max-label-errors", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = {**load_env(args.env_file), **os.environ}
    project_config = load_config(args.project_config)
    server_config = load_config(args.server_config) if args.server_config.exists() else {}

    results_root = Path(env.get("RESULTS_ROOT") or server_config.get("paths", {}).get("results_root") or PROJECT_ROOT / "results").expanduser()
    if not results_root.is_absolute():
        results_root = (PROJECT_ROOT / results_root).resolve()
    report: dict[str, Any] = {
        "ok": True,
        "env_file": str(args.env_file),
        "results_root": str(results_root),
        "checks": {},
        "errors": [],
        "warnings": [],
    }

    report["checks"]["results_root"] = check_writable_dir(results_root)
    if not report["checks"]["results_root"]["ok"]:
        report["errors"].append(report["checks"]["results_root"]["message"])
    if args.output_json:
        report_path = args.output_json
    elif report["checks"]["results_root"]["ok"]:
        report_path = results_root / "server_ready_report.json"
    else:
        report_path = PROJECT_ROOT / "results" / "server_ready_report.json"

    dataset_specs = [
        {
            "key": "vsb_curated",
            "env_var": "WOOD_DC_VSB_BASELINE_DATASET_YAML",
            "fallback": server_config.get("paths", {}).get("vsb_dataset_yaml"),
            "expected_classes": project_config["datasets"]["vsb_curated"]["classes"],
            "manifest_key": "vsb_manifest_path",
            "negative_source_category": None,
        },
        {
            "key": "vnwoodknot",
            "env_var": "WOOD_DC_VN_BASELINE_DATASET_YAML",
            "fallback": server_config.get("paths", {}).get("vnwoodknot_dataset_yaml"),
            "expected_classes": project_config["datasets"]["vnwoodknot"]["positive_classes"],
            "manifest_key": "vnwoodknot_manifest_path",
            "negative_source_category": project_config["datasets"]["vnwoodknot"]["negative_class"],
        },
    ]

    for spec in dataset_specs:
        yaml_value = env.get(spec["env_var"]) or spec["fallback"]
        if not yaml_value:
            result = {"ok": False, "error": f"Missing {spec['env_var']} and no server config fallback."}
        else:
            result = check_yolo_dataset(
                dataset_key=spec["key"],
                dataset_yaml=Path(str(yaml_value)).expanduser(),
                expected_classes=tuple(spec["expected_classes"]),
                max_label_errors=args.max_label_errors,
            )
        report["checks"][spec["key"]] = result
        collect_messages(report, result)

        manifest_path = resolve_repo_path(project_config["paths"][spec["manifest_key"]], PROJECT_ROOT)
        manifest_result = check_manifest(
            dataset_key=spec["key"],
            manifest_path=manifest_path,
            expected_classes=tuple(spec["expected_classes"]),
            negative_source_category=spec["negative_source_category"],
        )
        report["checks"][f"{spec['key']}_manifest"] = manifest_result
        collect_messages(report, manifest_result)

    report["ok"] = not report["errors"]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {report_path}")
    print("SERVER_READY=1" if report["ok"] else "SERVER_READY=0")
    if report["errors"]:
        print("Errors:")
        for error in report["errors"]:
            print(f"- {error}")
    raise SystemExit(0 if report["ok"] else 1)


def load_env(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def check_writable_dir(path: Path) -> dict[str, Any]:
    try:
        path.mkdir(parents=True, exist_ok=True)
        probe = path / ".server_ready_write_test"
        probe.write_text("ok\n", encoding="utf-8")
        probe.unlink()
    except Exception as exc:
        return {"ok": False, "path": str(path), "message": f"Results folder is not writable: {path} ({exc})"}
    return {"ok": True, "path": str(path), "message": "Results folder is writable."}


def check_yolo_dataset(
    *,
    dataset_key: str,
    dataset_yaml: Path,
    expected_classes: tuple[str, ...],
    max_label_errors: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": True,
        "dataset_yaml": str(dataset_yaml),
        "class_names": [],
        "splits": {},
        "errors": [],
        "warnings": [],
    }
    if not dataset_yaml.exists():
        result["ok"] = False
        result["errors"].append(f"{dataset_key}: dataset YAML missing: {dataset_yaml}")
        return result

    raw = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")) or {}
    names = normalize_names(raw.get("names"))
    result["class_names"] = names
    if tuple(names) != expected_classes:
        result["ok"] = False
        result["errors"].append(
            f"{dataset_key}: class names mismatch. expected={list(expected_classes)} observed={names}"
        )

    for split in EXPECTED_SPLITS:
        split_path = resolve_yolo_split_path(dataset_yaml, raw, split)
        split_result = check_split(
            split=split,
            split_path=split_path,
            num_classes=len(names),
            max_label_errors=max_label_errors,
        )
        result["splits"][split] = split_result
        if not split_result["ok"]:
            result["ok"] = False
        result["errors"].extend(split_result["errors"])
        result["warnings"].extend(split_result["warnings"])

    if dataset_key == "vnwoodknot":
        empty_by_split = {
            split: split_result["empty_label_files"]
            for split, split_result in result["splits"].items()
        }
        result["vnwoodknot_negative_check"] = {
            "ok": all(empty_by_split.get(split, 0) > 0 for split in EXPECTED_SPLITS),
            "empty_label_files_by_split": empty_by_split,
            "message": "VNWoodKnot has empty label files in all splits."
            if all(empty_by_split.get(split, 0) > 0 for split in EXPECTED_SPLITS)
            else "VNWoodKnot is missing empty/background label files in at least one split.",
        }
        if not result["vnwoodknot_negative_check"]["ok"]:
            result["ok"] = False
            result["errors"].append(f"{dataset_key}: missing negative/background label files in one or more splits.")
    return result


def normalize_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names)]
    if isinstance(names, list):
        return [str(name) for name in names]
    return []


def resolve_yolo_split_path(dataset_yaml: Path, raw: dict[str, Any], split: str) -> Path | None:
    value = raw.get(split)
    if value is None:
        return None
    if isinstance(value, list):
        value = value[0] if value else ""
    split_path = Path(str(value)).expanduser()
    if split_path.is_absolute():
        return split_path
    base = Path(str(raw.get("path") or dataset_yaml.parent)).expanduser()
    if not base.is_absolute():
        base = dataset_yaml.parent / base
    return (base / split_path).resolve()


def check_split(
    *,
    split: str,
    split_path: Path | None,
    num_classes: int,
    max_label_errors: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": True,
        "path": str(split_path) if split_path else "",
        "num_images": 0,
        "missing_labels": 0,
        "empty_label_files": 0,
        "label_format_errors": 0,
        "class_id_errors": 0,
        "errors": [],
        "warnings": [],
        "sample_label_errors": [],
    }
    if split_path is None:
        result["ok"] = False
        result["errors"].append(f"Missing split value in dataset YAML: {split}")
        return result
    if not split_path.exists():
        result["ok"] = False
        result["errors"].append(f"Split path does not exist: {split} -> {split_path}")
        return result

    image_paths = sorted(path for path in split_path.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    result["num_images"] = len(image_paths)
    if not image_paths:
        result["ok"] = False
        result["errors"].append(f"No images found for split {split}: {split_path}")
        return result

    for image_path in image_paths:
        label_path = label_for_image(image_path)
        if not label_path.exists():
            result["missing_labels"] += 1
            continue
        if label_path.stat().st_size == 0:
            result["empty_label_files"] += 1
            continue
        errors = validate_label_file(label_path, num_classes)
        if errors["format_errors"] or errors["class_id_errors"]:
            result["label_format_errors"] += errors["format_errors"]
            result["class_id_errors"] += errors["class_id_errors"]
            if len(result["sample_label_errors"]) < max_label_errors:
                result["sample_label_errors"].append({"label": str(label_path), **errors})

    if result["missing_labels"] > 0:
        result["ok"] = False
        result["errors"].append(f"{split}: {result['missing_labels']} images are missing label files.")
    if result["label_format_errors"] > 0 or result["class_id_errors"] > 0:
        result["ok"] = False
        result["errors"].append(
            f"{split}: label validation failed "
            f"(format_errors={result['label_format_errors']}, class_id_errors={result['class_id_errors']})."
        )
    return result


def label_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        index = parts.index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def validate_label_file(label_path: Path, num_classes: int) -> dict[str, Any]:
    format_errors = 0
    class_id_errors = 0
    bad_lines: list[str] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            format_errors += 1
            bad_lines.append(f"{line_number}: expected 5 columns")
            continue
        try:
            class_id = int(float(parts[0]))
            coords = [float(value) for value in parts[1:]]
        except ValueError:
            format_errors += 1
            bad_lines.append(f"{line_number}: non-numeric value")
            continue
        if class_id < 0 or class_id >= num_classes:
            class_id_errors += 1
            bad_lines.append(f"{line_number}: class id {class_id} outside [0, {num_classes - 1}]")
        if any(value < 0.0 or value > 1.0 for value in coords):
            format_errors += 1
            bad_lines.append(f"{line_number}: bbox coordinate outside [0, 1]")
        if coords[2] <= 0.0 or coords[3] <= 0.0:
            format_errors += 1
            bad_lines.append(f"{line_number}: bbox width/height must be positive")
    return {
        "format_errors": format_errors,
        "class_id_errors": class_id_errors,
        "bad_lines": bad_lines[:5],
    }


def check_manifest(
    *,
    dataset_key: str,
    manifest_path: Path,
    expected_classes: tuple[str, ...],
    negative_source_category: str | None,
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ok": True,
        "manifest": str(manifest_path),
        "records": 0,
        "split_counts": {},
        "observed_classes": [],
        "knot_free_by_split": {},
        "errors": [],
        "warnings": [],
    }
    if not manifest_path.exists():
        result["ok"] = False
        result["errors"].append(f"{dataset_key}: manifest missing: {manifest_path}")
        return result

    loaded = load_manifest_dataset(
        dataset_key=dataset_key,
        manifest_path=manifest_path,
        expected_classes=expected_classes,
        negative_source_category=negative_source_category,
        check_image_exists=False,
    )
    result["records"] = len(loaded.records)
    split_counts: dict[str, int] = defaultdict(int)
    knot_free_by_split: dict[str, int] = defaultdict(int)
    observed_classes = set()
    knot_free_with_boxes = 0
    for record in loaded.records:
        split_counts[record.split] += 1
        if record.is_knot_free:
            knot_free_by_split[record.split] += 1
            if record.annotations:
                knot_free_with_boxes += 1
        for annotation in record.annotations:
            observed_classes.add(annotation.class_name)
    result["split_counts"] = dict(sorted(split_counts.items()))
    result["observed_classes"] = sorted(observed_classes)
    result["knot_free_by_split"] = dict(sorted(knot_free_by_split.items()))
    result["warnings"] = list(loaded.warnings)

    if dataset_key == "vnwoodknot":
        missing_negative_splits = [split for split in EXPECTED_SPLITS if knot_free_by_split.get(split, 0) == 0]
        if missing_negative_splits:
            result["ok"] = False
            result["errors"].append(f"VNWoodKnot manifest missing knot_free records in splits: {missing_negative_splits}")
        if knot_free_with_boxes:
            result["ok"] = False
            result["errors"].append(f"VNWoodKnot manifest has {knot_free_with_boxes} knot_free records with boxes.")
    return result


def collect_messages(report: dict[str, Any], result: dict[str, Any]) -> None:
    report["errors"].extend(result.get("errors", []))
    report["warnings"].extend(result.get("warnings", []))


if __name__ == "__main__":
    main()
