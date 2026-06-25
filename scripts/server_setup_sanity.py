#!/usr/bin/env python3
"""Server setup and dataset sanity checks.

This script is intentionally verification-only. It does not train models,
materialize datasets, alter source datasets, or overwrite the standard project
reports. Outputs go to a caller-provided setup directory.
"""

from __future__ import annotations

import argparse
import copy
import csv
import importlib
import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent

if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from wood_defect_datacentric.datasets.statistics import compute_split_stats, verify_splits  # noqa: E402
from wood_defect_datacentric.evaluation.negative_aware_eval import load_vnwoodknot_records  # noqa: E402
from wood_defect_datacentric.scripts.dataset_stats import load_config, load_configured_datasets  # noqa: E402
from wood_defect_datacentric.scripts.launch_yolo_experiment import (  # noqa: E402
    build_ultralytics_command,
    load_yaml,
    process_one,
    resolve_env_values,
    resolve_repo_path as resolve_experiment_path,
    validate_experiment,
)


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
BASELINE_EXPERIMENTS = (
    "configs/experiments/vsb_yolov8s_baseline_e50.yaml",
    "configs/experiments/vn_t0_yolov8s_baseline_e50.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-config", type=Path, default=PROJECT_ROOT / "configs" / "project.yaml")
    parser.add_argument("--server-config", type=Path, default=PROJECT_ROOT / "configs" / "server.yaml")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "server_setup_sanity",
    )
    parser.add_argument(
        "--write-launcher-dry-run",
        action="store_true",
        help=(
            "Also call the standard launcher dry-run writer through copied configs "
            "whose output_root is inside --output-dir."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    project_config = load_config(args.project_config)
    server_config = load_config(args.server_config)
    report: dict[str, Any] = {
        "project_root": str(PROJECT_ROOT),
        "output_dir": str(output_dir),
        "environment": inspect_environment(),
        "paths": inspect_paths(server_config),
    }

    datasets = load_configured_datasets(project_config)
    report["dataset_manifests"] = write_dataset_checks(datasets, output_dir)
    report["yolo_dataset_checks"] = write_yolo_dataset_checks(server_config, output_dir)
    report["baseline_dry_run_checks"] = write_baseline_dry_run_checks(
        output_dir,
        write_launcher_dry_run=args.write_launcher_dry_run,
    )
    report["synthetic_negative_eval_predictions_jsonl"] = write_empty_predictions_for_eval(output_dir)

    report_path = output_dir / "server_setup_sanity_summary.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {report_path}")
    print(f"Wrote outputs under: {output_dir}")


def inspect_environment() -> dict[str, Any]:
    env: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "disk_project_root": disk_usage(PROJECT_ROOT),
        "nvidia_smi": run_command(["nvidia-smi"]),
        "packages": {},
    }
    for module_name in ("torch", "ultralytics", "cv2", "numpy", "pandas", "PIL", "yaml", "matplotlib"):
        env["packages"][module_name] = inspect_package(module_name)

    torch_info = env["packages"].get("torch", {})
    if torch_info.get("import_ok"):
        try:
            import torch

            torch_info["cuda_available"] = bool(torch.cuda.is_available())
            torch_info["cuda_version"] = torch.version.cuda
            torch_info["cuda_device_count"] = int(torch.cuda.device_count())
            if torch.cuda.is_available():
                torch_info["cuda_devices"] = [
                    {
                        "index": index,
                        "name": torch.cuda.get_device_name(index),
                        "total_memory_gb": round(
                            torch.cuda.get_device_properties(index).total_memory / (1024**3),
                            3,
                        ),
                    }
                    for index in range(torch.cuda.device_count())
                ]
        except Exception as exc:
            torch_info["cuda_probe_error"] = str(exc)
    return env


def inspect_package(module_name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {"import_ok": False, "error": str(exc)}

    version = getattr(module, "__version__", None)
    if module_name == "PIL":
        try:
            from PIL import Image

            version = getattr(Image, "__version__", version)
        except Exception:
            pass
    return {"import_ok": True, "version": version}


def run_command(command: list[str]) -> dict[str, Any]:
    executable = shutil.which(command[0])
    if executable is None:
        return {"available": False, "returncode": None, "stdout": "", "stderr": f"{command[0]} not found"}
    try:
        completed = subprocess.run(
            command,
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=20,
        )
    except Exception as exc:
        return {"available": True, "returncode": None, "stdout": "", "stderr": str(exc)}
    return {
        "available": True,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def disk_usage(path: Path) -> dict[str, Any]:
    usage = shutil.disk_usage(path)
    return {
        "path": str(path),
        "total_gb": round(usage.total / (1024**3), 3),
        "used_gb": round(usage.used / (1024**3), 3),
        "free_gb": round(usage.free / (1024**3), 3),
    }


def inspect_paths(server_config: dict[str, Any]) -> dict[str, Any]:
    checked: dict[str, Any] = {}
    for key, value in server_config.get("paths", {}).items():
        path = Path(str(value)).expanduser()
        checked[key] = {
            "path": str(path),
            "exists": path.exists(),
            "is_dir": path.is_dir(),
            "is_file": path.is_file(),
        }
    checked["current_code_root"] = {
        "path": str(PROJECT_ROOT),
        "exists": PROJECT_ROOT.exists(),
        "is_dir": PROJECT_ROOT.is_dir(),
        "is_file": PROJECT_ROOT.is_file(),
    }
    return checked


def write_dataset_checks(datasets: tuple[Any, ...], output_dir: Path) -> dict[str, Any]:
    verification_rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {}

    for dataset in datasets:
        rows = verify_splits(dataset)
        verification_rows.extend(rows)
        stats = compute_split_stats(dataset)
        observed_classes = sorted({ann.class_name for record in dataset.records for ann in record.annotations})
        summary[dataset.dataset_key] = {
            "manifest": str(dataset.source_path),
            "records": len(dataset.records),
            "expected_classes": list(dataset.expected_classes),
            "observed_classes": observed_classes,
            "warnings": list(dataset.warnings),
            "split_stats": [stat.__dict__ for stat in stats],
            "verification": rows,
            "missing_image_paths": sum("missing_image_path" in record.issues for record in dataset.records),
            "records_with_issues": sum(bool(record.issues) for record in dataset.records),
        }

    verification_path = output_dir / "dataset_split_verification.csv"
    with verification_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset_key", "check", "status", "count", "message"])
        writer.writeheader()
        writer.writerows(verification_rows)

    summary_path = output_dir / "dataset_manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {verification_path}")
    print(f"Wrote: {summary_path}")
    return summary


def write_yolo_dataset_checks(server_config: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    checks = [
        ("vsb_curated", Path(server_config["paths"]["vsb_dataset_yaml"])),
        ("vnwoodknot", Path(server_config["paths"]["vnwoodknot_dataset_yaml"])),
    ]
    rows: list[dict[str, Any]] = []
    for dataset_key, dataset_yaml in checks:
        rows.append(check_yolo_dataset(dataset_key, dataset_yaml.expanduser()))

    output_path = output_dir / "yolo_dataset_path_checks.json"
    output_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {output_path}")
    return rows


def check_yolo_dataset(dataset_key: str, dataset_yaml: Path) -> dict[str, Any]:
    row: dict[str, Any] = {
        "dataset_key": dataset_key,
        "dataset_yaml": str(dataset_yaml),
        "exists": dataset_yaml.exists(),
        "status": "missing_dataset_yaml",
        "class_names": [],
        "split_checks": [],
    }
    if not dataset_yaml.exists():
        return row

    raw = yaml.safe_load(dataset_yaml.read_text(encoding="utf-8")) or {}
    names = raw.get("names") or []
    if isinstance(names, dict):
        names = [names[index] for index in sorted(names)]
    row["class_names"] = list(names)
    row["status"] = "checked"
    for split in ("train", "val", "test"):
        split_dir = resolve_yolo_split_dir(dataset_yaml, raw, split)
        split_row = {
            "split": split,
            "path": str(split_dir) if split_dir else "",
            "exists": bool(split_dir and split_dir.exists()),
            "images": 0,
            "missing_labels": 0,
            "empty_labels": 0,
        }
        if split_dir and split_dir.exists() and split_dir.is_dir():
            images = sorted(path for path in split_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
            split_row["images"] = len(images)
            for image_path in images:
                label_path = label_for_image(image_path)
                if not label_path.exists():
                    split_row["missing_labels"] += 1
                elif label_path.stat().st_size == 0:
                    split_row["empty_labels"] += 1
        row["split_checks"].append(split_row)
    return row


def resolve_yolo_split_dir(dataset_yaml: Path, raw: dict[str, Any], split: str) -> Path | None:
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


def label_for_image(image_path: Path) -> Path:
    parts = list(image_path.parts)
    if "images" in parts:
        index = parts.index("images")
        parts[index] = "labels"
        return Path(*parts).with_suffix(".txt")
    return image_path.with_suffix(".txt")


def write_baseline_dry_run_checks(output_dir: Path, *, write_launcher_dry_run: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    launcher_root = output_dir / "launcher_dry_runs"
    launcher_config_dir = output_dir / "launcher_dry_run_configs"
    for config_value in BASELINE_EXPERIMENTS:
        config_path = resolve_experiment_path(config_value)
        raw = load_yaml(config_path)
        config = resolve_env_values(raw)
        issues = validate_experiment(config, config_path)
        dry_run_config = copy.deepcopy(raw)
        dry_run_config.setdefault("outputs", {})["output_root"] = str(launcher_root)
        dry_run_config_path = launcher_config_dir / Path(config_value).name
        dry_run_config_path.parent.mkdir(parents=True, exist_ok=True)
        dry_run_config_path.write_text(yaml.safe_dump(dry_run_config, sort_keys=False), encoding="utf-8")

        dry_run_resolved = resolve_env_values(dry_run_config)
        dry_run_command = build_ultralytics_command(
            dry_run_resolved,
            launcher_root / "runs" / dry_run_resolved["experiment_id"],
            resume=False,
        )
        row = {
            "experiment_id": config["experiment_id"],
            "config": str(config_path),
            "ok": not any(issue.severity == "error" for issue in issues),
            "issues": [issue.__dict__ for issue in issues],
            "command": dry_run_command,
            "safe_dry_run_config": str(dry_run_config_path),
            "safe_dry_run_output_root": str(launcher_root),
        }
        if write_launcher_dry_run:
            row["launcher_result"] = process_one(dry_run_config_path, dry_run=True, resume=False)
        rows.append(row)

    output_path = output_dir / "baseline_training_dry_run_checks.json"
    output_path.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {output_path}")
    return rows


def write_empty_predictions_for_eval(output_dir: Path) -> str:
    records = load_vnwoodknot_records(
        PROJECT_ROOT / "data" / "processed" / "vnwoodknot_manifest.jsonl",
        split="test",
        class_names=["live_knot", "dead_knot"],
    )
    output_path = output_dir / "synthetic_empty_predictions_test.jsonl"
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps({"image_id": record.image_id, "boxes": [], "labels": [], "scores": []}) + "\n")
    print(f"Wrote: {output_path}")
    return str(output_path)


if __name__ == "__main__":
    main()
