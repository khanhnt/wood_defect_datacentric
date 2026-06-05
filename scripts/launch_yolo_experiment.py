#!/usr/bin/env python3
"""Launch reproducible YOLOv8s data-centric experiments.

Default behavior is dry-run only. Use --execute explicitly to start training.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import random
import re
import shlex
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent

if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))


ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-(.*?))?\}")
ALLOWED_DATASETS = {"vsb_curated", "vnwoodknot"}
CONTROLLED_SEEDS = {42, 43, 44}


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    code: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--experiment-config", type=Path, help="Path to one experiment YAML.")
    source.add_argument("--matrix", type=Path, help="Path to experiment_matrix.csv.")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Validate and write run metadata without training.")
    mode.add_argument("--execute", action="store_true", help="Run training. This is never implied.")
    parser.add_argument("--experiment-id", help="Filter one experiment when using --matrix.")
    parser.add_argument("--include-optional", action="store_true", help="Include optional matrix rows.")
    parser.add_argument("--yes-run-matrix", action="store_true", help="Required with --execute --matrix.")
    parser.add_argument("--resume", action="store_true", help="Resume from results/runs/<id>/ultralytics/train/weights/last.pt.")
    parser.add_argument("--strict", action="store_true", help="Return non-zero on validation errors in dry-run mode.")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def expand_env_string(value: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(2)
        if name in os.environ:
            return os.environ[name]
        return "" if default is None else default

    return ENV_PATTERN.sub(replace, value)


def resolve_env_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: resolve_env_values(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve_env_values(item) for item in value]
    if isinstance(value, str):
        return expand_env_string(value)
    return value


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(str(path_value)).expanduser()
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


def load_matrix_configs(matrix_path: Path, *, experiment_id: str | None, include_optional: bool) -> list[Path]:
    rows = list(csv.DictReader(matrix_path.open("r", encoding="utf-8")))
    configs: list[Path] = []
    for row in rows:
        if experiment_id and row["experiment_id"] != experiment_id:
            continue
        optional = row.get("optional", "").strip().lower() == "true"
        if optional and not include_optional and not experiment_id:
            continue
        configs.append(resolve_repo_path(row["config_path"]))
    if experiment_id and not configs:
        raise SystemExit(f"Experiment ID not found in matrix: {experiment_id}")
    return configs


def validate_experiment(config: dict[str, Any], source_config: Path) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    experiment_id = str(config.get("experiment_id", "")).strip()
    dataset = config.get("dataset") or {}
    training = config.get("training") or {}
    transforms = config.get("transforms") or {}

    if not experiment_id:
        issues.append(ValidationIssue("error", "missing_experiment_id", "experiment_id is required."))

    dataset_key = dataset.get("key")
    if dataset_key not in ALLOWED_DATASETS:
        issues.append(ValidationIssue("error", "invalid_dataset", f"dataset.key must be one of {sorted(ALLOWED_DATASETS)}."))

    model = str(training.get("model", "")).strip()
    if model != "yolov8s":
        issues.append(ValidationIssue("error", "invalid_detector", f"Fixed detector must be yolov8s, got {model!r}."))

    if int(training.get("epochs", 0)) != 50:
        issues.append(ValidationIssue("warning", "nonstandard_epochs", "Main controlled budget is 50 epochs."))
    if int(training.get("imgsz", 0)) != 1024:
        issues.append(ValidationIssue("warning", "nonstandard_imgsz", "Expected image size is 1024 unless justified."))
    if int(training.get("seed", -1)) not in CONTROLLED_SEEDS:
        issues.append(
            ValidationIssue(
                "warning",
                "nonstandard_seed",
                f"Expected seed to be one of {sorted(CONTROLLED_SEEDS)} unless justified.",
            )
        )

    data_yaml = dataset.get("data_yaml")
    if not data_yaml:
        issues.append(ValidationIssue("error", "missing_data_yaml", "dataset.data_yaml is required."))
    else:
        data_yaml_path = Path(str(data_yaml)).expanduser()
        if not data_yaml_path.exists():
            issues.append(
                ValidationIssue(
                    "error",
                    "missing_data_yaml",
                    f"dataset YAML is not accessible locally: {data_yaml_path}",
                )
            )
        elif data_yaml_path.suffix not in {".yaml", ".yml"}:
            issues.append(ValidationIssue("error", "invalid_data_yaml_suffix", f"Expected .yaml/.yml: {data_yaml_path}"))

    preprocessing = str(transforms.get("preprocessing", "")).strip()
    augmentation = str(transforms.get("augmentation", "")).strip()
    if not (PROJECT_ROOT / "configs" / "preprocessing" / f"{preprocessing}.yaml").exists():
        issues.append(ValidationIssue("error", "missing_preprocessing_config", f"Unknown preprocessing variant: {preprocessing}"))
    if not (PROJECT_ROOT / "configs" / "augmentation" / f"{augmentation}.yaml").exists():
        issues.append(ValidationIssue("error", "missing_augmentation_config", f"Unknown augmentation variant: {augmentation}"))

    if not source_config.exists():
        issues.append(ValidationIssue("error", "missing_source_config", f"Source config missing: {source_config}"))

    return issues


def run_dir_for(config: dict[str, Any]) -> Path:
    output_root = resolve_repo_path((config.get("outputs") or {}).get("output_root", PROJECT_ROOT / "results"))
    return output_root / "runs" / str(config["experiment_id"])


def build_ultralytics_command(config: dict[str, Any], run_dir: Path, *, resume: bool) -> list[str]:
    training = config["training"]
    dataset = config["dataset"]
    project_dir = run_dir / "ultralytics"
    name = "train"
    if resume:
        last_ckpt = project_dir / name / "weights" / "last.pt"
        return [
            "yolo",
            "detect",
            "train",
            f"model={last_ckpt}",
            "resume=True",
        ]
    command = [
        "yolo",
        "detect",
        "train",
        "model=yolov8s.pt",
        f"data={dataset['data_yaml']}",
        f"epochs={int(training['epochs'])}",
        f"imgsz={int(training['imgsz'])}",
        f"batch={int(training['batch'])}",
        f"device={training['device']}",
        f"workers={int(training['workers'])}",
        f"seed={int(training['seed'])}",
        f"patience={int(training['patience'])}",
        f"project={project_dir}",
        f"name={name}",
        "exist_ok=True",
        f"optimizer={training.get('optimizer', 'auto')}",
        f"pretrained={bool(training.get('pretrained', True))}",
        f"deterministic={bool(training.get('deterministic', True))}",
        f"single_cls={bool(training.get('single_cls', False))}",
    ]
    return command


def write_run_metadata(
    *,
    config: dict[str, Any],
    source_config: Path,
    run_dir: Path,
    command: list[str],
    issues: list[ValidationIssue],
    dry_run: bool,
    resume: bool,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (run_dir / "source_config_path.txt").write_text(str(source_config.resolve()) + "\n", encoding="utf-8")
    (run_dir / "command.txt").write_text(" ".join(shlex.quote(part) for part in command) + "\n", encoding="utf-8")
    (run_dir / "validation_status.json").write_text(
        json.dumps(
            {
                "experiment_id": config["experiment_id"],
                "dry_run": dry_run,
                "resume": resume,
                "ok": not any(issue.severity == "error" for issue in issues),
                "issues": [issue.__dict__ for issue in issues],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    checkpoint_path = run_dir / "ultralytics" / "train" / "weights" / "best.pt"
    (run_dir / "checkpoint_path.txt").write_text(str(checkpoint_path) + "\n", encoding="utf-8")
    if dry_run:
        (run_dir / "training.log").write_text(
            "DRY RUN: no training executed.\n"
            f"Experiment: {config['experiment_id']}\n"
            f"Command: {' '.join(shlex.quote(part) for part in command)}\n",
            encoding="utf-8",
        )
        (run_dir / "validation_metrics.json").write_text(
            json.dumps({"status": "not_available_dry_run", "metrics": {}}, indent=2) + "\n",
            encoding="utf-8",
        )


def execute_training(config: dict[str, Any], run_dir: Path, *, resume: bool) -> dict[str, Any]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required for execution. Dry-run does not require it.") from exc

    training = config["training"]
    seed = int(training["seed"])
    set_reproducible_seed(seed)
    project_dir = run_dir / "ultralytics"
    train_name = "train"
    log_path = run_dir / "training.log"
    train_kwargs: dict[str, Any] = {
        "data": str(Path(str(config["dataset"]["data_yaml"])).expanduser()),
        "epochs": int(training["epochs"]),
        "imgsz": int(training["imgsz"]),
        "batch": int(training["batch"]),
        "device": str(training["device"]),
        "workers": int(training["workers"]),
        "seed": seed,
        "patience": int(training["patience"]),
        "project": str(project_dir),
        "name": train_name,
        "exist_ok": True,
        "optimizer": training.get("optimizer", "auto"),
        "pretrained": bool(training.get("pretrained", True)),
        "deterministic": bool(training.get("deterministic", True)),
        "single_cls": bool(training.get("single_cls", False)),
        "verbose": True,
    }
    if resume:
        last_ckpt = project_dir / train_name / "weights" / "last.pt"
        if not last_ckpt.exists():
            raise FileNotFoundError(f"Cannot resume; missing checkpoint: {last_ckpt}")
        model = YOLO(str(last_ckpt))
        train_kwargs = {"resume": True}
    else:
        model = YOLO("yolov8s.pt")

    with log_path.open("a", encoding="utf-8") as log_handle:
        with contextlib.redirect_stdout(log_handle), contextlib.redirect_stderr(log_handle):
            result = model.train(**train_kwargs)

    metrics = collect_metrics(project_dir / train_name)
    (run_dir / "validation_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    summary = {
        "experiment_id": config["experiment_id"],
        "status": "complete",
        "result": str(result),
        "best_checkpoint_path": str(project_dir / train_name / "weights" / "best.pt"),
        "last_checkpoint_path": str(project_dir / train_name / "weights" / "last.pt"),
        "metrics_path": str(run_dir / "validation_metrics.json"),
    }
    (run_dir / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def set_reproducible_seed(seed: int) -> None:
    """Set Python, NumPy, and PyTorch seeds before creating the YOLO model."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    except Exception:
        pass


def collect_metrics(train_dir: Path) -> dict[str, Any]:
    results_csv = train_dir / "results.csv"
    if not results_csv.exists():
        return {"status": "results_csv_missing", "results_csv": str(results_csv), "metrics": {}}
    rows = list(csv.DictReader(results_csv.open("r", encoding="utf-8")))
    if not rows:
        return {"status": "results_csv_empty", "results_csv": str(results_csv), "metrics": {}}
    last_row = {key.strip(): value for key, value in rows[-1].items()}
    best_row = max(rows, key=lambda row: _float_or_minus_inf(_metric_value(row, "metrics/mAP50(B)")))
    return {
        "status": "ok",
        "results_csv": str(results_csv),
        "last_epoch": last_row,
        "best_by_map50": {key.strip(): value for key, value in best_row.items()},
    }


def _float_or_minus_inf(value: str | None) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return float("-inf")


def _metric_value(row: dict[str, str], metric_name: str) -> str | None:
    for key, value in row.items():
        if key.strip() == metric_name:
            return value
    return None


def process_one(source_config: Path, *, dry_run: bool, resume: bool) -> dict[str, Any]:
    raw = load_yaml(source_config)
    config = resolve_env_values(raw)
    run_dir = run_dir_for(config)
    issues = validate_experiment(config, source_config)
    command = build_ultralytics_command(config, run_dir, resume=resume)
    write_run_metadata(
        config=config,
        source_config=source_config,
        run_dir=run_dir,
        command=command,
        issues=issues,
        dry_run=dry_run,
        resume=resume,
    )

    has_errors = any(issue.severity == "error" for issue in issues)
    status = "OK" if not has_errors else "BLOCKED"
    print(f"[{status}] {config['experiment_id']} -> {run_dir}")
    for issue in issues:
        print(f"  {issue.severity}: {issue.code}: {issue.message}")

    if dry_run:
        return {
            "experiment_id": config["experiment_id"],
            "source_config": str(source_config),
            "run_dir": str(run_dir),
            "ok": not has_errors,
            "error_count": sum(issue.severity == "error" for issue in issues),
            "warning_count": sum(issue.severity == "warning" for issue in issues),
        }
    if has_errors:
        raise SystemExit(f"Refusing to execute {config['experiment_id']} because validation errors exist.")
    execute_training(config, run_dir, resume=resume)
    return {
        "experiment_id": config["experiment_id"],
        "source_config": str(source_config),
        "run_dir": str(run_dir),
        "ok": True,
        "error_count": 0,
        "warning_count": sum(issue.severity == "warning" for issue in issues),
    }


def write_dry_run_summary(rows: list[dict[str, Any]]) -> Path:
    output_path = PROJECT_ROOT / "results" / "runs" / "dry_run_summary.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["experiment_id", "ok", "error_count", "warning_count", "source_config", "run_dir"]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def main() -> None:
    args = parse_args()
    dry_run = not args.execute
    if args.execute and args.matrix and not args.yes_run_matrix:
        raise SystemExit("Refusing to execute a matrix without --yes-run-matrix.")

    if args.matrix:
        config_paths = load_matrix_configs(
            resolve_repo_path(args.matrix),
            experiment_id=args.experiment_id,
            include_optional=args.include_optional,
        )
    else:
        config_paths = [resolve_repo_path(args.experiment_config)]

    results = [process_one(path, dry_run=dry_run, resume=args.resume) for path in config_paths]
    if dry_run:
        summary_path = write_dry_run_summary(results)
        print(f"Dry-run summary: {summary_path}")
    if dry_run and args.strict and not all(row["ok"] for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
