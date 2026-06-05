#!/usr/bin/env python3
"""Run the full multiseed YOLOv8s experiment queue across isolated GPUs.

This script is intended for the Vast.ai server. It does not use
DataParallel/DDP; each worker process is isolated with CUDA_VISIBLE_DEVICES and
sees a single GPU as device 0.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import threading
import time
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VSB_YAML = "/workspace/data/main_dataset/benchmarks/vsb7_3600_rare_first_yolo/dataset.yaml"
DEFAULT_VN_YAML = "/workspace/data/vnwoodknot/benchmarks/vnwoodknot_live_dead_2class_yolo/dataset.yaml"
DEFAULT_GENERATED_ROOT = "/workspace/data/wood_defect_datacentric/generated_yolo/multiseed"
SEEDS = (42, 43, 44)


@dataclass(frozen=True)
class VariantSpec:
    dataset: str
    variant: str
    label: str
    base_config: Path
    kind: str
    preprocessing: str | None = None
    augmentation: str | None = None


@dataclass(frozen=True)
class Job:
    spec: VariantSpec
    seed: int
    index: int
    total: int

    @property
    def job_id(self) -> str:
        return f"{self.spec.dataset}_{self.spec.variant}_seed{self.seed}"

    @property
    def run_id(self) -> str:
        return f"{self.spec.variant}_seed{self.seed}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--gpus", default="0,1", help="Comma-separated physical GPU IDs, e.g. 0,1.")
    parser.add_argument("--resume", action="store_true", help="Skip jobs that already have best.pt and metrics.")
    parser.add_argument("--dataset", choices=("vnwoodknot", "vsb", "all"), default="all")
    parser.add_argument("--dry-run", action="store_true", help="Print the job list without materializing or training.")
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "4")))
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--retry-batch-step", type=int, default=8)
    parser.add_argument("--vsb-yaml", default=os.environ.get("WOOD_DC_VSB_BASELINE_DATASET_YAML", DEFAULT_VSB_YAML))
    parser.add_argument("--vn-yaml", default=os.environ.get("WOOD_DC_VN_BASELINE_DATASET_YAML", DEFAULT_VN_YAML))
    parser.add_argument("--generated-root", default=os.environ.get("GENERATED_DATA_ROOT", DEFAULT_GENERATED_ROOT))
    parser.add_argument("--jpg-quality", type=int, default=95)
    parser.add_argument("--overwrite-materialized", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpu_ids = parse_gpus(args.gpus)
    jobs = build_jobs(dataset_filter=args.dataset)

    if args.dry_run:
        print_dry_run(args, jobs, gpu_ids)
        return

    prepare_runtime_dirs()
    run_log = PROJECT_ROOT / "results" / "gpu_optimization" / "run_log.csv"
    ensure_run_log(run_log)

    print(f"Starting {len(jobs)} jobs on GPUs: {', '.join(gpu_ids)}")
    print(f"Batch size: {args.batch_size}; retry batch step: {args.retry_batch_step}")
    print(f"Run log: {run_log}")

    work_queue: queue.Queue[Job] = queue.Queue()
    for job in jobs:
        work_queue.put(job)

    state = RunnerState(total=len(jobs), run_log=run_log)
    threads = []
    for gpu_id in gpu_ids:
        thread = threading.Thread(target=worker_loop, args=(gpu_id, args, work_queue, state), daemon=False)
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    print("All queued jobs have finished or been skipped.")


def parse_gpus(value: str) -> list[str]:
    gpu_ids = [item.strip() for item in value.split(",") if item.strip()]
    if not gpu_ids:
        raise SystemExit("--gpus must contain at least one GPU ID.")
    return gpu_ids


def build_jobs(*, dataset_filter: str) -> list[Job]:
    specs = []
    if dataset_filter in {"vnwoodknot", "all"}:
        specs.extend(vn_specs())
    if dataset_filter in {"vsb", "all"}:
        specs.extend(vsb_specs())

    jobs_without_index = [(spec, seed) for spec in specs for seed in SEEDS]
    total = len(jobs_without_index)
    return [Job(spec=spec, seed=seed, index=index + 1, total=total) for index, (spec, seed) in enumerate(jobs_without_index)]


def vn_specs() -> list[VariantSpec]:
    return [
        VariantSpec("vnwoodknot", "baseline", "Baseline", Path("configs/experiments/vn_t0_yolov8s_baseline_e50.yaml"), "baseline"),
        VariantSpec(
            "vnwoodknot",
            "p2_illumination",
            "P2 illumination",
            Path("configs/experiments/vn_yolov8s_p2_illumination_e50.yaml"),
            "preprocess",
            preprocessing="P2_illumination_normalization",
        ),
        VariantSpec(
            "vnwoodknot",
            "a1_crop",
            "A1 crop",
            Path("configs/experiments/vn_yolov8s_a1_crop_e50.yaml"),
            "augment",
            augmentation="A1_defect_preserving_crop",
        ),
        VariantSpec(
            "vnwoodknot",
            "a2_colorjitter",
            "A2 color jitter",
            Path("configs/experiments/vn_yolov8s_a2_colorjitter_e50.yaml"),
            "augment",
            augmentation="A2_texture_aware_color_jitter",
        ),
        VariantSpec(
            "vnwoodknot",
            "p4_a4_combined",
            "P4+A4 combined",
            Path("configs/experiments/vn_yolov8s_p4_a4_combined_e50.yaml"),
            "combined",
            preprocessing="P4_combined_safe",
            augmentation="A4_combined_best",
        ),
    ]


def vsb_specs() -> list[VariantSpec]:
    return [
        VariantSpec("vsb_rarefirst", "baseline", "Baseline", Path("configs/experiments/vsb_yolov8s_baseline_e50.yaml"), "baseline"),
        VariantSpec(
            "vsb_rarefirst",
            "p1_clahe",
            "P1 CLAHE",
            Path("configs/experiments/vsb_yolov8s_p1_clahe_e50.yaml"),
            "preprocess",
            preprocessing="P1_CLAHE_luminance",
        ),
        VariantSpec(
            "vsb_rarefirst",
            "p2_illumination",
            "P2 illumination",
            Path("configs/experiments/vsb_yolov8s_p2_illumination_e50.yaml"),
            "preprocess",
            preprocessing="P2_illumination_normalization",
        ),
        VariantSpec(
            "vsb_rarefirst",
            "p3_unsharp",
            "P3 unsharp",
            Path("configs/experiments/vsb_yolov8s_p3_unsharp_e50.yaml"),
            "preprocess",
            preprocessing="P3_mild_unsharp",
        ),
        VariantSpec(
            "vsb_rarefirst",
            "a1_crop",
            "A1 crop",
            Path("configs/experiments/vsb_yolov8s_a1_crop_e50.yaml"),
            "augment",
            augmentation="A1_defect_preserving_crop",
        ),
        VariantSpec(
            "vsb_rarefirst",
            "a2_colorjitter",
            "A2 color jitter",
            Path("configs/experiments/vsb_yolov8s_a2_colorjitter_e50.yaml"),
            "augment",
            augmentation="A2_texture_aware_color_jitter",
        ),
        VariantSpec(
            "vsb_rarefirst",
            "p4_a4_combined",
            "P4+A4 combined",
            Path("configs/experiments/vsb_yolov8s_p4_a4_combined_e50.yaml"),
            "combined",
            preprocessing="P4_combined_safe",
            augmentation="A4_combined_best",
        ),
    ]


def print_dry_run(args: argparse.Namespace, jobs: list[Job], gpu_ids: list[str]) -> None:
    print(f"Dry run: {len(jobs)} jobs")
    print(f"GPUs: {', '.join(gpu_ids)}")
    print(f"Batch size: {args.batch_size}")
    for job in jobs:
        run_dir = run_dir_for(job)
        planned_yaml = planned_dataset_yaml(job, args)
        print(
            f"{job.index:02d}/{job.total} | {job.spec.dataset} | {job.spec.variant} | "
            f"seed={job.seed} | run_dir={run_dir} | data={planned_yaml}"
        )


def prepare_runtime_dirs() -> None:
    for path in [
        PROJECT_ROOT / "results" / "gpu_optimization" / "job_logs",
        PROJECT_ROOT / "results" / "gpu_optimization" / "generated_configs",
        PROJECT_ROOT / "results" / "gpu_optimization" / "materialization_logs",
        PROJECT_ROOT / "results" / "multiseed" / "vnwoodknot" / "per_seed",
        PROJECT_ROOT / "results" / "multiseed" / "vsb_rarefirst" / "per_seed",
    ]:
        path.mkdir(parents=True, exist_ok=True)


class RunnerState:
    def __init__(self, *, total: int, run_log: Path) -> None:
        self.total = total
        self.run_log = run_log
        self.completed = 0
        self.progress_lock = threading.Lock()
        self.csv_lock = threading.Lock()
        self.materialization_guard = threading.Lock()
        self.materialization_locks: dict[Path, threading.Lock] = {}

    def lock_for_materialized_path(self, path: Path) -> threading.Lock:
        with self.materialization_guard:
            if path not in self.materialization_locks:
                self.materialization_locks[path] = threading.Lock()
            return self.materialization_locks[path]

    def mark_complete(self) -> int:
        with self.progress_lock:
            self.completed += 1
            return self.completed


def worker_loop(gpu_id: str, args: argparse.Namespace, work_queue: queue.Queue[Job], state: RunnerState) -> None:
    while True:
        try:
            job = work_queue.get_nowait()
        except queue.Empty:
            return
        try:
            run_job(gpu_id, args, job, state)
        finally:
            work_queue.task_done()


def run_job(gpu_id: str, args: argparse.Namespace, job: Job, state: RunnerState) -> None:
    run_dir = run_dir_for(job)
    first_started = utc_now()

    if args.resume and job_completed(run_dir):
        finished = utc_now()
        row = base_log_row(job, gpu_id, args.batch_size, first_started, finished)
        row.update({"status": "skipped_completed", "mAP50": extract_map50(run_dir), "attempt": 0, "error": ""})
        append_run_log(state, row)
        count = state.mark_complete()
        print_progress(count, job, gpu_id, "skipped", first_started, finished, row["mAP50"])
        return

    if run_dir.exists() and not args.resume:
        finished = utc_now()
        row = base_log_row(job, gpu_id, args.batch_size, first_started, finished)
        row.update({"status": "failed_existing_output", "mAP50": "", "attempt": 0, "error": f"Run directory exists: {run_dir}"})
        append_run_log(state, row)
        count = state.mark_complete()
        print_progress(count, job, gpu_id, "failed_existing_output", first_started, finished, "")
        return

    current_batch = int(args.batch_size)
    final_status = "failed"
    final_map50 = ""
    final_finished = first_started
    final_error = ""

    for attempt in (1, 2):
        started = utc_now()
        try:
            data_yaml = ensure_dataset_for_job(job, args, state)
            config_path = write_job_config(job, data_yaml=data_yaml, batch_size=current_batch, args=args, attempt=attempt)
            log_path = job_log_path(job, attempt=attempt)
            status, error = launch_training(job, config_path=config_path, gpu_id=gpu_id, log_path=log_path)
            finished = utc_now()
            map50 = extract_map50(run_dir)
            row = base_log_row(job, gpu_id, current_batch, started, finished)
            row.update({"status": status, "mAP50": map50, "attempt": attempt, "error": error})
            append_run_log(state, row)

            final_status = status
            final_map50 = map50
            final_finished = finished
            final_error = error
            if status == "ok":
                break
            if attempt == 1 and current_batch > args.retry_batch_step:
                current_batch -= int(args.retry_batch_step)
                print(f"{job.job_id} failed with {status}; retrying once at batch={current_batch}.", flush=True)
                continue
            break
        except Exception as exc:
            finished = utc_now()
            row = base_log_row(job, gpu_id, current_batch, started, finished)
            row.update({"status": "failed", "mAP50": "", "attempt": attempt, "error": str(exc)})
            append_run_log(state, row)
            final_status = "failed"
            final_map50 = ""
            final_finished = finished
            final_error = str(exc)
            if attempt == 1 and current_batch > args.retry_batch_step:
                current_batch -= int(args.retry_batch_step)
                print(f"{job.job_id} failed before training; retrying once at batch={current_batch}: {exc}", flush=True)
                continue
            break

    count = state.mark_complete()
    status_text = "done" if final_status == "ok" else final_status
    if final_error and final_status != "ok":
        status_text = f"{status_text}: {final_error[:80]}"
    print_progress(count, job, gpu_id, status_text, first_started, final_finished, final_map50)


def ensure_dataset_for_job(job: Job, args: argparse.Namespace, state: RunnerState) -> Path:
    planned = planned_dataset_yaml(job, args)
    if job.spec.kind == "baseline":
        if not planned.exists():
            raise FileNotFoundError(f"Missing baseline dataset YAML: {planned}")
        return planned

    lock = state.lock_for_materialized_path(planned.parent)
    with lock:
        if planned.exists():
            return planned
        if planned.parent.exists() and any(planned.parent.iterdir()) and not args.overwrite_materialized:
            raise FileExistsError(f"Materialized dataset directory exists but dataset.yaml is missing: {planned.parent}")
        if job.spec.kind == "preprocess":
            materialize_preprocess(job, args, output_root=planned.parent)
        elif job.spec.kind == "augment":
            materialize_augment(job, args, source_yaml=base_dataset_yaml(job.spec.dataset, args), output_root=planned.parent)
        elif job.spec.kind == "combined":
            pre_yaml = shared_preprocess_yaml(job, args)
            ensure_shared_preprocess(job, args, state, pre_yaml=pre_yaml)
            materialize_augment(job, args, source_yaml=pre_yaml, output_root=planned.parent)
        else:
            raise ValueError(f"Unknown materialization kind: {job.spec.kind}")
    if not planned.exists():
        raise FileNotFoundError(f"Materialization did not produce dataset YAML: {planned}")
    return planned


def ensure_shared_preprocess(job: Job, args: argparse.Namespace, state: RunnerState, *, pre_yaml: Path) -> None:
    lock = state.lock_for_materialized_path(pre_yaml.parent)
    with lock:
        if pre_yaml.exists():
            return
        if pre_yaml.parent.exists() and any(pre_yaml.parent.iterdir()) and not args.overwrite_materialized:
            raise FileExistsError(f"Shared preprocessing directory exists but dataset.yaml is missing: {pre_yaml.parent}")
        materialize_preprocess(job, args, output_root=pre_yaml.parent)
        if not pre_yaml.exists():
            raise FileNotFoundError(f"Shared preprocessing did not produce dataset YAML: {pre_yaml}")


def materialize_preprocess(job: Job, args: argparse.Namespace, *, output_root: Path) -> None:
    if not job.spec.preprocessing:
        raise ValueError(f"Job has no preprocessing variant: {job.job_id}")
    command = [
        sys.executable,
        "scripts/materialize_preprocessed_yolo.py",
        "--source-yaml",
        str(base_dataset_yaml(job.spec.dataset, args)),
        "--variant-config",
        str(PROJECT_ROOT / "configs" / "preprocessing" / f"{job.spec.preprocessing}.yaml"),
        "--output-root",
        str(output_root),
        "--image-format",
        "jpg",
        "--jpg-quality",
        str(args.jpg_quality),
    ]
    if args.overwrite_materialized:
        command.append("--overwrite")
    run_materialization_command(job, command, output_root)


def materialize_augment(job: Job, args: argparse.Namespace, *, source_yaml: Path, output_root: Path) -> None:
    if not job.spec.augmentation:
        raise ValueError(f"Job has no augmentation variant: {job.job_id}")
    command = [
        sys.executable,
        "scripts/materialize_augmented_yolo.py",
        "--source-yaml",
        str(source_yaml),
        "--variant-config",
        str(PROJECT_ROOT / "configs" / "augmentation" / f"{job.spec.augmentation}.yaml"),
        "--output-root",
        str(output_root),
        "--seed",
        str(job.seed),
        "--image-format",
        "jpg",
        "--jpg-quality",
        str(args.jpg_quality),
    ]
    if args.overwrite_materialized:
        command.append("--overwrite")
    run_materialization_command(job, command, output_root)


def run_materialization_command(job: Job, command: list[str], output_root: Path) -> None:
    log_dir = PROJECT_ROOT / "results" / "gpu_optimization" / "materialization_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{job.job_id}_{output_root.name}.log"
    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"Materialization failed for {job.job_id}. See {log_path}")


def launch_training(job: Job, *, config_path: Path, gpu_id: str, log_path: Path) -> tuple[str, str]:
    command = [
        sys.executable,
        "scripts/launch_yolo_experiment.py",
        "--experiment-config",
        str(config_path),
        "--strict",
        "--execute",
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["DEVICE"] = "0"
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )

    if completed.returncode == 0 and job_completed(run_dir_for(job)):
        return "ok", ""

    text = log_path.read_text(encoding="utf-8", errors="replace")
    if is_oom_text(text):
        return "oom", f"CUDA OOM. See {log_path}"
    return "failed", f"Exit code {completed.returncode}. See {log_path}"


def write_job_config(job: Job, *, data_yaml: Path, batch_size: int, args: argparse.Namespace, attempt: int) -> Path:
    source = PROJECT_ROOT / job.spec.base_config
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["experiment_id"] = job.run_id
    config["dataset"]["data_yaml"] = str(data_yaml)
    config["training"]["batch"] = int(batch_size)
    config["training"]["seed"] = int(job.seed)
    config["training"]["device"] = "0"
    config["training"]["imgsz"] = int(args.imgsz)
    config["training"]["epochs"] = int(args.epochs)
    config["training"]["workers"] = int(args.workers)
    config["training"]["deterministic"] = True
    config.setdefault("outputs", {})["output_root"] = str(output_root_for(job.spec.dataset))
    config["description"] = f"{config.get('description', '').strip()} Multiseed {job.spec.dataset} {job.spec.variant} seed {job.seed}."

    config_dir = PROJECT_ROOT / "results" / "gpu_optimization" / "generated_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"{job.job_id}_attempt{attempt}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def planned_dataset_yaml(job: Job, args: argparse.Namespace) -> Path:
    if job.spec.kind == "baseline":
        return base_dataset_yaml(job.spec.dataset, args)
    root = Path(args.generated_root).expanduser()
    if job.spec.kind == "preprocess":
        return root / job.spec.dataset / "shared" / str(job.spec.preprocessing) / "dataset.yaml"
    if job.spec.kind == "augment":
        return root / job.spec.dataset / f"seed{job.seed}" / str(job.spec.augmentation) / "dataset.yaml"
    if job.spec.kind == "combined":
        return root / job.spec.dataset / f"seed{job.seed}" / f"{job.spec.preprocessing}__{job.spec.augmentation}" / "dataset.yaml"
    raise ValueError(f"Unknown materialization kind: {job.spec.kind}")


def shared_preprocess_yaml(job: Job, args: argparse.Namespace) -> Path:
    if not job.spec.preprocessing:
        raise ValueError(f"Job has no preprocessing variant: {job.job_id}")
    return Path(args.generated_root).expanduser() / job.spec.dataset / "shared" / job.spec.preprocessing / "dataset.yaml"


def base_dataset_yaml(dataset: str, args: argparse.Namespace) -> Path:
    if dataset == "vnwoodknot":
        return Path(args.vn_yaml).expanduser()
    if dataset == "vsb_rarefirst":
        return Path(args.vsb_yaml).expanduser()
    raise ValueError(f"Unknown dataset: {dataset}")


def output_root_for(dataset: str) -> Path:
    return PROJECT_ROOT / "results" / "multiseed" / dataset / "per_seed"


def run_dir_for(job: Job) -> Path:
    # launch_yolo_experiment.py always appends "runs/<experiment_id>" to
    # outputs.output_root, so mirror that layout when checking completion.
    return output_root_for(job.spec.dataset) / "runs" / job.run_id


def job_log_path(job: Job, *, attempt: int) -> Path:
    return PROJECT_ROOT / "results" / "gpu_optimization" / "job_logs" / f"{job.job_id}_attempt{attempt}.log"


def job_completed(run_dir: Path) -> bool:
    best = run_dir / "ultralytics" / "train" / "weights" / "best.pt"
    metrics = run_dir / "validation_metrics.json"
    if not best.exists() or not metrics.exists():
        return False
    try:
        data = json.loads(metrics.read_text(encoding="utf-8"))
    except Exception:
        return False
    return data.get("status") == "ok"


def extract_map50(run_dir: Path) -> str:
    metrics = run_dir / "validation_metrics.json"
    if not metrics.exists():
        return ""
    try:
        data = json.loads(metrics.read_text(encoding="utf-8"))
    except Exception:
        return ""
    value = (data.get("best_by_map50") or {}).get("metrics/mAP50(B)", "")
    return str(value)


def ensure_run_log(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_log_fieldnames())
        writer.writeheader()


def append_run_log(state: RunnerState, row: dict[str, Any]) -> None:
    with state.csv_lock:
        with state.run_log.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=run_log_fieldnames())
            writer.writerow(row)


def run_log_fieldnames() -> list[str]:
    return [
        "job_id",
        "variant",
        "seed",
        "dataset",
        "gpu_id",
        "batch_size",
        "start_time",
        "end_time",
        "duration_min",
        "status",
        "mAP50",
        "attempt",
        "error",
    ]


def base_log_row(job: Job, gpu_id: str, batch_size: int, start: datetime, end: datetime) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "variant": job.spec.variant,
        "seed": job.seed,
        "dataset": job.spec.dataset,
        "gpu_id": gpu_id,
        "batch_size": batch_size,
        "start_time": start.isoformat(),
        "end_time": end.isoformat(),
        "duration_min": round((end - start).total_seconds() / 60.0, 4),
    }


def print_progress(count: int, job: Job, gpu_id: str, status: str, start: datetime, end: datetime, map50: str) -> None:
    duration = format_duration((end - start).total_seconds())
    map_text = f"mAP50={map50}" if map50 else "mAP50=n/a"
    print(
        f"[{count}/{job.total}] GPU-{gpu_id} | {dataset_display(job.spec.dataset)} | "
        f"{job.spec.label} | seed={job.seed} | {status} in {duration} | {map_text}",
        flush=True,
    )


def format_duration(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    return f"{minutes}m{sec:02d}s"


def dataset_display(dataset: str) -> str:
    return "VNWoodKnot" if dataset == "vnwoodknot" else "VSB rare-first"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def is_oom_text(text: str) -> bool:
    lowered = text.lower()
    return "outofmemory" in lowered or "out of memory" in lowered or "cuda out" in lowered


if __name__ == "__main__":
    main()
