#!/usr/bin/env python3
"""Run the nine locked VNWoodKnot Faster R-CNN jobs with one process per GPU."""

from __future__ import annotations

import argparse
import csv
import json
import os
import queue
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from second_detector.protocol import JOB_VARIANTS, SEEDS, build_jobs, locked_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuilt-root", type=Path, default=Path("/workspace/data/datasets_rebuilt"))
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--variants", nargs="+", choices=JOB_VARIANTS, default=list(JOB_VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--retry-batch-step",
        type=int,
        default=0,
        help="Disabled by default so every manuscript run uses one protocol-wide batch size.",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def completed(output_dir: Path) -> bool:
    required = [output_dir / "weights" / "best.pt", output_dir / "weights" / "last.pt", output_dir / "training_summary.json"]
    if not all(path.is_file() and path.stat().st_size > 0 for path in required):
        return False
    try:
        return json.loads(required[-1].read_text(encoding="utf-8")).get("status") == "ok"
    except Exception:
        return False


def print_dry_run(args: argparse.Namespace, jobs) -> None:
    config = locked_config(epochs=args.epochs, batch_size=args.batch_size, workers=args.workers)
    print("dataset\tvariant\tseed\tdata_yaml\tbatch\tepochs\timgsz\tpretrained\tbest_metric\toutput_path")
    for job in jobs:
        print(
            f"vnwoodknot\t{job.variant}\t{job.seed}\t{job.data_yaml}\t{config['batch_size']}\t"
            f"{config['epochs']}\t{config['image_size']}\tCOCO_DEFAULT\t{config['best_metric']}\t{job.output_dir}"
        )


def append_row(path: Path, lock: threading.Lock, row: dict) -> None:
    with lock:
        exists = path.exists()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    jobs = build_jobs(
        rebuilt_root=args.rebuilt_root,
        results_root=args.results_root,
        variants=args.variants,
        seeds=args.seeds,
    )
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpu_ids:
        raise SystemExit("At least one GPU ID is required.")
    if args.dry_run:
        print_dry_run(args, jobs)
        return
    missing = [str(job.data_yaml) for job in jobs if not job.data_yaml.exists()]
    if missing:
        raise SystemExit("Missing dataset YAML(s):\n" + "\n".join(missing))

    run_log = args.results_root.resolve() / "fasterrcnn" / "run_log.csv"
    log_dir = args.results_root.resolve() / "fasterrcnn" / "job_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    work: queue.Queue = queue.Queue()
    for job in jobs:
        work.put(job)
    csv_lock = threading.Lock()
    progress_lock = threading.Lock()
    progress = {"done": 0}
    failures: list[str] = []

    def worker(gpu_id: str) -> None:
        while True:
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            started = datetime.now(timezone.utc)
            status = "failed"
            error = ""
            used_batch = int(args.batch_size)
            attempt = 0
            if args.resume and completed(job.output_dir):
                status = "skipped_completed"
            else:
                attempts = (1, 2) if args.retry_batch_step > 0 else (1,)
                for attempt in attempts:
                    log_path = log_dir / f"{job.job_id}_attempt{attempt}.log"
                    command = [
                        sys.executable,
                        str(PROJECT_ROOT / "scripts" / "train_fasterrcnn.py"),
                        "--data-yaml", str(job.data_yaml),
                        "--output-dir", str(job.output_dir),
                        "--variant", job.variant,
                        "--seed", str(job.seed),
                        "--device", "cuda:0",
                        "--epochs", str(args.epochs),
                        "--batch-size", str(used_batch),
                        "--workers", str(args.workers),
                    ]
                    env = os.environ.copy()
                    env["CUDA_VISIBLE_DEVICES"] = gpu_id
                    with log_path.open("w", encoding="utf-8") as log_handle:
                        result = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
                    if result.returncode == 0 and completed(job.output_dir):
                        status = "ok"
                        break
                    error = f"exit_code={result.returncode}; log={log_path}"
                    if job.output_dir.exists():
                        archived = job.output_dir.with_name(f"{job.output_dir.name}_failed_attempt{attempt}")
                        if archived.exists():
                            archived = archived.with_name(f"{archived.name}_{int(time.time())}")
                        job.output_dir.rename(archived)
                    if len(attempts) == 2 and attempt == 1 and used_batch > args.retry_batch_step:
                        used_batch -= args.retry_batch_step
                    else:
                        break
            ended = datetime.now(timezone.utc)
            map50 = ""
            if completed(job.output_dir):
                history = list(csv.DictReader((job.output_dir / "results.csv").open()))
                if history:
                    best = max(history, key=lambda row: float(row["metrics/mAP50-95(B)"]))
                    map50 = best["metrics/mAP50(B)"]
            row = {
                "job_id": job.job_id,
                "variant": job.variant,
                "seed": job.seed,
                "gpu_id": gpu_id,
                "batch_size": used_batch,
                "start_time": started.isoformat(),
                "end_time": ended.isoformat(),
                "duration_min": round((ended - started).total_seconds() / 60.0, 3),
                "status": status,
                "mAP50": map50,
                "attempt": attempt,
                "error": error,
            }
            append_row(run_log, csv_lock, row)
            if status not in {"ok", "skipped_completed"}:
                failures.append(job.job_id)
            with progress_lock:
                progress["done"] += 1
                count = progress["done"]
            print(
                f"[{count}/{len(jobs)}] GPU-{gpu_id} | {job.variant} | seed={job.seed} | "
                f"{status} in {row['duration_min']:.1f}m | mAP50={map50 or 'n/a'}",
                flush=True,
            )
            work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu_id,), daemon=False) for gpu_id in gpu_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    print(f"Finished {len(jobs)} Faster R-CNN jobs. Run log: {run_log}")
    if failures:
        raise SystemExit("Failed jobs: " + ", ".join(failures))


if __name__ == "__main__":
    main()
