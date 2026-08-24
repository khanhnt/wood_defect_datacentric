#!/usr/bin/env python3
"""Export low-confidence Faster R-CNN predictions for the locked nine-run matrix."""

from __future__ import annotations

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from second_detector.protocol import JOB_VARIANTS, SEEDS, build_jobs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuilt-root", type=Path, default=Path("/workspace/data/datasets_rebuilt"))
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("val", "test"), required=True)
    parser.add_argument("--gpus", default="0,1")
    parser.add_argument("--variants", nargs="+", choices=JOB_VARIANTS, default=list(JOB_VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, choices=SEEDS, default=list(SEEDS))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--single-job", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--checkpoint", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--data-yaml", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--variant", choices=JOB_VARIANTS, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="cuda:0", help=argparse.SUPPRESS)
    return parser.parse_args()


def run_single(args: argparse.Namespace) -> None:
    from second_detector.engine import export_checkpoint

    output = args.output_dir / f"{args.variant}_seed{args.seed}_predictions.json"
    result = export_checkpoint(
        checkpoint=args.checkpoint,
        data_yaml=args.data_yaml,
        split=args.split,
        output_json=output,
        variant=args.variant,
        seed=args.seed,
        device_name=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
    )
    print(json.dumps(result, indent=2))


def main() -> None:
    args = parse_args()
    if args.single_job:
        run_single(args)
        return
    jobs = build_jobs(
        rebuilt_root=args.rebuilt_root,
        results_root=args.results_root,
        variants=args.variants,
        seeds=args.seeds,
    )
    gpu_ids = [item.strip() for item in args.gpus.split(",") if item.strip()]
    work: queue.Queue = queue.Queue()
    for job in jobs:
        work.put(job)
    errors: list[str] = []
    lock = threading.Lock()
    progress = {"done": 0}

    def worker(gpu_id: str) -> None:
        while True:
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            output = args.output_dir / f"{job.variant}_seed{job.seed}_predictions.json"
            checkpoint = job.output_dir / "weights" / "best.pt"
            status = "skipped" if output.exists() and not args.overwrite else "ok"
            if status == "ok":
                command = [
                    sys.executable, str(Path(__file__).resolve()),
                    "--single-job",
                    "--results-root", str(args.results_root),
                    "--output-dir", str(args.output_dir),
                    "--split", args.split,
                    "--checkpoint", str(checkpoint),
                    "--data-yaml", str(job.data_yaml),
                    "--variant", job.variant,
                    "--seed", str(job.seed),
                    "--device", "cuda:0",
                    "--batch-size", str(args.batch_size),
                    "--workers", str(args.workers),
                ]
                env = os.environ.copy()
                env["CUDA_VISIBLE_DEVICES"] = gpu_id
                result = subprocess.run(command, cwd=PROJECT_ROOT, env=env)
                if result.returncode != 0:
                    status = "failed"
                    errors.append(job.job_id)
            with lock:
                progress["done"] += 1
                print(f"[{progress['done']}/{len(jobs)}] GPU-{gpu_id} | {job.variant} | seed={job.seed} | {status}", flush=True)
            work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False) for gpu in gpu_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if errors:
        raise SystemExit("Failed exports: " + ", ".join(errors))


if __name__ == "__main__":
    main()
