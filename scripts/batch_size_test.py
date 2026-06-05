#!/usr/bin/env python3
"""Probe YOLOv8s batch sizes on VNWoodKnot.

This script is intended to run on the Vast.ai training server, not on the local
development machine. It runs short VNWoodKnot baseline trainings for candidate
batch sizes and records peak CUDA memory usage.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_YAML = "/workspace/data/vnwoodknot/benchmarks/vnwoodknot_live_dead_2class_yolo/dataset.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-yaml", default=os.environ.get("WOOD_DC_VN_BASELINE_DATASET_YAML", DEFAULT_DATA_YAML))
    parser.add_argument("--weights", default=os.environ.get("YOLO_WEIGHTS", "yolov8s.pt"))
    parser.add_argument("--batches", nargs="+", type=int, default=[24, 32, 40])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--device", default=os.environ.get("DEVICE", "0"))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("WORKERS", "4")))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-vram-gb", type=float, default=22.0)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "results" / "gpu_optimization" / "batch_size_test.csv",
    )
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=PROJECT_ROOT / "results" / "gpu_optimization" / "batch_size_probes",
    )
    parser.add_argument("--probe-batch", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--probe-output-json", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.probe_batch is not None:
        run_probe(args)
        return

    output_csv = args.output_csv.expanduser().resolve()
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.project_dir.expanduser().resolve().mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for batch_size in args.batches:
        row = run_probe_subprocess(args, batch_size=batch_size)
        rows.append(row)
        write_csv(output_csv, rows)
        print(format_probe_row(row), flush=True)

    recommendation = recommend_batch(rows, max_vram_gb=args.max_vram_gb)
    if recommendation is None:
        print(f"No candidate batch size stayed within {args.max_vram_gb:.1f} GB.")
    else:
        print(
            f"Recommended batch size: {recommendation['batch_size']} "
            f"(peak={recommendation['peak_vram_gb']:.2f} GB <= {args.max_vram_gb:.1f} GB)"
        )
    print(f"Wrote: {output_csv}")


def run_probe_subprocess(args: argparse.Namespace, *, batch_size: int) -> dict[str, Any]:
    output_dir = args.output_csv.expanduser().resolve().parent
    output_json = output_dir / f"batch_size_{batch_size}_probe.json"
    log_path = output_dir / f"batch_size_{batch_size}_probe.log"
    if output_json.exists():
        output_json.unlink()

    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--probe-batch",
        str(batch_size),
        "--probe-output-json",
        str(output_json),
        "--data-yaml",
        str(args.data_yaml),
        "--weights",
        str(args.weights),
        "--epochs",
        str(args.epochs),
        "--imgsz",
        str(args.imgsz),
        "--device",
        str(args.device),
        "--workers",
        str(args.workers),
        "--seed",
        str(args.seed),
        "--project-dir",
        str(args.project_dir),
    ]
    env = os.environ.copy()
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    start = time.perf_counter()
    with log_path.open("w", encoding="utf-8") as log_handle:
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
    elapsed = time.perf_counter() - start

    if output_json.exists():
        row = json.loads(output_json.read_text(encoding="utf-8"))
    else:
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        row = {
            "batch_size": batch_size,
            "peak_vram_gb": "",
            "time_per_epoch_sec": "",
            "status": "oom" if is_oom_text(log_text) else "failed",
            "elapsed_sec": round(elapsed, 3),
            "log_path": str(log_path),
        }
    if completed.returncode != 0 and row.get("status") == "ok":
        row["status"] = "failed"
    row.setdefault("log_path", str(log_path))
    return normalize_row(row)


def run_probe(args: argparse.Namespace) -> None:
    batch_size = int(args.probe_batch)
    output_json = args.probe_output_json.expanduser().resolve()
    output_json.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    row: dict[str, Any] = {
        "batch_size": batch_size,
        "peak_vram_gb": "",
        "time_per_epoch_sec": "",
        "status": "failed",
    }

    try:
        import numpy as np
        import torch
        from ultralytics import YOLO

        set_seed(args.seed, torch=torch, np=np)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available.")

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        model = YOLO(str(args.weights))
        model.train(
            data=str(Path(args.data_yaml).expanduser()),
            epochs=int(args.epochs),
            imgsz=int(args.imgsz),
            batch=batch_size,
            device=str(args.device),
            workers=int(args.workers),
            seed=int(args.seed),
            project=str(args.project_dir.expanduser().resolve()),
            name=f"batch_{batch_size}",
            exist_ok=True,
            optimizer="auto",
            pretrained=True,
            deterministic=True,
            single_cls=False,
            verbose=True,
            plots=False,
        )
        peak_gb = torch.cuda.max_memory_allocated() / (1024**3)
        elapsed = time.perf_counter() - start
        row.update(
            {
                "peak_vram_gb": round(peak_gb, 4),
                "time_per_epoch_sec": round(elapsed / max(int(args.epochs), 1), 4),
                "status": "ok",
                "elapsed_sec": round(elapsed, 3),
            }
        )
    except RuntimeError as exc:
        elapsed = time.perf_counter() - start
        row.update(
            {
                "status": "oom" if is_oom_text(str(exc)) else "failed",
                "elapsed_sec": round(elapsed, 3),
                "error": str(exc),
            }
        )
        try:
            import torch

            if torch.cuda.is_available():
                row["peak_vram_gb"] = round(torch.cuda.max_memory_allocated() / (1024**3), 4)
                torch.cuda.empty_cache()
        except Exception:
            pass
    except Exception as exc:
        elapsed = time.perf_counter() - start
        row.update(
            {
                "status": "failed",
                "elapsed_sec": round(elapsed, 3),
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )

    output_json.write_text(json.dumps(normalize_row(row), indent=2) + "\n", encoding="utf-8")


def set_seed(seed: int, *, torch: Any, np: Any) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def is_oom_text(text: str) -> bool:
    lowered = text.lower()
    return "outofmemory" in lowered or "out of memory" in lowered or "cuda out" in lowered


def normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "batch_size": int(row.get("batch_size", 0)),
        "peak_vram_gb": row.get("peak_vram_gb", ""),
        "time_per_epoch_sec": row.get("time_per_epoch_sec", ""),
        "status": row.get("status", "failed"),
        "elapsed_sec": row.get("elapsed_sec", ""),
        "log_path": row.get("log_path", ""),
        "error": row.get("error", ""),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["batch_size", "peak_vram_gb", "time_per_epoch_sec", "status", "elapsed_sec", "log_path", "error"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def recommend_batch(rows: list[dict[str, Any]], *, max_vram_gb: float) -> dict[str, Any] | None:
    candidates = []
    for row in rows:
        try:
            peak = float(row["peak_vram_gb"])
        except Exception:
            continue
        if row.get("status") == "ok" and peak <= max_vram_gb:
            candidates.append(row)
    if not candidates:
        return None
    return max(candidates, key=lambda item: int(item["batch_size"]))


def format_probe_row(row: dict[str, Any]) -> str:
    return (
        f"batch={row['batch_size']} status={row['status']} "
        f"peak_vram_gb={row['peak_vram_gb']} "
        f"time_per_epoch_sec={row['time_per_epoch_sec']}"
    )


if __name__ == "__main__":
    main()
