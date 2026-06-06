#!/usr/bin/env python3
"""Export low-confidence VNWoodKnot predictions for offline threshold analysis.

This script is intended for the Vast.ai GPU server. The parent process only
queues jobs. Each checkpoint is evaluated in a subprocess with
CUDA_VISIBLE_DEVICES set to one physical GPU, so no DataParallel/DDP is used.
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
import re
import subprocess
import sys
import threading
import time
from typing import Any

import numpy as np
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINT_ROOT = PROJECT_ROOT / "results" / "multiseed" / "vnwoodknot" / "per_seed" / "runs"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "negative_aware" / "predictions"
DEFAULT_DATA_YAML = Path(
    os.environ.get(
        "WOOD_DC_VN_BASELINE_DATASET_YAML",
        "/workspace/data/vnwoodknot/benchmarks/vnwoodknot_live_dead_2class_yolo/dataset.yaml",
    )
)
VARIANTS = ("baseline", "p2_illumination", "a1_crop", "a2_colorjitter", "p4_a4_combined")
SEEDS = (42, 43, 44)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RUN_RE = re.compile(r"^(?P<variant>.+)_seed(?P<seed>\d+)$")


@dataclass(frozen=True)
class InferenceJob:
    variant: str
    seed: int
    run_dir: Path
    checkpoint: Path
    output_path: Path
    index: int
    total: int

    @property
    def job_id(self) -> str:
        return f"{self.variant}_seed{self.seed}"


@dataclass(frozen=True)
class TestRecord:
    image_path: Path
    image_name: str
    canonical_id: str
    width: int
    height: int
    boxes_xyxy: np.ndarray
    labels: np.ndarray
    class_names: tuple[str, ...]

    @property
    def is_knot_free(self) -> bool:
        return len(self.labels) == 0

    @property
    def gt_boxes_json(self) -> list[list[Any]]:
        boxes = []
        for box, label in zip(self.boxes_xyxy, self.labels):
            x1, y1, x2, y2 = [float(value) for value in box]
            class_id = int(label)
            class_name = self.class_names[class_id] if 0 <= class_id < len(self.class_names) else f"class_{class_id}"
            boxes.append([round(x1, 4), round(y1, 4), round(x2 - x1, 4), round(y2 - y1, 4), class_name])
        return boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", default="0,1", help="Comma-separated physical GPU IDs.")
    parser.add_argument("--checkpoint-root", type=Path, default=DEFAULT_CHECKPOINT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--data-yaml",
        type=Path,
        default=None,
        help="Optional fixed VNWoodKnot dataset YAML. Defaults to each run's config_used.yaml data_yaml.",
    )
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--conf", type=float, default=0.01, help="Low base confidence used to export candidate detections.")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing prediction JSON files.")
    parser.add_argument("--dry-run", action="store_true", help="Print jobs without running inference.")
    parser.add_argument("--single-run-dir", type=Path, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--device", default="0", help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.single_run_dir:
        run_single_checkpoint(args)
        return

    jobs = discover_jobs(args)
    if args.dry_run:
        print(f"Dry run: {len(jobs)} checkpoint(s)")
        for job in jobs:
            print(f"{job.index:02d}/{job.total} {job.job_id} checkpoint={job.checkpoint} output={job.output_path}")
        return

    gpus = parse_gpus(args.gpus)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = PROJECT_ROOT / "results" / "negative_aware" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    run_log = log_dir / "threshold_sweep_inference_log.csv"
    ensure_run_log(run_log)

    work_queue: queue.Queue[InferenceJob] = queue.Queue()
    for job in jobs:
        work_queue.put(job)

    state = RunnerState(run_log=run_log)
    print(f"Starting {len(jobs)} VNWoodKnot inference jobs on GPUs: {', '.join(gpus)}")
    print(f"Prediction output: {args.output_dir.resolve()}")
    threads = []
    for gpu_id in gpus:
        thread = threading.Thread(target=worker_loop, args=(gpu_id, args, work_queue, state), daemon=False)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    print("All queued inference jobs finished or were skipped.")


def run_single_checkpoint(args: argparse.Namespace) -> None:
    run_dir = args.single_run_dir.expanduser().resolve()
    checkpoint = run_dir / "ultralytics" / "train" / "weights" / "best.pt"
    if not checkpoint.exists():
        raise SystemExit(f"Missing checkpoint: {checkpoint}")
    variant, seed = parse_run_name(run_dir.name)
    output_path = args.output_dir.expanduser().resolve() / f"{variant}_seed{seed}_predictions.json"
    if output_path.exists() and not args.overwrite:
        print(f"Existing output; use --overwrite to replace: {output_path}")
        return

    data_yaml = resolve_data_yaml(run_dir, args.data_yaml)
    records, class_names = load_yolo_test_records(data_yaml)
    predictions_by_key = predict_records(
        checkpoint=checkpoint,
        records=records,
        class_names=class_names,
        imgsz=args.imgsz,
        batch=args.batch,
        max_det=args.max_det,
        conf=args.conf,
        device=args.device,
    )

    payload = {
        "checkpoint": f"{variant}_seed{seed}",
        "variant": variant,
        "seed": seed,
        "checkpoint_path": str(checkpoint),
        "dataset_yaml": str(data_yaml),
        "split": "test",
        "base_confidence_threshold": float(args.conf),
        "class_names": list(class_names),
        "num_images": len(records),
        "num_knot_free_images": sum(1 for record in records if record.is_knot_free),
        "images": [],
    }
    for record in records:
        payload["images"].append(
            {
                "image": record.image_name,
                "canonical_id": record.canonical_id,
                "image_path": str(record.image_path),
                "width": record.width,
                "height": record.height,
                "is_knot_free": record.is_knot_free,
                "gt_boxes": record.gt_boxes_json,
                "predictions": predictions_by_key.get(record.canonical_id, []),
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote: {output_path}")


def discover_jobs(args: argparse.Namespace) -> list[InferenceJob]:
    checkpoint_root = args.checkpoint_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    jobs: list[InferenceJob] = []
    expected = [(variant, seed) for variant in args.variants for seed in args.seeds]
    total = len(expected)
    for index, (variant, seed) in enumerate(expected, start=1):
        run_dir = checkpoint_root / f"{variant}_seed{seed}"
        checkpoint = run_dir / "ultralytics" / "train" / "weights" / "best.pt"
        output_path = output_dir / f"{variant}_seed{seed}_predictions.json"
        if not checkpoint.exists():
            print(f"WARNING: missing checkpoint for {variant}_seed{seed}: {checkpoint}")
            continue
        jobs.append(
            InferenceJob(
                variant=variant,
                seed=seed,
                run_dir=run_dir,
                checkpoint=checkpoint,
                output_path=output_path,
                index=index,
                total=total,
            )
        )
    return jobs


def worker_loop(gpu_id: str, args: argparse.Namespace, work_queue: queue.Queue[InferenceJob], state: "RunnerState") -> None:
    while True:
        try:
            job = work_queue.get_nowait()
        except queue.Empty:
            return
        try:
            run_job(gpu_id, args, job, state)
        finally:
            work_queue.task_done()


def run_job(gpu_id: str, args: argparse.Namespace, job: InferenceJob, state: "RunnerState") -> None:
    started = utc_now()
    if job.output_path.exists() and not args.overwrite:
        finished = utc_now()
        state.append(log_row(job, gpu_id, started, finished, "skipped_existing", ""))
        count = state.mark_complete()
        print_progress(count, job, gpu_id, "skipped_existing", started, finished)
        return

    log_path = PROJECT_ROOT / "results" / "negative_aware" / "logs" / f"{job.job_id}_inference.log"
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--single-run-dir",
        str(job.run_dir),
        "--output-dir",
        str(args.output_dir),
        "--conf",
        str(args.conf),
        "--imgsz",
        str(args.imgsz),
        "--batch",
        str(args.batch),
        "--max-det",
        str(args.max_det),
        "--device",
        "0",
    ]
    if args.overwrite:
        command.append("--overwrite")
    if args.data_yaml:
        command.extend(["--data-yaml", str(args.data_yaml)])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    env["DEVICE"] = "0"
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False)
    finished = utc_now()
    if completed.returncode == 0 and job.output_path.exists():
        status, error = "ok", ""
    else:
        status, error = "failed", f"Exit code {completed.returncode}. See {log_path}"
    state.append(log_row(job, gpu_id, started, finished, status, error))
    count = state.mark_complete()
    print_progress(count, job, gpu_id, status if status == "ok" else error, started, finished)


def resolve_data_yaml(run_dir: Path, override: Path | None) -> Path:
    if override is not None:
        path = override.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Override data YAML does not exist: {path}")
        return path
    config_path = run_dir / "config_used.yaml"
    if config_path.exists():
        config = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        value = (config.get("dataset") or {}).get("data_yaml")
        if value:
            path = Path(str(value)).expanduser()
            if path.exists():
                return path.resolve()
            raise FileNotFoundError(f"Run config dataset YAML is not accessible: {path}")
    if DEFAULT_DATA_YAML.exists():
        return DEFAULT_DATA_YAML.resolve()
    raise FileNotFoundError(f"Could not resolve VNWoodKnot dataset YAML for run: {run_dir}")


def load_yolo_test_records(data_yaml: Path) -> tuple[list[TestRecord], tuple[str, ...]]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    dataset_root = resolve_dataset_root(data_yaml, data)
    class_names = tuple(normalize_names(data.get("names")))
    test_dirs = resolve_split_dirs(data_yaml, data, "test")
    if not test_dirs:
        raise ValueError(f"Dataset YAML has no test split: {data_yaml}")

    records: list[TestRecord] = []
    for test_dir in test_dirs:
        image_paths = sorted(path for path in test_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
        for image_path in image_paths:
            label_path = label_for_image(image_path, dataset_root=dataset_root, split="test", split_dir=test_dir)
            with Image.open(image_path) as image:
                width, height = image.size
            boxes_xyxy, labels = load_yolo_label_file(label_path, width=width, height=height)
            image_name = relative_image_name(image_path, dataset_root, test_dir)
            records.append(
                TestRecord(
                    image_path=image_path,
                    image_name=image_name,
                    canonical_id=canonical_id(image_name),
                    width=width,
                    height=height,
                    boxes_xyxy=boxes_xyxy,
                    labels=labels,
                    class_names=class_names,
                )
            )
    if not records:
        raise ValueError(f"No test images found from dataset YAML: {data_yaml}")
    print(
        f"Loaded test set: images={len(records)} knot_free={sum(record.is_knot_free for record in records)} "
        f"dataset={data_yaml}"
    )
    return records, class_names


def predict_records(
    *,
    checkpoint: Path,
    records: list[TestRecord],
    class_names: tuple[str, ...],
    imgsz: int,
    batch: int,
    max_det: int,
    conf: float,
    device: str,
) -> dict[str, list[dict[str, Any]]]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ImportError("ultralytics is required for threshold-sweep inference on Vast.") from exc

    model = YOLO(str(checkpoint))
    predictions_by_key: dict[str, list[dict[str, Any]]] = {}
    batch = max(1, int(batch))
    for start in range(0, len(records), batch):
        batch_records = records[start : start + batch]
        result_iter = model.predict(
            source=[str(record.image_path) for record in batch_records],
            stream=True,
            imgsz=int(imgsz),
            conf=float(conf),
            max_det=int(max_det),
            device=str(device),
            batch=len(batch_records),
            verbose=False,
        )
        for record, result in zip(batch_records, result_iter):
            boxes = result.boxes
            rows: list[dict[str, Any]] = []
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32).reshape(-1, 4)
                scores = boxes.conf.detach().cpu().numpy().astype(np.float32)
                labels = boxes.cls.detach().cpu().numpy().astype(np.int64)
                ious = box_iou(xyxy, record.boxes_xyxy) if len(record.boxes_xyxy) else np.zeros((len(xyxy), 0), dtype=np.float32)
                for box, score, label, row_ious in zip(xyxy, scores, labels, ious):
                    x1, y1, x2, y2 = [float(value) for value in box]
                    class_id = int(label)
                    class_name = class_names[class_id] if 0 <= class_id < len(class_names) else f"class_{class_id}"
                    rows.append(
                        {
                            "bbox": [round(x1, 4), round(y1, 4), round(x2 - x1, 4), round(y2 - y1, 4)],
                            "conf": round(float(score), 6),
                            "class": class_name,
                            "class_id": class_id,
                            "max_iou_gt": round(float(np.max(row_ious)), 6) if row_ious.size else 0.0,
                        }
                    )
            predictions_by_key[record.canonical_id] = rows
    return predictions_by_key


def load_yolo_label_file(label_path: Path, *, width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    boxes: list[list[float]] = []
    labels: list[int] = []
    if label_path.exists():
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError(f"Invalid YOLO label at {label_path}:{line_number}")
            class_id = int(float(parts[0]))
            cx, cy, bw, bh = [float(value) for value in parts[1:]]
            x1 = (cx - bw / 2.0) * width
            y1 = (cy - bh / 2.0) * height
            x2 = (cx + bw / 2.0) * width
            y2 = (cy + bh / 2.0) * height
            boxes.append([x1, y1, x2, y2])
            labels.append(class_id)
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4), np.asarray(labels, dtype=np.int64)


def box_iou(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    if boxes1.size == 0 or boxes2.size == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)
    top_left = np.maximum(boxes1[:, None, :2], boxes2[None, :, :2])
    bottom_right = np.minimum(boxes1[:, None, 2:], boxes2[None, :, 2:])
    wh = np.clip(bottom_right - top_left, 0.0, None)
    inter = wh[..., 0] * wh[..., 1]
    area1 = np.clip(boxes1[:, 2] - boxes1[:, 0], 0.0, None) * np.clip(boxes1[:, 3] - boxes1[:, 1], 0.0, None)
    area2 = np.clip(boxes2[:, 2] - boxes2[:, 0], 0.0, None) * np.clip(boxes2[:, 3] - boxes2[:, 1], 0.0, None)
    union = np.clip(area1[:, None] + area2[None, :] - inter, 1e-8, None)
    return inter / union


def parse_run_name(name: str) -> tuple[str, int]:
    match = RUN_RE.match(name)
    if not match:
        raise ValueError(f"Unexpected run directory name: {name}")
    return match.group("variant"), int(match.group("seed"))


def normalize_names(names: Any) -> list[str]:
    if isinstance(names, dict):
        return [str(names[key]) for key in sorted(names, key=lambda item: int(item))]
    if isinstance(names, list):
        return [str(name) for name in names]
    raise ValueError("Dataset YAML must define names as a list or mapping.")


def resolve_dataset_root(data_yaml: Path, data: dict[str, Any]) -> Path:
    root = Path(str(data.get("path") or data_yaml.parent)).expanduser()
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.resolve()


def resolve_split_dirs(data_yaml: Path, data: dict[str, Any], split: str) -> list[Path]:
    value = data.get(split)
    values = value if isinstance(value, list) else [value]
    root = resolve_dataset_root(data_yaml, data)
    dirs = []
    for item in values:
        if not item:
            continue
        path = Path(str(item)).expanduser()
        dirs.append(path.resolve() if path.is_absolute() else (root / path).resolve())
    return dirs


def label_for_image(image_path: Path, *, dataset_root: Path, split: str, split_dir: Path) -> Path:
    try:
        rel = image_path.relative_to(split_dir)
        return (dataset_root / "labels" / split / rel).with_suffix(".txt")
    except ValueError:
        pass
    try:
        rel = image_path.relative_to(dataset_root)
        parts = list(rel.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            return (dataset_root / Path(*parts)).with_suffix(".txt")
    except ValueError:
        pass
    return image_path.with_suffix(".txt")


def relative_image_name(image_path: Path, dataset_root: Path, split_dir: Path) -> str:
    for base in (dataset_root, split_dir):
        try:
            return image_path.relative_to(base).as_posix()
        except ValueError:
            continue
    return image_path.name


def canonical_id(value: str | Path) -> str:
    return str(value).replace("\\", "/").lower()


def parse_gpus(value: str) -> list[str]:
    gpus = [item.strip() for item in value.split(",") if item.strip()]
    if not gpus:
        raise SystemExit("--gpus must contain at least one GPU id")
    return gpus


class RunnerState:
    def __init__(self, *, run_log: Path) -> None:
        self.run_log = run_log
        self.completed = 0
        self.progress_lock = threading.Lock()
        self.csv_lock = threading.Lock()

    def mark_complete(self) -> int:
        with self.progress_lock:
            self.completed += 1
            return self.completed

    def append(self, row: dict[str, Any]) -> None:
        with self.csv_lock:
            with self.run_log.open("a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=run_log_fieldnames())
                writer.writerow(row)


def ensure_run_log(path: Path) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=run_log_fieldnames())
        writer.writeheader()


def run_log_fieldnames() -> list[str]:
    return ["job_id", "variant", "seed", "gpu_id", "start_time", "end_time", "duration_min", "status", "error"]


def log_row(job: InferenceJob, gpu_id: str, started: datetime, finished: datetime, status: str, error: str) -> dict[str, Any]:
    return {
        "job_id": job.job_id,
        "variant": job.variant,
        "seed": job.seed,
        "gpu_id": gpu_id,
        "start_time": started.isoformat(),
        "end_time": finished.isoformat(),
        "duration_min": f"{((finished - started).total_seconds() / 60.0):.3f}",
        "status": status,
        "error": error,
    }


def print_progress(count: int, job: InferenceJob, gpu_id: str, status: str, started: datetime, finished: datetime) -> None:
    duration = human_duration((finished - started).total_seconds())
    print(f"[{count}/{job.total}] GPU-{gpu_id} | {job.variant} | seed={job.seed} | {status} in {duration}", flush=True)


def human_duration(seconds: float) -> str:
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{sec:02d}s"
    return f"{minutes}m{sec:02d}s"


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


if __name__ == "__main__":
    main()
