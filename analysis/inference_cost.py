#!/usr/bin/env python3
"""Measure YOLOv8s inference cost and preprocessing overhead.

The script is intended to run on the GPU server with existing trained
checkpoints and test images. It does not train models.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time
from typing import Any, Iterable

import cv2
import numpy as np
import yaml
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from wood_defect_datacentric.preprocessing.variants import apply_preprocessing, load_variant_config  # noqa: E402


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
DEFAULT_CHECKPOINTS = (
    PROJECT_ROOT / "results" / "multiseed" / "vnwoodknot" / "per_seed" / "runs" / "baseline_seed42" / "ultralytics" / "train" / "weights" / "best.pt",
    PROJECT_ROOT / "results 3" / "multiseed" / "vnwoodknot" / "per_seed" / "runs" / "baseline_seed42" / "ultralytics" / "train" / "weights" / "best.pt",
    PROJECT_ROOT / "results 2" / "multiseed" / "vnwoodknot" / "per_seed" / "runs" / "baseline_seed42" / "ultralytics" / "train" / "weights" / "best.pt",
    PROJECT_ROOT / "results" / "runs" / "vn_t0_yolov8s_baseline_e50" / "ultralytics" / "train" / "weights" / "best.pt",
)
DEFAULT_DATA_YAMLS = (
    os.environ.get("WOOD_DC_VSB_BASELINE_DATASET_YAML"),
    Path("/workspace/data/main_dataset/benchmarks/vsb7_3600_rare_first_yolo/dataset.yaml"),
    PROJECT_ROOT / "results 3" / "calibration_vsb" / "dataset.yaml",
    os.environ.get("WOOD_DC_VN_BASELINE_DATASET_YAML"),
    Path("/workspace/data/vnwoodknot/benchmarks/vnwoodknot_live_dead_2class_yolo/dataset.yaml"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-yaml", type=Path, default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output-json", type=Path, default=PROJECT_ROOT / "results" / "inference_cost.json")
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--precision", choices=("fp32", "fp16"), default="fp32")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--timed-images", type=int, default=300)
    parser.add_argument("--source-mode", choices=("ndarray", "path"), default="ndarray")
    parser.add_argument("--allow-repeat", action="store_true", help="Repeat images if the selected test split has fewer than --timed-images images.")
    parser.add_argument("--throughput-batches", nargs="*", type=int, default=[8, 16])
    parser.add_argument("--preprocess-config", type=Path, default=PROJECT_ROOT / "configs" / "preprocessing" / "P4_combined_safe.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint = resolve_checkpoint(args.checkpoint)
    data_yaml = resolve_data_yaml(args.data_yaml)
    image_paths = list_split_images(data_yaml, args.split)
    if len(image_paths) < int(args.timed_images) and not args.allow_repeat:
        raise SystemExit(
            f"Split has only {len(image_paths)} images, fewer than --timed-images={args.timed_images}. "
            "Use the VSB test split or pass --allow-repeat intentionally."
        )
    selected_paths = select_images(image_paths, int(args.timed_images), allow_repeat=bool(args.allow_repeat))

    print_reuse_report(checkpoint, data_yaml, args)
    env = inspect_environment(args)
    print_environment(env)

    try:
        import torch
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("torch and ultralytics are required on the timing machine.") from exc

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; run this benchmark on the RTX 3090 server.")

    model = YOLO(str(checkpoint))
    half = args.precision == "fp16"
    model_info = collect_model_info(model, checkpoint, int(args.imgsz))

    images = load_images(selected_paths, source_mode=args.source_mode)
    latency = measure_batch1_latency(
        model,
        images,
        image_paths=selected_paths,
        args=args,
        half=half,
        torch_module=torch,
    )
    preprocessing = measure_preprocessing_cost(selected_paths, args.preprocess_config)
    throughput = measure_batch_throughput(model, images, args=args, half=half, torch_module=torch)

    output = {
        "reused": {
            "checkpoint": str(checkpoint),
            "data_yaml": str(data_yaml),
            "evaluation_settings_source": "scripts/evaluate_corrected_common.py model.val(... imgsz, conf, iou)",
            "preprocessing_source": "preprocessing/variants.py apply_preprocessing + P4_combined_safe.yaml",
        },
        "environment": env,
        "settings": {
            "imgsz": int(args.imgsz),
            "conf": float(args.conf),
            "iou": float(args.iou),
            "device": str(args.device),
            "max_det": int(args.max_det),
            "precision": args.precision.upper(),
            "half": bool(half),
            "warmup": int(args.warmup),
            "timed_images": int(args.timed_images),
            "source_mode": args.source_mode,
            "split": args.split,
        },
        "model": model_info,
        "latency_batch1": latency,
        "preprocessing_overhead": preprocessing,
        "batch_throughput": throughput,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n", encoding="utf-8")

    print(f"\nJSON_DUMP: {args.output_json}")
    print_latex_table(model_info, latency)
    print_preprocessing_summary(preprocessing)
    print_summary_sentence(model_info, latency, preprocessing)


def resolve_checkpoint(path: Path | None) -> Path:
    if path is not None:
        checkpoint = path.expanduser().resolve()
        if not checkpoint.exists():
            raise SystemExit(f"Checkpoint does not exist: {checkpoint}")
        return checkpoint
    for candidate in DEFAULT_CHECKPOINTS:
        if str(candidate) and candidate.exists():
            return candidate.resolve()
    searched = "\n".join(f"- {candidate}" for candidate in DEFAULT_CHECKPOINTS)
    raise SystemExit(f"No default best.pt found. Pass --checkpoint. Searched:\n{searched}")


def resolve_data_yaml(path: Path | None) -> Path:
    if path is not None:
        data_yaml = path.expanduser().resolve()
        if not data_yaml.exists():
            raise SystemExit(f"Dataset YAML does not exist: {data_yaml}")
        return data_yaml
    for candidate in DEFAULT_DATA_YAMLS:
        if not candidate:
            continue
        candidate_path = Path(candidate).expanduser()
        if candidate_path.exists() and candidate_path.is_file():
            return candidate_path.resolve()
    searched = "\n".join(f"- {candidate}" for candidate in DEFAULT_DATA_YAMLS if candidate)
    raise SystemExit(f"No default dataset YAML found. Pass --data-yaml. Searched:\n{searched}")


def list_split_images(data_yaml: Path, split: str) -> list[Path]:
    raw = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(str(raw.get("path") or data_yaml.parent)).expanduser()
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    split_value = raw.get(split)
    if split_value is None:
        raise SystemExit(f"Split '{split}' not found in {data_yaml}")
    split_path = Path(str(split_value))
    if not split_path.is_absolute():
        split_path = root / split_path
    if split_path.is_file():
        paths = [Path(line.strip()) for line in split_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        paths = [path if path.is_absolute() else root / path for path in paths]
    else:
        paths = [path for path in split_path.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS]
    paths = sorted(path.resolve() for path in paths if path.exists())
    if not paths:
        raise SystemExit(f"No images found for split '{split}' from {data_yaml}: {split_path}")
    return paths


def select_images(paths: list[Path], count: int, *, allow_repeat: bool) -> list[Path]:
    if len(paths) >= count:
        return paths[:count]
    if not allow_repeat:
        return paths
    selected = []
    while len(selected) < count:
        selected.extend(paths)
    return selected[:count]


def load_images(paths: list[Path], *, source_mode: str) -> list[Any]:
    if source_mode == "path":
        return [str(path) for path in paths]
    images = []
    for path in paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"Failed to read image: {path}")
        images.append(image)
    return images


def collect_model_info(model: Any, checkpoint: Path, imgsz: int) -> dict[str, Any]:
    module = model.model
    params = int(sum(param.numel() for param in module.parameters()))
    trainable = int(sum(param.numel() for param in module.parameters() if param.requires_grad))
    gflops = get_gflops(module, imgsz)
    return {
        "parameters": params,
        "parameters_millions": params / 1_000_000.0,
        "trainable_parameters": trainable,
        "gflops_1024": gflops,
        "weights_mb": checkpoint.stat().st_size / (1024.0 * 1024.0),
        "checkpoint": str(checkpoint),
    }


def get_gflops(module: Any, imgsz: int) -> float | None:
    try:
        from ultralytics.utils.torch_utils import get_flops

        value = get_flops(module, imgsz=imgsz)
        if value is not None and not math.isnan(float(value)):
            return float(value)
    except Exception:
        pass
    try:
        info = module.info(verbose=False, imgsz=imgsz)
        if isinstance(info, (tuple, list)):
            for value in reversed(info):
                if isinstance(value, (int, float)) and float(value) > 0:
                    return float(value)
    except Exception:
        pass
    return None


def measure_batch1_latency(model: Any, images: list[Any], *, image_paths: list[Path], args: argparse.Namespace, half: bool, torch_module: Any) -> dict[str, Any]:
    warmup_count = max(30, int(args.warmup))
    timed_count = int(args.timed_images)
    warmup_sources = cycle_items(images, warmup_count)
    timed_sources = cycle_items(images, timed_count)

    for source in warmup_sources:
        _ = predict_once(model, source, args=args, half=half)
    synchronize(torch_module)

    rows = []
    for index, source in enumerate(timed_sources):
        synchronize(torch_module)
        start = time.perf_counter()
        result = predict_once(model, source, args=args, half=half)
        synchronize(torch_module)
        wall_ms = (time.perf_counter() - start) * 1000.0
        speed = getattr(result, "speed", {}) or {}
        preprocess_ms = float(speed.get("preprocess", float("nan")))
        inference_ms = float(speed.get("inference", float("nan")))
        postprocess_ms = float(speed.get("postprocess", float("nan")))
        total_speed_ms = preprocess_ms + inference_ms + postprocess_ms
        rows.append(
            {
                "index": index,
                "image": str(image_paths[index % len(image_paths)]),
                "preprocess_ms": preprocess_ms,
                "inference_ms": inference_ms,
                "postprocess_ms": postprocess_ms,
                "total_speed_ms": total_speed_ms,
                "wall_ms": wall_ms,
            }
        )

    summary = {
        "n_images": len(rows),
        "warmup_iterations": warmup_count,
        "preprocess_ms": summarize(row["preprocess_ms"] for row in rows),
        "inference_ms": summarize(row["inference_ms"] for row in rows),
        "postprocess_ms": summarize(row["postprocess_ms"] for row in rows),
        "total_speed_ms": summarize(row["total_speed_ms"] for row in rows),
        "wall_ms": summarize(row["wall_ms"] for row in rows),
        "fps_from_total_speed_mean": 1000.0 / mean(row["total_speed_ms"] for row in rows),
        "fps_from_wall_mean": 1000.0 / mean(row["wall_ms"] for row in rows),
        "per_image": rows,
    }
    return summary


def predict_once(model: Any, source: Any, *, args: argparse.Namespace, half: bool) -> Any:
    results = model.predict(
        source=source,
        imgsz=int(args.imgsz),
        conf=float(args.conf),
        iou=float(args.iou),
        max_det=int(args.max_det),
        device=str(args.device),
        batch=1,
        half=bool(half),
        augment=False,
        verbose=False,
        save=False,
        stream=False,
    )
    return results[0] if isinstance(results, list) else list(results)[0]


def measure_preprocessing_cost(image_paths: list[Path], config_path: Path) -> dict[str, Any]:
    variants = [
        PROJECT_ROOT / "configs" / "preprocessing" / "P1_CLAHE_luminance.yaml",
        PROJECT_ROOT / "configs" / "preprocessing" / "P2_illumination_normalization.yaml",
        config_path.expanduser().resolve(),
    ]
    output = {}
    rgb_images = []
    for path in image_paths:
        with Image.open(path) as image:
            rgb_images.append(np.asarray(image.convert("RGB")))
    for variant_path in variants:
        variant = load_variant_config(variant_path)
        for image in rgb_images[:10]:
            _ = apply_preprocessing(image, variant)
        rows = []
        for path, image in zip(image_paths, rgb_images):
            start = time.perf_counter()
            processed = apply_preprocessing(image, variant)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            rows.append({"image": str(path), "ms": elapsed_ms, "shape": list(processed.shape)})
        output[variant.name] = {
            "config": str(variant_path),
            "description": variant.description,
            "ms": summarize(row["ms"] for row in rows),
            "per_image": rows,
        }
    return output


def measure_batch_throughput(model: Any, images: list[Any], *, args: argparse.Namespace, half: bool, torch_module: Any) -> dict[str, Any]:
    output = {}
    for batch_size in args.throughput_batches:
        if batch_size <= 1:
            continue
        sources = cycle_items(images, int(args.timed_images))
        chunks = [sources[index : index + batch_size] for index in range(0, len(sources), batch_size)]
        chunks = [chunk for chunk in chunks if len(chunk) == batch_size]
        if not chunks:
            continue
        for chunk in chunks[:5]:
            _ = predict_batch(model, chunk, args=args, half=half, batch_size=batch_size)
        synchronize(torch_module)
        elapsed = 0.0
        n_images = 0
        for chunk in chunks:
            synchronize(torch_module)
            start = time.perf_counter()
            _ = predict_batch(model, chunk, args=args, half=half, batch_size=batch_size)
            synchronize(torch_module)
            elapsed += time.perf_counter() - start
            n_images += len(chunk)
        output[f"batch_{batch_size}"] = {
            "batch_size": int(batch_size),
            "n_images": int(n_images),
            "elapsed_sec": elapsed,
            "images_per_sec": n_images / elapsed if elapsed > 0 else float("nan"),
            "ms_per_image_wall": (elapsed * 1000.0 / n_images) if n_images else float("nan"),
        }
    return output


def predict_batch(model: Any, sources: list[Any], *, args: argparse.Namespace, half: bool, batch_size: int) -> list[Any]:
    results = model.predict(
        source=sources,
        imgsz=int(args.imgsz),
        conf=float(args.conf),
        iou=float(args.iou),
        max_det=int(args.max_det),
        device=str(args.device),
        batch=int(batch_size),
        half=bool(half),
        augment=False,
        verbose=False,
        save=False,
        stream=False,
    )
    return results if isinstance(results, list) else list(results)


def inspect_environment(args: argparse.Namespace) -> dict[str, Any]:
    env: dict[str, Any] = {
        "precision": args.precision.upper(),
        "torch_num_threads": None,
        "opencv_num_threads": None,
        "torch_version": None,
        "torch_cuda_version": None,
        "ultralytics_version": None,
        "gpu_name": None,
        "driver_version": None,
        "nvidia_smi_cuda_version": None,
    }
    try:
        import torch

        env["torch_version"] = torch.__version__
        env["torch_cuda_version"] = torch.version.cuda
        env["torch_num_threads"] = torch.get_num_threads()
        if torch.cuda.is_available():
            env["gpu_name"] = torch.cuda.get_device_name(int(str(args.device).split(",")[0]))
    except Exception as exc:
        env["torch_error"] = str(exc)
    try:
        import ultralytics

        env["ultralytics_version"] = getattr(ultralytics, "__version__", None)
    except Exception as exc:
        env["ultralytics_error"] = str(exc)
    try:
        env["opencv_num_threads"] = cv2.getNumThreads()
    except Exception:
        pass
    smi = run_command(["nvidia-smi"])
    env["nvidia_smi"] = smi
    env.update(parse_nvidia_smi(smi))
    return env


def run_command(command: list[str]) -> str:
    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        return ""
    return (completed.stdout or completed.stderr or "").strip()


def parse_nvidia_smi(text: str) -> dict[str, str | None]:
    out: dict[str, str | None] = {"driver_version": None, "nvidia_smi_cuda_version": None}
    driver_match = re.search(r"Driver Version:\s*([0-9.]+)", text)
    cuda_match = re.search(r"CUDA Version:\s*([0-9.]+)", text)
    if driver_match:
        out["driver_version"] = driver_match.group(1)
    if cuda_match:
        out["nvidia_smi_cuda_version"] = cuda_match.group(1)
    return out


def print_reuse_report(checkpoint: Path, data_yaml: Path, args: argparse.Namespace) -> None:
    print("REUSED PATHS / SETTINGS")
    print(f"- checkpoint: {checkpoint}")
    print(f"- data_yaml: {data_yaml}")
    print("- evaluation settings: scripts/evaluate_corrected_common.py uses model.val(... imgsz, conf, iou)")
    print(f"- current settings: imgsz={args.imgsz}, conf={args.conf}, iou={args.iou}, augment=False")
    print(f"- preprocessing function: preprocessing.variants.apply_preprocessing ({source_location(apply_preprocessing)})")
    print(f"- preprocessing config: {args.preprocess_config}")


def print_environment(env: dict[str, Any]) -> None:
    print("\nENVIRONMENT")
    for key in (
        "gpu_name",
        "driver_version",
        "nvidia_smi_cuda_version",
        "torch_version",
        "torch_cuda_version",
        "ultralytics_version",
        "precision",
        "torch_num_threads",
        "opencv_num_threads",
    ):
        print(f"- {key}: {env.get(key)}")


def print_latex_table(model_info: dict[str, Any], latency: dict[str, Any]) -> None:
    total = latency["total_speed_ms"]
    fps = latency["fps_from_total_speed_mean"]
    print("\nLATEX TABLE ROWS")
    print(f"Params (M) & {model_info['parameters_millions']:.2f} \\\\")
    gflops = model_info["gflops_1024"]
    print(f"GFLOPs & {format_optional(gflops, digits=1)} \\\\")
    print(f"Weights (MB) & {model_info['weights_mb']:.1f} \\\\")
    print(f"Preprocess ms & {fmt_pm(latency['preprocess_ms'])} \\\\")
    print(f"Inference ms & {fmt_pm(latency['inference_ms'])} \\\\")
    print(f"Postprocess ms & {fmt_pm(latency['postprocess_ms'])} \\\\")
    print(f"Total ms @ bs=1 & {fmt_pm(total)} \\\\")
    print(f"FPS @ bs=1 & {fps:.1f} \\\\")


def print_preprocessing_summary(preprocessing: dict[str, Any]) -> None:
    print("\nPREPROCESSING OVERHEAD")
    for name, row in preprocessing.items():
        print(f"- {name}: {fmt_pm(row['ms'])} ms/image")


def print_summary_sentence(model_info: dict[str, Any], latency: dict[str, Any], preprocessing: dict[str, Any]) -> None:
    p4 = preprocessing.get("P4_combined_safe", {})
    p4_ms = p4.get("ms", {}).get("mean", float("nan"))
    sentence = (
        f"YOLOv8s has {model_info['parameters_millions']:.2f}M parameters and "
        f"{format_optional(model_info['gflops_1024'], digits=1)} GFLOPs at 1024x1024; "
        f"on a single RTX 3090 it runs at {latency['total_speed_ms']['mean']:.2f} ms/image "
        f"({latency['fps_from_total_speed_mean']:.1f} FPS) at batch size 1, of which the detector accounts for "
        f"{latency['inference_ms']['mean']:.2f} ms; the preprocessing variants add up to {p4_ms:.2f} ms/image."
    )
    print("\nLATEX SUMMARY SENTENCE")
    print(sentence)


def cycle_items(items: list[Any], count: int) -> list[Any]:
    if len(items) >= count:
        return items[:count]
    output = []
    while len(output) < count:
        output.extend(items)
    return output[:count]


def synchronize(torch_module: Any) -> None:
    if torch_module.cuda.is_available():
        torch_module.cuda.synchronize()


def summarize(values: Iterable[float]) -> dict[str, float]:
    vals = [float(value) for value in values if not math.isnan(float(value))]
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "mean": mean(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def mean(values: Iterable[float]) -> float:
    vals = [float(value) for value in values]
    return sum(vals) / len(vals) if vals else float("nan")


def fmt_pm(row: dict[str, float]) -> str:
    return f"${row['mean']:.2f}\\pm{row['std']:.2f}$"


def format_optional(value: float | None, *, digits: int) -> str:
    return "n/a" if value is None else f"{float(value):.{digits}f}"


def source_location(func: Any) -> str:
    path = Path(inspect.getsourcefile(func) or "")
    try:
        rel = path.relative_to(PROJECT_ROOT)
    except ValueError:
        rel = path
    _, start_line = inspect.getsourcelines(func)
    return f"{rel}:{start_line}"


if __name__ == "__main__":
    main()
