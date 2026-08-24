"""Training, evaluation, and export engine for the locked Faster R-CNN protocol."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from analysis.fasterrcnn_prediction_adapter import adapt_fasterrcnn_image
from second_detector.data import build_loader
from second_detector.metrics import box_iou_numpy, match_predictions, summarize_stats
from second_detector.model import build_model
from second_detector.protocol import locked_config


def set_deterministic_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment_payload() -> dict[str, Any]:
    import torch
    import torchvision

    try:
        import ultralytics

        ultralytics_version = ultralytics.__version__
    except Exception:
        ultralytics_version = None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "ultralytics": ultralytics_version,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def move_targets(targets, device):
    return [{key: value.to(device) if hasattr(value, "to") else value for key, value in target.items()} for target in targets]


def evaluate(
    model,
    loader,
    device,
    class_names: list[str],
    *,
    export_images: bool = False,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    model.eval()
    stats: list[dict[str, np.ndarray]] = []
    exported: list[dict[str, Any]] = []
    with torch.inference_mode():
        for images, targets, metadata_rows in loader:
            outputs = model([image.to(device, non_blocking=True) for image in images])
            for output, target, metadata in zip(outputs, targets, metadata_rows):
                pred_boxes = output["boxes"].detach().cpu().numpy().astype(np.float32)
                pred_scores = output["scores"].detach().cpu().numpy().astype(np.float32)
                pred_classes = output["labels"].detach().cpu().numpy().astype(np.int64) - 1
                true_boxes = target["boxes"].detach().cpu().numpy().astype(np.float32)
                true_classes = target["labels"].detach().cpu().numpy().astype(np.int64) - 1
                correct = match_predictions(pred_classes, true_classes, box_iou_numpy(true_boxes, pred_boxes))
                stats.append(
                    {
                        "correct": correct,
                        "confidence": pred_scores,
                        "pred_class": pred_classes,
                        "target_class": true_classes,
                    }
                )
                if export_images:
                    exported.append(
                        adapt_fasterrcnn_image(
                            image=metadata["image"],
                            canonical_id=metadata["canonical_id"],
                            image_path=metadata["image_path"],
                            width=metadata["width"],
                            height=metadata["height"],
                            gt_boxes_xyxy=true_boxes,
                            gt_class_ids=true_classes,
                            pred_boxes_xyxy=pred_boxes,
                            pred_scores=pred_scores,
                            pred_class_ids=output["labels"].detach().cpu().numpy(),
                            class_names=class_names,
                            min_confidence=0.001,
                        )
                    )
    summary = summarize_stats(stats, class_names)
    summary["n_images"] = len(loader.dataset)
    return summary, exported


def save_checkpoint(path: Path, *, model, optimizer, scheduler, epoch: int, best_metric: float, config: dict) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "best_metric": float(best_metric),
            "best_metric_name": "mAP50_95",
            "class_names": ["live_knot", "dead_knot"],
            "config": config,
        },
        path,
    )


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def train_one(
    *,
    data_yaml: Path,
    output_dir: Path,
    variant: str,
    seed: int,
    device_name: str,
    epochs: int,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    import torch

    set_deterministic_seed(seed)
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    weights_dir = output_dir / "weights"
    weights_dir.mkdir()
    config = locked_config(epochs=epochs, batch_size=batch_size, workers=workers)
    config.update(
        {
            "dataset": "vnwoodknot",
            "variant": variant,
            "seed": int(seed),
            "data_yaml": str(data_yaml.expanduser().resolve()),
            "device": device_name,
        }
    )
    (output_dir / "config_used.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    (output_dir / "environment.json").write_text(json.dumps(environment_payload(), indent=2), encoding="utf-8")

    device = torch.device(device_name)
    train_loader = build_loader(data_yaml, "train", batch_size, workers, seed, True)
    val_loader = build_loader(data_yaml, "val", batch_size, workers, seed, False)
    class_names = list(train_loader.dataset.class_names)
    if class_names != ["live_knot", "dead_knot"]:
        raise ValueError(f"Unexpected VNWoodKnot classes: {class_names}")

    model = build_model(num_foreground_classes=len(class_names), pretrained=True).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[35, 45], gamma=0.1)
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    history: list[dict[str, Any]] = []
    best_metric = float("-inf")
    best_epoch = 0
    start = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, int(epochs) + 1):
        epoch_start = time.perf_counter()
        model.train()
        loss_totals: dict[str, float] = {}
        batches = 0
        for images, targets, _ in train_loader:
            images = [image.to(device, non_blocking=True) for image in images]
            targets = move_targets(targets, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                losses = model(images, targets)
                total_loss = sum(losses.values())
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            for name, value in losses.items():
                loss_totals[name] = loss_totals.get(name, 0.0) + float(value.detach())
            loss_totals["loss_total"] = loss_totals.get("loss_total", 0.0) + float(total_loss.detach())
            batches += 1

        validation, _ = evaluate(model, val_loader, device, class_names)
        metric = float(validation["mAP50_95"])
        scheduler.step()
        row = {
            "epoch": epoch,
            "time_sec": round(time.perf_counter() - epoch_start, 3),
            **{f"train/{name}": value / max(batches, 1) for name, value in sorted(loss_totals.items())},
            "metrics/precision(B)": validation["precision"],
            "metrics/recall(B)": validation["recall"],
            "metrics/mAP50(B)": validation["mAP50"],
            "metrics/mAP50-95(B)": validation["mAP50_95"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        write_history(output_dir / "results.csv", history)
        if metric >= best_metric:
            best_metric = metric
            best_epoch = epoch
            save_checkpoint(
                weights_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                best_metric=best_metric,
                config=config,
            )
        save_checkpoint(
            weights_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            best_metric=best_metric,
            config=config,
        )
        print(
            f"epoch={epoch}/{epochs} loss={row['train/loss_total']:.4f} "
            f"mAP50={validation['mAP50']:.5f} mAP50-95={metric:.5f}",
            flush=True,
        )

    elapsed = time.perf_counter() - start
    summary = {
        "status": "ok",
        "dataset": "vnwoodknot",
        "variant": variant,
        "seed": int(seed),
        "epochs": int(epochs),
        "best_epoch": best_epoch,
        "best_metric_name": "mAP50_95",
        "best_metric": best_metric,
        "elapsed_sec": elapsed,
        "time_per_epoch_sec": elapsed / max(int(epochs), 1),
        "peak_vram_gb": (
            torch.cuda.max_memory_allocated(device) / (1024**3) if device.type == "cuda" else 0.0
        ),
        "num_train_images": len(train_loader.dataset),
        "num_val_images": len(val_loader.dataset),
        "best_checkpoint": str(weights_dir / "best.pt"),
        "last_checkpoint": str(weights_dir / "last.pt"),
    }
    summary["best_checkpoint_sha256"] = sha256(weights_dir / "best.pt")
    summary["last_checkpoint_sha256"] = sha256(weights_dir / "last.pt")
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def load_checkpoint_model(checkpoint: Path, device_name: str):
    import torch

    device = torch.device(device_name)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model = build_model(num_foreground_classes=2, pretrained=False)
    model.load_state_dict(payload["model_state_dict"])
    model.to(device)
    return model, payload, device


def export_checkpoint(
    *,
    checkpoint: Path,
    data_yaml: Path,
    split: str,
    output_json: Path,
    variant: str,
    seed: int,
    device_name: str,
    batch_size: int,
    workers: int,
) -> dict[str, Any]:
    model, checkpoint_payload, device = load_checkpoint_model(checkpoint, device_name)
    loader = build_loader(data_yaml, split, batch_size, workers, seed, False)
    class_names = list(loader.dataset.class_names)
    summary, images = evaluate(model, loader, device, class_names, export_images=True)
    payload = {
        "dataset": "vnwoodknot",
        "detector": "fasterrcnn_mobilenet_v3_large_fpn",
        "variant": variant,
        "seed": int(seed),
        "split": split,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256(checkpoint),
        "checkpoint_epoch": checkpoint_payload.get("epoch"),
        "data_yaml": str(data_yaml.resolve()),
        "class_names": class_names,
        "inference_path": "torchvision_fasterrcnn_with_ultralytics_8_4_60_matching",
        "tp_source": "ultralytics_8_4_60_BaseValidator_match_predictions_equivalent",
        "settings": {"imgsz": 1024, "conf": 0.001, "iou": 0.7, "max_det": 300, "augment": False},
        "validator_metrics": summary,
        "images": images,
    }
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload), encoding="utf-8")
    return {"output_json": str(output_json), **summary}
