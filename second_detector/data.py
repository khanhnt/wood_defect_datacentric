"""YOLO-tree loader used by the locked Faster R-CNN protocol."""

from __future__ import annotations

import random
import os
from pathlib import Path
from typing import Any

import numpy as np
import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def read_dataset_yaml(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    root = Path(payload.get("path", path.parent)).expanduser()
    if not root.is_absolute():
        root = (path.parent / root).resolve()
    names_raw = payload.get("names", {})
    if isinstance(names_raw, list):
        names = list(names_raw)
    else:
        names = [str(names_raw[index] if index in names_raw else names_raw[str(index)]) for index in range(len(names_raw))]
    return {**payload, "yaml_path": path, "root": root, "names": names}


def split_paths(dataset_yaml: Path, split: str) -> tuple[Path, Path, list[str]]:
    payload = read_dataset_yaml(dataset_yaml)
    image_root = Path(payload[split])
    if not image_root.is_absolute():
        image_root = payload["root"] / image_root
    image_root = image_root.resolve()
    try:
        relative = image_root.relative_to(payload["root"] / "images")
        label_root = (payload["root"] / "labels" / relative).resolve()
    except ValueError:
        label_root = Path(str(image_root).replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}", 1))
    return image_root, label_root, payload["names"]


def list_images(image_root: Path) -> list[Path]:
    return sorted(path for path in image_root.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)


class YoloDetectionDataset:
    """Torch dataset backed by an already materialized YOLO image/label tree."""

    def __init__(self, dataset_yaml: Path, split: str) -> None:
        import torch
        from PIL import Image

        self.torch = torch
        self.Image = Image
        self.dataset_yaml = dataset_yaml.expanduser().resolve()
        self.split = split
        self.image_root, self.label_root, self.class_names = split_paths(self.dataset_yaml, split)
        if not self.image_root.exists():
            raise FileNotFoundError(f"Missing {split} image directory: {self.image_root}")
        self.images = list_images(self.image_root)
        if not self.images:
            raise FileNotFoundError(f"No images found in {self.image_root}")

    def __len__(self) -> int:
        return len(self.images)

    def label_path(self, image_path: Path) -> Path:
        return self.label_root / image_path.relative_to(self.image_root).with_suffix(".txt")

    def __getitem__(self, index: int):
        image_path = self.images[index]
        label_path = self.label_path(image_path)
        if not label_path.exists():
            raise FileNotFoundError(f"Missing label for {image_path}: {label_path}")
        image = self.Image.open(image_path).convert("RGB")
        width, height = image.size
        array = np.array(image, dtype=np.float32, copy=True) / 255.0
        image_tensor = self.torch.from_numpy(array).permute(2, 0, 1).contiguous()

        boxes: list[list[float]] = []
        labels: list[int] = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id, x, y, box_width, box_height = map(float, line.split()[:5])
            x1 = max(0.0, (x - box_width / 2.0) * width)
            y1 = max(0.0, (y - box_height / 2.0) * height)
            x2 = min(float(width), (x + box_width / 2.0) * width)
            y2 = min(float(height), (y + box_height / 2.0) * height)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"Invalid box in {label_path}: {line}")
            boxes.append([x1, y1, x2, y2])
            labels.append(int(class_id) + 1)

        box_tensor = self.torch.tensor(boxes, dtype=self.torch.float32).reshape(-1, 4)
        label_tensor = self.torch.tensor(labels, dtype=self.torch.int64)
        area = (box_tensor[:, 2] - box_tensor[:, 0]) * (box_tensor[:, 3] - box_tensor[:, 1])
        target = {
            "boxes": box_tensor,
            "labels": label_tensor,
            "image_id": self.torch.tensor(index, dtype=self.torch.int64),
            "area": area,
            "iscrowd": self.torch.zeros((len(box_tensor),), dtype=self.torch.int64),
        }
        metadata = {
            "image": image_path.name,
            "canonical_id": str(image_path.relative_to(self.image_root)),
            "image_path": str(image_path),
            "width": width,
            "height": height,
            "is_knot_free": len(boxes) == 0,
        }
        return image_tensor, target, metadata


def collate_detection(batch):
    images, targets, metadata = zip(*batch)
    return list(images), list(targets), list(metadata)


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = __import__("torch").initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_loader(dataset_yaml: Path, split: str, batch_size: int, workers: int, seed: int, shuffle: bool):
    import torch
    from torch.utils.data import DataLoader

    dataset = YoloDetectionDataset(dataset_yaml, split)
    generator = torch.Generator().manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(workers),
        collate_fn=collate_detection,
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=bool(workers),
        pin_memory=True,
    )
