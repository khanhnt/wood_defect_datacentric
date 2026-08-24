"""Fixed protocol and dataset resolution for the second-detector experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


SEEDS = (42, 43, 44)
JOB_VARIANTS = ("baseline", "a1_crop", "a2_colorjitter")
VARIANT_LABELS = {
    "baseline": "Baseline",
    "a1_crop": "A1 crop",
    "a2_colorjitter": "A2 color jitter",
}


@dataclass(frozen=True)
class FasterRCNNJob:
    variant: str
    seed: int
    data_yaml: Path
    output_dir: Path

    @property
    def job_id(self) -> str:
        return f"vnwoodknot_{self.variant}_seed{self.seed}"

    def to_dict(self) -> dict:
        result = asdict(self)
        result["data_yaml"] = str(self.data_yaml)
        result["output_dir"] = str(self.output_dir)
        result["job_id"] = self.job_id
        return result


def dataset_yaml_for(variant: str, seed: int, rebuilt_root: Path) -> Path:
    root = rebuilt_root.expanduser().resolve()
    if variant == "baseline":
        return root / "canonical" / "vnwoodknot" / "dataset.yaml"
    if variant == "a1_crop":
        name = "A1_defect_preserving_crop"
    elif variant == "a2_colorjitter":
        name = "A2_texture_aware_color_jitter"
    else:
        raise ValueError(f"Unsupported Faster R-CNN variant: {variant}")
    return root / "variants" / "vnwoodknot" / "augmentation" / f"seed{seed}" / name / "dataset.yaml"


def build_jobs(
    *,
    rebuilt_root: Path,
    results_root: Path,
    variants: Iterable[str] = JOB_VARIANTS,
    seeds: Iterable[int] = SEEDS,
) -> list[FasterRCNNJob]:
    jobs = []
    for variant in variants:
        if variant not in JOB_VARIANTS:
            raise ValueError(f"Unsupported Faster R-CNN variant: {variant}")
        for seed in seeds:
            seed = int(seed)
            jobs.append(
                FasterRCNNJob(
                    variant=variant,
                    seed=seed,
                    data_yaml=dataset_yaml_for(variant, seed, rebuilt_root),
                    output_dir=(
                        results_root.expanduser().resolve()
                        / "fasterrcnn"
                        / "vnwoodknot"
                        / "per_seed"
                        / "runs"
                        / f"{variant}_seed{seed}"
                    ),
                )
            )
    return jobs


def locked_config(*, epochs: int = 50, batch_size: int = 4, workers: int = 4) -> dict:
    return {
        "architecture": "fasterrcnn_mobilenet_v3_large_fpn",
        "initialization": "torchvision COCO DEFAULT weights",
        "num_foreground_classes": 2,
        "trainable_backbone_layers": 3,
        "image_size": 1024,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "workers": int(workers),
        "optimizer": "SGD",
        "learning_rate": 0.005,
        "momentum": 0.9,
        "weight_decay": 0.0005,
        "lr_scheduler": "MultiStepLR",
        "lr_milestones": [35, 45],
        "lr_gamma": 0.1,
        "amp": True,
        "best_metric": "mAP50_95",
        "score_threshold": 0.001,
        "nms_threshold": 0.7,
        "max_detections": 300,
        "rpn_score_threshold": 0.05,
        "online_augmentation": "none",
        "deterministic": True,
    }
