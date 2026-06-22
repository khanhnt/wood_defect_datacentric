#!/usr/bin/env python3
"""Evaluate best.pt checkpoints on corrected common val/test sets.

This script is intended for the Vast.ai box. It fixes the evaluation-set
confound where augmentation-materialized datasets baked crop/color jitter into
val/test. The corrected rule is:

- augmentation variants evaluate on the raw canonical dataset;
- preprocessing variants evaluate on full-image preprocessed canonical images;
- P4+A4 evaluates on P4 preprocessing only, with no crop/color jitter at eval.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 43, 44)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VSB_CLASSES = ("live_knot", "dead_knot", "resin", "knot_with_crack", "crack", "marrow", "knot_missing")


@dataclass(frozen=True)
class VariantSpec:
    dataset: str
    variant: str
    label: str
    preprocessing: str | None = None


SPECS = (
    VariantSpec("vnwoodknot", "baseline", "Baseline"),
    VariantSpec("vnwoodknot", "p2_illumination", "P2 illumination", "P2_illumination_normalization"),
    VariantSpec("vnwoodknot", "a1_crop", "A1 crop"),
    VariantSpec("vnwoodknot", "a2_colorjitter", "A2 color jitter"),
    VariantSpec("vnwoodknot", "p4_a4_combined", "P4+A4 combined", "P4_combined_safe"),
    VariantSpec("vsb_rarefirst", "baseline", "Baseline"),
    VariantSpec("vsb_rarefirst", "p1_clahe", "P1 CLAHE", "P1_CLAHE_luminance"),
    VariantSpec("vsb_rarefirst", "p2_illumination", "P2 illumination", "P2_illumination_normalization"),
    VariantSpec("vsb_rarefirst", "p3_unsharp", "P3 unsharp", "P3_mild_unsharp"),
    VariantSpec("vsb_rarefirst", "a1_crop", "A1 crop"),
    VariantSpec("vsb_rarefirst", "a2_colorjitter", "A2 color jitter"),
    VariantSpec("vsb_rarefirst", "p4_a4_combined", "P4+A4 combined", "P4_combined_safe"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-csv", type=Path, default=Path("results/corrected_common_eval/test_best_summary.csv"))
    parser.add_argument(
        "--eval-dataset-root",
        type=Path,
        default=Path("/workspace/data/wood_defect_datacentric/corrected_common_eval_yolo"),
    )
    parser.add_argument(
        "--vn-canonical-yaml",
        type=Path,
        default=Path("/workspace/data/vnwoodknot/benchmarks/vnwoodknot_live_dead_2class_yolo/dataset.yaml"),
    )
    parser.add_argument(
        "--vsb-canonical-yaml",
        type=Path,
        default=Path("/workspace/data/main_dataset/benchmarks/vsb7_3600_rare_first_yolo/dataset.yaml"),
    )
    parser.add_argument(
        "--vsb-manifest",
        type=Path,
        default=Path("/workspace/data/main_dataset/benchmarks/vsb7_3600_rare_first/manifest.jsonl"),
    )
    parser.add_argument("--vsb-images-root", type=Path, default=Path("/workspace/data/main_dataset/images"))
    parser.add_argument("--rebuild-vsb-canonical", action="store_true")
    parser.add_argument("--overwrite-vsb-canonical", action="store_true")
    parser.add_argument("--overwrite-eval-datasets", action="store_true")
    parser.add_argument("--dataset", choices=("all", "vnwoodknot", "vsb_rarefirst"), default="all")
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--splits", nargs="+", choices=("val", "test"), default=["val", "test"])
    parser.add_argument("--imgsz", type=int, default=1024)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--device", default="0")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--allow-missing", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rebuild_vsb_canonical:
        rebuild_vsb_canonical(args)

    specs = selected_specs(args)
    eval_yamls = prepare_eval_yamls(args, specs)
    write_eval_yaml_manifest(args, eval_yamls)
    if args.prepare_only:
        print("Prepared corrected eval datasets.")
        for key, path in sorted(eval_yamls.items()):
            print(f"{key[0]}:{key[1]} -> {path}")
        return

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is required on the Vast evaluation machine.") from exc

    rows: list[dict[str, Any]] = []
    for spec in specs:
        data_yaml = eval_yamls[(spec.dataset, spec.variant)]
        for seed in args.seeds:
            run = f"{spec.variant}_seed{seed}"
            checkpoint = (
                args.results_root.expanduser()
                / "multiseed"
                / spec.dataset
                / "per_seed"
                / "runs"
                / run
                / "ultralytics"
                / "train"
                / "weights"
                / "best.pt"
            )
            if not checkpoint.exists():
                message = f"Missing checkpoint: {checkpoint}"
                if args.allow_missing:
                    print(f"WARNING: {message}")
                    continue
                raise SystemExit(message)
            model = YOLO(str(checkpoint))
            for split in args.splits:
                n_images, n_instances = count_split(data_yaml, split)
                result = model.val(
                    data=str(data_yaml),
                    split=split,
                    imgsz=int(args.imgsz),
                    batch=int(args.batch),
                    conf=float(args.conf),
                    iou=float(args.iou),
                    device=str(args.device),
                    plots=False,
                    verbose=False,
                )
                row = {
                    "dataset": spec.dataset,
                    "run": run,
                    "variant": spec.variant,
                    "seed": seed,
                    "split": split,
                    "n_images": n_images,
                    "n_instances": n_instances,
                    "precision": f"{float(result.box.mp):.6f}",
                    "recall": f"{float(result.box.mr):.6f}",
                    "mAP50": f"{float(result.box.map50):.6f}",
                    "mAP50_95": f"{float(result.box.map):.6f}",
                }
                rows.append(row)
                print(
                    f"[ok] {spec.dataset:13s} {run:28s} {split:4s} "
                    f"images={n_images} instances={n_instances} mAP50={row['mAP50']}"
                )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "dataset",
                "run",
                "variant",
                "seed",
                "split",
                "n_images",
                "n_instances",
                "precision",
                "recall",
                "mAP50",
                "mAP50_95",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote: {args.output_csv}")


def selected_specs(args: argparse.Namespace) -> list[VariantSpec]:
    variants = set(args.variants or [])
    specs = [spec for spec in SPECS if args.dataset == "all" or spec.dataset == args.dataset]
    if variants:
        specs = [spec for spec in specs if spec.variant in variants]
    return specs


def prepare_eval_yamls(args: argparse.Namespace, specs: list[VariantSpec]) -> dict[tuple[str, str], Path]:
    canonical = {
        "vnwoodknot": args.vn_canonical_yaml.expanduser().resolve(),
        "vsb_rarefirst": args.vsb_canonical_yaml.expanduser().resolve(),
    }
    for dataset, path in canonical.items():
        if not path.exists():
            raise SystemExit(f"Missing canonical {dataset} dataset YAML: {path}")

    preprocessing_yamls: dict[tuple[str, str], Path] = {}
    for spec in specs:
        if not spec.preprocessing:
            continue
        key = (spec.dataset, spec.preprocessing)
        if key in preprocessing_yamls:
            continue
        preprocessing_yamls[key] = materialize_preprocessed_eval_dataset(
            source_yaml=canonical[spec.dataset],
            preprocessing=spec.preprocessing,
            output_root=args.eval_dataset_root.expanduser().resolve() / spec.dataset / spec.preprocessing,
            overwrite=bool(args.overwrite_eval_datasets),
        )

    eval_yamls: dict[tuple[str, str], Path] = {}
    for spec in specs:
        if spec.preprocessing:
            eval_yamls[(spec.dataset, spec.variant)] = preprocessing_yamls[(spec.dataset, spec.preprocessing)]
        else:
            eval_yamls[(spec.dataset, spec.variant)] = canonical[spec.dataset]
    return eval_yamls


def materialize_preprocessed_eval_dataset(*, source_yaml: Path, preprocessing: str, output_root: Path, overwrite: bool) -> Path:
    dataset_yaml = output_root / "dataset.yaml"
    if dataset_yaml.exists() and not overwrite:
        return dataset_yaml.resolve()
    command = [
        sys.executable,
        "scripts/materialize_preprocessed_yolo.py",
        "--source-yaml",
        str(source_yaml),
        "--variant-config",
        str(PROJECT_ROOT / "configs" / "preprocessing" / f"{preprocessing}.yaml"),
        "--output-root",
        str(output_root),
        "--image-format",
        "jpg",
        "--jpg-quality",
        "95",
    ]
    if overwrite:
        command.append("--overwrite")
    print("Materializing corrected preprocessing eval dataset:", " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"Preprocessing materialization failed for {preprocessing}: {output_root}")
    if not dataset_yaml.exists():
        raise SystemExit(f"Materialization did not create dataset YAML: {dataset_yaml}")
    return dataset_yaml.resolve()


def rebuild_vsb_canonical(args: argparse.Namespace) -> None:
    output_root = args.vsb_canonical_yaml.expanduser().resolve().parent
    command = [
        sys.executable,
        "scripts/materialize_yolo_from_manifest.py",
        "--manifest",
        str(args.vsb_manifest.expanduser()),
        "--images-root",
        str(args.vsb_images_root.expanduser()),
        "--output-root",
        str(output_root),
        "--dataset-name",
        output_root.name,
        "--classes",
        *VSB_CLASSES,
        "--split-strategy",
        "manifest",
        "--link-mode",
        "symlink",
    ]
    if args.overwrite_vsb_canonical:
        command.append("--overwrite")
    print("Rebuilding VSB canonical YOLO dataset:", " ".join(command))
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode != 0:
        raise SystemExit(f"VSB canonical materialization failed: {output_root}")


def write_eval_yaml_manifest(args: argparse.Namespace, eval_yamls: dict[tuple[str, str], Path]) -> None:
    manifest = args.output_csv.parent / "corrected_eval_dataset_map.csv"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["dataset", "variant", "data_yaml"])
        writer.writeheader()
        for (dataset, variant), data_yaml in sorted(eval_yamls.items()):
            writer.writerow({"dataset": dataset, "variant": variant, "data_yaml": str(data_yaml)})
    print(f"Wrote: {manifest}")


def count_split(data_yaml: Path, split: str) -> tuple[int, int]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
    dataset_root = resolve_dataset_root(data_yaml, data)
    n_images = 0
    n_instances = 0
    for split_dir in resolve_split_dirs(data_yaml, data, split):
        for image_path in sorted(path for path in split_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS):
            n_images += 1
            label_path = label_for_image(image_path, dataset_root=dataset_root, split=split, split_dir=split_dir)
            if label_path.exists():
                n_instances += sum(1 for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip())
    return n_images, n_instances


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


if __name__ == "__main__":
    main()
