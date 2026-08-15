#!/usr/bin/env python3
"""Create source-disjoint VSB clean validation/test views for every evaluation variant."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import random
import shutil

import yaml


VARIANT_ROOTS = {
    "baseline": ("canonical", "vsb_strict_clean"),
    "p1_clahe": ("variants", "vsb_strict_clean", "preprocessing", "P1_CLAHE_luminance"),
    "p2_illumination": ("variants", "vsb_strict_clean", "preprocessing", "P2_illumination_normalization"),
    "p3_unsharp": ("variants", "vsb_strict_clean", "preprocessing", "P3_mild_unsharp"),
    "a1_crop": ("canonical", "vsb_strict_clean"),
    "a2_colorjitter": ("canonical", "vsb_strict_clean"),
    "p4_a4_combined": ("variants", "vsb_strict_clean", "preprocessing", "P4_combined_safe"),
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
CLASS_NAMES = ["live_knot", "dead_knot", "resin", "knot_with_crack", "crack", "marrow", "knot_missing"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuilt-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selection-sources", type=int, default=996)
    parser.add_argument("--mode", choices=("hardlink", "copy"), default="hardlink")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def place_file(source: Path, target: Path, mode: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode == "hardlink":
        os.link(source, target)
    else:
        shutil.copy2(source, target)


def image_index(root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    duplicates: set[str] = set()
    for path in sorted((root / "images" / "test").rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        if path.stem in index:
            duplicates.add(path.stem)
        index[path.stem] = path
    if duplicates:
        raise SystemExit(f"Ambiguous clean tile stems under {root}: {sorted(duplicates)[:10]}")
    return index


def source_id_from_tile(stem: str) -> str:
    marker = "__x"
    if marker not in stem:
        raise ValueError(f"Clean tile does not encode source ID: {stem}")
    return stem.split(marker, 1)[0]


def partition_source_ids(
    source_ids: list[str],
    *,
    seed: int,
    selection_sources: int,
) -> tuple[set[str], set[str], list[str]]:
    unique_ids = sorted(set(source_ids))
    if selection_sources <= 0 or selection_sources >= len(unique_ids):
        raise ValueError("selection_sources must leave nonempty selection and final-test partitions")
    shuffled = list(unique_ids)
    random.Random(seed).shuffle(shuffled)
    selection = set(shuffled[:selection_sources])
    final_test = set(shuffled[selection_sources:])
    if selection & final_test or selection | final_test != set(unique_ids):
        raise RuntimeError("Source-level partition is not disjoint and exhaustive")
    return selection, final_test, shuffled


def dataset_yaml(root: Path) -> None:
    payload = {
        "path": str(root.resolve()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(CLASS_NAMES)},
    }
    (root / "dataset.yaml").write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    rebuilt_root = args.rebuilt_root.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    samples_path = rebuilt_root / "canonical" / "vsb_strict_clean" / "clean_materialized_samples.csv"
    if not samples_path.exists():
        raise SystemExit(f"Missing strict-clean materialization records: {samples_path}")
    samples = read_rows(samples_path)
    source_ids = sorted({row["source_id"] for row in samples})
    if len(source_ids) != 1992:
        raise SystemExit(f"Expected 1,992 clean source IDs, found {len(source_ids)}")
    tile_counts = {source_id: 0 for source_id in source_ids}
    for row in samples:
        tile_counts[row["source_id"]] += 1
    unexpected = {source_id: count for source_id, count in tile_counts.items() if count != 3}
    if unexpected:
        raise SystemExit(f"Expected exactly three clean tiles per source: {list(unexpected.items())[:10]}")
    try:
        selection, final_test, shuffled = partition_source_ids(
            source_ids,
            seed=args.seed,
            selection_sources=args.selection_sources,
        )
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(str(exc)) from exc

    if output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output exists: {output_root}. Use --overwrite intentionally.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)

    rank = {source_id: index for index, source_id in enumerate(shuffled)}
    source_rows = [
        {
            "source_id": source_id,
            "partition": "threshold_selection" if source_id in selection else "final_test",
            "yolo_split": "val" if source_id in selection else "test",
            "seed": args.seed,
            "shuffle_rank": rank[source_id],
        }
        for source_id in sorted(source_ids)
    ]
    write_rows(output_root / "source_partition_manifest.csv", source_rows)

    tile_rows: list[dict[str, object]] = []
    map_rows: list[dict[str, object]] = []
    variant_reports: dict[str, dict[str, int]] = {}
    for variant, relative_parts in VARIANT_ROOTS.items():
        source_root = rebuilt_root.joinpath(*relative_parts)
        source_yaml = source_root / "dataset.yaml"
        if not source_yaml.exists():
            raise SystemExit(f"Missing strict-clean variant dataset: {source_yaml}")
        images = image_index(source_root)
        if len(images) != 5976:
            raise SystemExit(f"Expected 5,976 clean tiles for {variant}, found {len(images)}")
        variant_root = output_root / variant
        counts = {"val": 0, "test": 0}
        for stem, source_image in sorted(images.items()):
            source_id = source_id_from_tile(stem)
            split = "val" if source_id in selection else "test"
            relative = source_image.relative_to(source_root / "images" / "test")
            source_label = (source_root / "labels" / "test" / relative).with_suffix(".txt")
            if not source_label.exists() or source_label.read_text(encoding="utf-8").strip():
                raise SystemExit(f"Strict-clean label is missing or nonempty: {source_label}")
            target_image = variant_root / "images" / split / relative
            target_label = (variant_root / "labels" / split / relative).with_suffix(".txt")
            place_file(source_image, target_image, args.mode)
            place_file(source_label, target_label, args.mode)
            counts[split] += 1
            tile_rows.append(
                {
                    "variant": variant,
                    "source_id": source_id,
                    "tile_id": stem,
                    "partition": "threshold_selection" if split == "val" else "final_test",
                    "yolo_split": split,
                    "source_image": str(source_image),
                    "view_image": str(target_image),
                }
            )
        for split in ("train", "val", "test"):
            (variant_root / "images" / split).mkdir(parents=True, exist_ok=True)
            (variant_root / "labels" / split).mkdir(parents=True, exist_ok=True)
        dataset_yaml(variant_root)
        variant_reports[variant] = counts
        map_rows.append({"dataset": "vsb_strict_clean", "variant": variant, "data_yaml": str(variant_root / "dataset.yaml")})

    write_rows(output_root / "tile_partition_manifest.csv", tile_rows)
    write_rows(output_root / "eval_dataset_map.csv", map_rows)
    report = {
        "status": "PASS",
        "seed": args.seed,
        "ratio": "996/996 (50/50 by source ID)",
        "selection_source_count": len(selection),
        "final_test_source_count": len(final_test),
        "source_overlap": len(selection & final_test),
        "tiles_per_source": 3,
        "selection_tiles_per_variant": args.selection_sources * 3,
        "final_test_tiles_per_variant": len(final_test) * 3,
        "variant_counts": variant_reports,
        "mode": args.mode,
        "reason": "Equal source-level halves balance threshold-selection precision and final-test power while keeping all tiles from one board in one partition.",
    }
    (output_root / "partition_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote: {output_root / 'source_partition_manifest.csv'}")
    print(f"Wrote: {output_root / 'tile_partition_manifest.csv'}")
    print("VSB CLEAN SOURCE SPLIT: PASS")


if __name__ == "__main__":
    main()
