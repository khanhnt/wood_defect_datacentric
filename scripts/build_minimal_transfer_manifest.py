#!/usr/bin/env python3
"""Build the minimal rebuilt-dataset file list required by the corrected generation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FULL_TREES = (
    "canonical/vnwoodknot",
    "variants/vnwoodknot/preprocessing/P1_CLAHE_luminance",
    "variants/vnwoodknot/preprocessing/P3_mild_unsharp",
    "variants/vnwoodknot/augmentation/seed42/A1_defect_preserving_crop",
    "variants/vnwoodknot/augmentation/seed43/A1_defect_preserving_crop",
    "variants/vnwoodknot/augmentation/seed44/A1_defect_preserving_crop",
    "variants/vnwoodknot/augmentation/seed42/A2_texture_aware_color_jitter",
    "variants/vnwoodknot/augmentation/seed43/A2_texture_aware_color_jitter",
    "variants/vnwoodknot/augmentation/seed44/A2_texture_aware_color_jitter",
    "variants/vnwoodknot/augmentation/seed42/P4_combined_safe__A4_combined_best",
    "variants/vnwoodknot/augmentation/seed43/P4_combined_safe__A4_combined_best",
    "variants/vnwoodknot/augmentation/seed44/P4_combined_safe__A4_combined_best",
    "canonical/vsb_rarefirst",
    "variants/vsb_rarefirst/augmentation/seed42/A1_defect_preserving_crop",
    "variants/vsb_rarefirst/augmentation/seed43/A1_defect_preserving_crop",
    "variants/vsb_rarefirst/augmentation/seed44/A1_defect_preserving_crop",
    "variants/vsb_rarefirst/augmentation/seed42/A2_texture_aware_color_jitter",
    "variants/vsb_rarefirst/augmentation/seed43/A2_texture_aware_color_jitter",
    "variants/vsb_rarefirst/augmentation/seed44/A2_texture_aware_color_jitter",
    "variants/vsb_rarefirst/augmentation/seed42/P4_combined_safe__A4_combined_best",
    "variants/vsb_rarefirst/augmentation/seed43/P4_combined_safe__A4_combined_best",
    "variants/vsb_rarefirst/augmentation/seed44/P4_combined_safe__A4_combined_best",
    "canonical/vsb_strict_clean",
    "variants/vsb_strict_clean/preprocessing/P1_CLAHE_luminance",
    "variants/vsb_strict_clean/preprocessing/P2_illumination_normalization",
    "variants/vsb_strict_clean/preprocessing/P3_mild_unsharp",
    "variants/vsb_strict_clean/preprocessing/P4_combined_safe",
)

EVAL_ONLY_TREES = (
    "variants/vnwoodknot/preprocessing/P2_illumination_normalization",
    "variants/vnwoodknot/preprocessing/P4_combined_safe",
    "variants/vsb_rarefirst/preprocessing/P1_CLAHE_luminance",
    "variants/vsb_rarefirst/preprocessing/P2_illumination_normalization",
    "variants/vsb_rarefirst/preprocessing/P3_mild_unsharp",
    "variants/vsb_rarefirst/preprocessing/P4_combined_safe",
)

METADATA_NAMES = {"dataset.yaml", "materialization_report.json"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rebuilt-root", type=Path, default=Path("revised/datasets_rebuilt"))
    parser.add_argument("--output-list", type=Path, default=Path("revised/minimal_transfer_files.txt"))
    parser.add_argument("--output-summary", type=Path, default=Path("revised/minimal_transfer_summary.csv"))
    return parser.parse_args()


def selected_files(root: Path, tree: str, eval_only: bool) -> list[Path]:
    tree_root = root / tree
    if not tree_root.is_dir():
        raise FileNotFoundError(f"Required rebuilt tree is missing: {tree_root}")

    files: list[Path] = []
    for path in tree_root.rglob("*"):
        if not path.is_file():
            continue
        if not eval_only:
            files.append(path)
            continue
        relative = path.relative_to(tree_root)
        if path.name in METADATA_NAMES or (
            len(relative.parts) >= 2
            and relative.parts[0] in {"images", "labels"}
            and relative.parts[1] in {"val", "test"}
        ):
            files.append(path)
    return files


def inode_key(path: Path) -> tuple[int, int]:
    stat = path.stat()
    return stat.st_dev, stat.st_ino


def unique_inode_bytes(paths: list[Path]) -> int:
    sizes: dict[tuple[int, int], int] = {}
    for path in paths:
        sizes.setdefault(inode_key(path), path.stat().st_size)
    return sum(sizes.values())


def main() -> None:
    args = parse_args()
    root = args.rebuilt_root.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Rebuilt root does not exist: {root}")

    selections: list[tuple[str, bool]] = [(tree, False) for tree in FULL_TREES]
    selections.extend((tree, True) for tree in EVAL_ONLY_TREES)
    all_files: dict[str, Path] = {}
    rows: list[dict[str, object]] = []

    for tree, eval_only in selections:
        paths = selected_files(root, tree, eval_only)
        for path in paths:
            all_files[path.relative_to(root).as_posix()] = path
        rows.append(
            {
                "tree": tree,
                "scope": "val_test_only" if eval_only else "full_tree",
                "files": len(paths),
                "logical_gib": f"{sum(path.stat().st_size for path in paths) / (1024**3):.3f}",
                "physical_gib_with_hardlinks": f"{unique_inode_bytes(paths) / (1024**3):.3f}",
            }
        )

    args.output_list.parent.mkdir(parents=True, exist_ok=True)
    args.output_list.write_text("".join(f"{path}\n" for path in sorted(all_files)), encoding="utf-8")

    args.output_summary.parent.mkdir(parents=True, exist_ok=True)
    with args.output_summary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("tree", "scope", "files", "logical_gib", "physical_gib_with_hardlinks"),
        )
        writer.writeheader()
        writer.writerows(rows)

    paths = list(all_files.values())
    print(f"Wrote file list: {args.output_list.resolve()}")
    print(f"Wrote summary: {args.output_summary.resolve()}")
    print(f"Selected files: {len(paths):,}")
    print(f"Logical size: {sum(path.stat().st_size for path in paths) / (1024**3):.2f} GiB")
    print(f"Physical size with hardlinks preserved: {unique_inode_bytes(paths) / (1024**3):.2f} GiB")


if __name__ == "__main__":
    main()
