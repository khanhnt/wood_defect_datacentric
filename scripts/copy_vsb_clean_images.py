#!/usr/bin/env python3
"""Find VSB clean-source images by source ID and copy them to a target folder."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_IDS_FILE = Path("configs/datasets/vsb_clean_source_ids.txt")
DEFAULT_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


@dataclass
class FoundImage:
    source_id: str
    source_path: str
    output_path: str
    copied: bool
    reason: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Search a raw image root for VSB clean-source IDs and copy matching images "
            "to an output directory."
        )
    )
    parser.add_argument("source", type=Path, help="Root directory to search recursively.")
    parser.add_argument("output", type=Path, help="Directory where matched images are copied.")
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=DEFAULT_IDS_FILE,
        help=f"Text file with one clean source ID per line. Default: {DEFAULT_IDS_FILE}",
    )
    parser.add_argument(
        "--extensions",
        default=",".join(DEFAULT_EXTENSIONS),
        help="Comma-separated image extensions to search.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report matches without copying.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite files already in output.")
    parser.add_argument(
        "--preserve-relative",
        action="store_true",
        help="Preserve the relative directory layout under the output directory.",
    )
    return parser.parse_args()


def load_ids(path: Path) -> set[str]:
    if not path.exists():
        raise FileNotFoundError(f"Clean source ID file not found: {path}")
    ids = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    if not ids:
        raise ValueError(f"No clean source IDs found in: {path}")
    return ids


def source_id_from_image(path: Path, clean_ids: set[str]) -> str | None:
    stem = path.stem
    if stem in clean_ids:
        return stem
    prefix = stem.split("__", 1)[0]
    if prefix in clean_ids:
        return prefix
    return None


def unique_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    for idx in range(1, 10000):
        candidate = path.with_name(f"{path.stem}_dup{idx}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not create a unique output path for {path}")


def iter_image_files(root: Path, extensions: set[str]):
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in extensions:
            yield path


def main() -> None:
    args = parse_args()
    source_root = args.source.expanduser().resolve()
    output_root = args.output.expanduser().resolve()
    ids_file = args.ids_file.expanduser().resolve()
    extensions = {ext.strip().lower() for ext in args.extensions.split(",") if ext.strip()}
    extensions = {ext if ext.startswith(".") else f".{ext}" for ext in extensions}

    if not source_root.exists():
        raise FileNotFoundError(f"Source directory not found: {source_root}")
    if not source_root.is_dir():
        raise NotADirectoryError(f"Source must be a directory: {source_root}")

    clean_ids = load_ids(ids_file)
    output_root.mkdir(parents=True, exist_ok=True)

    found: list[FoundImage] = []
    found_ids: set[str] = set()
    scanned = 0

    for image_path in iter_image_files(source_root, extensions):
        scanned += 1
        source_id = source_id_from_image(image_path, clean_ids)
        if source_id is None:
            continue

        found_ids.add(source_id)
        if args.preserve_relative:
            rel = image_path.relative_to(source_root)
            dest = output_root / rel
        else:
            dest = output_root / image_path.name

        copied = False
        reason = "dry_run"
        if not args.dry_run:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists() and not args.overwrite:
                dest = unique_output_path(dest)
            shutil.copy2(image_path, dest)
            copied = True
            reason = "copied"

        found.append(
            FoundImage(
                source_id=source_id,
                source_path=str(image_path),
                output_path=str(dest),
                copied=copied,
                reason=reason,
            )
        )

    missing = sorted(clean_ids - found_ids)
    found_csv = output_root / "found_clean_source_images.csv"
    missing_txt = output_root / "missing_clean_source_ids.txt"
    report_json = output_root / "copy_report.json"

    with found_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(FoundImage.__dataclass_fields__))
        writer.writeheader()
        for row in found:
            writer.writerow(asdict(row))

    missing_txt.write_text("\n".join(missing) + ("\n" if missing else ""))
    report = {
        "source_root": str(source_root),
        "output_root": str(output_root),
        "ids_file": str(ids_file),
        "extensions": sorted(extensions),
        "dry_run": args.dry_run,
        "overwrite": args.overwrite,
        "preserve_relative": args.preserve_relative,
        "num_clean_ids": len(clean_ids),
        "num_image_files_scanned": scanned,
        "num_matched_files": len(found),
        "num_matched_source_ids": len(found_ids),
        "num_missing_source_ids": len(missing),
        "found_csv": str(found_csv),
        "missing_txt": str(missing_txt),
    }
    report_json.write_text(json.dumps(report, indent=2) + "\n")

    print("VSB clean image copy summary")
    print(f"- source: {source_root}")
    print(f"- output: {output_root}")
    print(f"- clean IDs: {len(clean_ids)}")
    print(f"- image files scanned: {scanned}")
    print(f"- matched files: {len(found)}")
    print(f"- matched source IDs: {len(found_ids)}")
    print(f"- missing source IDs: {len(missing)}")
    print(f"- found CSV: {found_csv}")
    print(f"- missing IDs: {missing_txt}")
    print(f"- report JSON: {report_json}")


if __name__ == "__main__":
    main()
