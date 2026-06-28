#!/usr/bin/env python3
"""Materialize the VNWoodKnot two-class YOLO dataset from a release manifest."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data" / "vnwoodknot_split" / "manifest.jsonl")
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--link-mode", choices=("symlink", "hardlink", "copy"), default="symlink")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "materialize_yolo_from_manifest.py"),
        "--manifest",
        str(args.manifest),
        "--images-root",
        str(args.images_root),
        "--output-root",
        str(args.output_root),
        "--dataset-name",
        "vnwoodknot_live_dead_2class_yolo",
        "--classes",
        "live_knot",
        "dead_knot",
        "--split-strategy",
        "manifest",
        "--link-mode",
        args.link_mode,
    ]
    if args.overwrite:
        command.append("--overwrite")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


if __name__ == "__main__":
    main()
