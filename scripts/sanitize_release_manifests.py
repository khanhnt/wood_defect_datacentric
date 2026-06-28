#!/usr/bin/env python3
"""Rewrite tracked JSONL manifests with portable relative paths."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    sanitize_vnwoodknot(PROJECT_ROOT / "data" / "processed" / "vnwoodknot_manifest.jsonl")
    sanitize_vsb(PROJECT_ROOT / "data" / "processed" / "large_scale_wood_surface_defects_manifest.jsonl")


def sanitize_vnwoodknot(path: Path) -> None:
    rows = []
    for row in read_jsonl(path):
        image_id = str(row["image_id"])
        row["image_path"] = f"{image_id}.jpg"
        row["annotation_path"] = f"{image_id}.txt" if row.get("annotations") else None
        row["semantic_map_path"] = None
        rows.append(row)
    write_jsonl(path, rows)
    print(f"Sanitized: {path}")


def sanitize_vsb(path: Path) -> None:
    rows = []
    for row in read_jsonl(path):
        image_id = str(row["image_id"])
        stem = Path(image_id).name
        row["image_path"] = f"{image_id}.bmp"
        row["annotation_path"] = f"Bouding Boxes/{stem}_anno.txt"
        row["semantic_map_path"] = f"Semantic Maps/{stem}_segm.bmp"
        rows.append(row)
    write_jsonl(path, rows)
    print(f"Sanitized: {path}")


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
