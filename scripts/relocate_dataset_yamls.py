#!/usr/bin/env python3
"""Relocate copied dataset YAML files to their current parent directories."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    paths = sorted(root.rglob("dataset.yaml"))
    if not paths:
        raise SystemExit(f"No dataset.yaml files under {root}")
    for path in paths:
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        config["path"] = str(path.parent.resolve())
        path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"Relocated {len(paths)} dataset YAML files under {root}")


if __name__ == "__main__":
    main()
