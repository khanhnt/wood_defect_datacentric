#!/usr/bin/env python3
"""Smoke test for the independent data-centric project.

This script does not train models or access full datasets. It checks the repo
layout and import path. Optional CV-heavy modules can be enforced with
``--strict-optional`` after installing ``requirements.txt``.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PACKAGE_PARENT = PROJECT_ROOT.parent

if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))


EXPECTED_DIRS = [
    "configs",
    "datasets",
    "preprocessing",
    "augmentation",
    "training",
    "evaluation",
    "visualization",
    "scripts",
    "results",
    "docs",
    "data/processed",
]

CORE_MODULES = [
    "wood_defect_datacentric",
    "wood_defect_datacentric.datasets",
    "wood_defect_datacentric.training",
    "wood_defect_datacentric.visualization",
]

OPTIONAL_MODULES = [
    "wood_defect_datacentric.preprocessing",
    "wood_defect_datacentric.augmentation",
    "wood_defect_datacentric.evaluation",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict-optional",
        action="store_true",
        help="Fail if optional CV/evaluation dependencies such as OpenCV are missing.",
    )
    return parser.parse_args()


def import_modules(module_names: list[str]) -> list[tuple[str, str]]:
    failures: list[tuple[str, str]] = []
    for module_name in module_names:
        try:
            importlib.import_module(module_name)
        except ModuleNotFoundError as exc:
            failures.append((module_name, str(exc)))
    return failures


def main() -> None:
    args = parse_args()

    missing_dirs = [dirname for dirname in EXPECTED_DIRS if not (PROJECT_ROOT / dirname).exists()]
    if missing_dirs:
        raise SystemExit(f"Missing expected directories: {missing_dirs}")

    config_path = PROJECT_ROOT / "configs" / "project.yaml"
    if not config_path.exists():
        raise SystemExit(f"Missing config template: {config_path}")

    core_failures = import_modules(CORE_MODULES)
    if core_failures:
        details = "; ".join(f"{module}: {error}" for module, error in core_failures)
        raise SystemExit(f"Core import check failed: {details}")

    optional_failures = import_modules(OPTIONAL_MODULES)
    if optional_failures and args.strict_optional:
        details = "; ".join(f"{module}: {error}" for module, error in optional_failures)
        raise SystemExit(f"Optional import check failed: {details}")

    print("wood_defect_datacentric smoke test passed")
    print(f"project_root={PROJECT_ROOT}")
    print(f"config={config_path}")
    if optional_failures:
        print("optional_dependency_warnings:")
        for module, error in optional_failures:
            print(f"- {module}: {error}")
        print("Install requirements.txt and rerun with --strict-optional on the training server.")


if __name__ == "__main__":
    main()
