#!/usr/bin/env python3
"""Verify the pinned training runtime and record it as JSON."""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import subprocess


EXPECTED_PYTHON_PREFIX = "3.12."
MINIMUM_DRIVER = (550, 54)
EXPECTED = {
    "torch": "2.6.0",
    "torchvision": "0.21.0",
    "ultralytics": "8.4.60",
    "numpy": "2.1.1",
    "pandas": "2.2.3",
    "opencv": "4.10.0",
    "Pillow": "11.0.0",
    "PyYAML": "6.0.2",
    "matplotlib": "3.10.0",
}


def base_version(value: str) -> str:
    return value.split("+", 1)[0]


def nvidia_driver_version() -> str:
    completed = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.splitlines()[0].strip() if completed.returncode == 0 and completed.stdout.strip() else "unavailable"


def version_tuple(value: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in value.split("."))
    except ValueError:
        return ()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-gpus", type=int, default=2)
    args = parser.parse_args()

    import cv2
    import matplotlib
    import numpy
    import pandas
    import PIL
    import torch
    import torchvision
    import ultralytics
    import yaml

    observed = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "ultralytics": ultralytics.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "opencv": cv2.__version__,
        "Pillow": PIL.__version__,
        "PyYAML": yaml.__version__,
        "matplotlib": matplotlib.__version__,
    }
    mismatches = {
        name: {"expected": expected, "observed": observed[name]}
        for name, expected in EXPECTED.items()
        if base_version(observed[name]) != expected
    }
    python_version = platform.python_version()
    if not python_version.startswith(EXPECTED_PYTHON_PREFIX):
        mismatches["python"] = {"expected": "3.12.x", "observed": python_version}
    gpu_count = torch.cuda.device_count()
    if not torch.cuda.is_available():
        mismatches["cuda_available"] = {"expected": True, "observed": False}
    if gpu_count != args.expected_gpus:
        mismatches["gpu_count"] = {"expected": args.expected_gpus, "observed": gpu_count}
    driver_version = nvidia_driver_version()
    if version_tuple(driver_version) < MINIMUM_DRIVER:
        mismatches["nvidia_driver"] = {"expected": ">=550.54", "observed": driver_version}
    completed = subprocess.run(["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False)
    report = {
        "status": "PASS" if not mismatches else "FAIL",
        "python": python_version,
        "packages": observed,
        "package_paths": {
            "ultralytics": str(Path(ultralytics.__file__).resolve()),
            "torch": str(Path(torch.__file__).resolve()),
        },
        "torch_cuda": torch.version.cuda,
        "nvidia_driver": driver_version,
        "cuda_available": torch.cuda.is_available(),
        "gpu_count": gpu_count,
        "gpus": [torch.cuda.get_device_name(index) for index in range(gpu_count)],
        "git_commit": completed.stdout.strip() if completed.returncode == 0 else "unavailable",
        "mismatches": mismatches,
    }
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote: {output}")
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
