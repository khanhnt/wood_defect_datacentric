#!/usr/bin/env python3
"""Print corrected-24 progress and an estimated remaining wall time."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


EXPECTED = {"vnwoodknot": 15, "vsb_rarefirst": 9}
FALLBACK_MINUTES = {"vnwoodknot": 13.44, "vsb_rarefirst": 82.83}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--gpus", type=int, default=2)
    args = parser.parse_args()
    path = args.generation_root.expanduser().resolve() / "gpu_optimization" / "run_log.csv"
    if not path.exists():
        raise SystemExit(f"Missing run log: {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    latest = {}
    for row in rows:
        latest[row["job_id"]] = row
    completed = [row for row in latest.values() if row["status"] in {"ok", "skipped_completed"}]
    remaining_gpu_minutes = 0.0
    print(f"Completed: {len(completed)}/{sum(EXPECTED.values())}")
    for dataset, expected in EXPECTED.items():
        done = [row for row in completed if row["dataset"] == dataset]
        measured = [float(row["duration_min"]) for row in done if row["status"] == "ok" and float(row["duration_min"] or 0) > 0]
        estimate = sum(measured) / len(measured) if measured else FALLBACK_MINUTES[dataset]
        left = expected - len(done)
        remaining_gpu_minutes += left * estimate
        print(f"- {dataset}: {len(done)}/{expected}; estimated {estimate:.1f} min/job")
    print(f"Estimated remaining wall time on {args.gpus} GPUs: {remaining_gpu_minutes / max(args.gpus, 1) / 60:.2f} h")
    failed = [row for row in latest.values() if row["status"] not in {"ok", "skipped_completed"}]
    if failed:
        print("Latest non-success attempts:")
        for row in sorted(failed, key=lambda item: item["job_id"]):
            print(f"- {row['job_id']}: {row['status']} ({row['error']})")


if __name__ == "__main__":
    main()
