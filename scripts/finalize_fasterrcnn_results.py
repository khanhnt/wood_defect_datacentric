#!/usr/bin/env python3
"""Audit and aggregate the completed nine-run Faster R-CNN generation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.fasterrcnn_prediction_adapter import validate_saved_export
from second_detector.protocol import build_jobs


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rebuilt-root", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--predictions-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    jobs = build_jobs(rebuilt_root=args.rebuilt_root, results_root=args.results_root)

    registry = []
    per_seed = []
    prediction_audit = []
    artifacts: list[Path] = []
    failures = []
    for job in jobs:
        required = {
            "best": job.output_dir / "weights" / "best.pt",
            "last": job.output_dir / "weights" / "last.pt",
            "results": job.output_dir / "results.csv",
            "config": job.output_dir / "config_used.yaml",
            "environment": job.output_dir / "environment.json",
            "summary": job.output_dir / "training_summary.json",
        }
        missing = [name for name, path in required.items() if not path.is_file() or path.stat().st_size == 0]
        status = "PASS" if not missing else "FAIL"
        row = {
            "job_id": job.job_id,
            "variant": job.variant,
            "seed": job.seed,
            "data_yaml": str(job.data_yaml),
            "output_dir": str(job.output_dir),
            "best_pt": str(required["best"]),
            "best_sha256": sha256(required["best"]) if required["best"].exists() else "",
            "last_pt": str(required["last"]),
            "last_sha256": sha256(required["last"]) if required["last"].exists() else "",
            "missing": ";".join(missing),
            "status": status,
        }
        registry.append(row)
        if missing:
            failures.append(row)
            continue
        artifacts.extend(required.values())
        for split, expected_images in (("val", 226), ("test", 229)):
            prediction_path = args.predictions_root / split / f"{job.variant}_seed{job.seed}_predictions.json"
            if not prediction_path.exists():
                failures.append({"job_id": job.job_id, "split": split, "reason": "missing_prediction_export"})
                continue
            artifacts.append(prediction_path)
            payload = json.loads(prediction_path.read_text(encoding="utf-8"))
            metrics = payload["validator_metrics"]
            per_seed.append(
                {
                    "dataset": "vnwoodknot",
                    "detector": "fasterrcnn_mobilenet_v3_large_fpn",
                    "variant": job.variant,
                    "seed": job.seed,
                    "split": split,
                    "n_images": metrics["n_images"],
                    "n_instances": metrics["n_instances"],
                    "precision": metrics["precision"],
                    "recall": metrics["recall"],
                    "mAP50": metrics["mAP50"],
                    "mAP50_95": metrics["mAP50_95"],
                }
            )
            audit = validate_saved_export(prediction_path)
            audit.update({"job_id": job.job_id, "variant": job.variant, "seed": job.seed, "split": split})
            if metrics["n_images"] != expected_images:
                audit["status"] = "FAIL"
                audit["image_count_error"] = f"expected {expected_images}, observed {metrics['n_images']}"
            prediction_audit.append(audit)
            if audit["status"] != "PASS":
                failures.append(audit)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "checkpoint_registry.csv", registry)
    if per_seed:
        write_csv(args.output_dir / "per_seed_metrics.csv", per_seed)
    if prediction_audit:
        compact = [
            {key: row.get(key) for key in ("job_id", "variant", "seed", "split", "num_images", "num_predictions", "prediction_mask_exact_rate", "tp50_exact_rate", "status")}
            for row in prediction_audit
        ]
        write_csv(args.output_dir / "prediction_audit.csv", compact)

    summary_rows = []
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in per_seed:
        grouped[(row["variant"], row["split"])].append(row)
    for (variant, split), rows in sorted(grouped.items()):
        summary = {"dataset": "vnwoodknot", "detector": "fasterrcnn_mobilenet_v3_large_fpn", "variant": variant, "split": split, "n_seeds": len(rows)}
        for metric in ("precision", "recall", "mAP50", "mAP50_95"):
            values = np.asarray([float(row[metric]) for row in rows])
            summary[f"{metric}_mean"] = float(values.mean())
            summary[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        summary_rows.append(summary)
    if summary_rows:
        write_csv(args.output_dir / "summary.csv", summary_rows)

    checksum_lines = []
    for artifact in sorted(set(path.resolve() for path in artifacts)):
        checksum_lines.append(f"{sha256(artifact)}  {artifact}")
    (args.output_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    report = {
        "status": "PASS" if not failures and len(registry) == 9 and len(per_seed) == 18 else "FAIL",
        "checkpoint_runs": len(registry),
        "metric_rows": len(per_seed),
        "prediction_exports": len(prediction_audit),
        "failures": failures,
    }
    (args.output_dir / "finalization_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
