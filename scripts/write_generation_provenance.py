#!/usr/bin/env python3
"""Write immutable provenance records for one evaluation/prediction generation."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-root", type=Path, required=True)
    parser.add_argument("--fair-summary", type=Path, required=True)
    parser.add_argument("--checkpoint-registry", type=Path, required=True)
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--deprecated-checkpoint-registry", type=Path)
    parser.add_argument("--deprecated-fair-summary", type=Path)
    parser.add_argument("--deprecated-prediction-root", type=Path)
    parser.add_argument("--pretrained-weights", type=Path, default=Path("yolov8s.pt"))
    parser.add_argument("--vn-manifest", type=Path, default=Path("data/vnwoodknot_split/manifest.jsonl"))
    parser.add_argument("--vsb-manifest", type=Path, default=Path("data/vsb_rarefirst_split/manifest.jsonl"))
    parser.add_argument("--vsb-clean-manifest", type=Path, default=Path("data/vsb_clean_manifest/clean_tile_manifest.csv"))
    parser.add_argument("--extra-manifest", action="append", type=Path, default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def copy_yaml(source: Path, output_root: Path, dataset: str, variant: str) -> tuple[Path, str]:
    target = output_root / "dataset_yamls" / dataset / variant / "dataset.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and sha256(target) != sha256(source):
        raise SystemExit(f"Conflicting evaluation YAML for {dataset}:{variant}: {source}")
    if not target.exists():
        shutil.copy2(source, target)
    return target, sha256(target)


def environment() -> dict[str, str]:
    versions = {}
    try:
        import torch

        versions["torch_version"] = str(torch.__version__)
        versions["torch_cuda_version"] = str(torch.version.cuda)
    except Exception:
        versions["torch_version"] = "unavailable"
        versions["torch_cuda_version"] = "unavailable"
    try:
        import ultralytics

        versions["ultralytics_version"] = str(ultralytics.__version__)
    except Exception:
        versions["ultralytics_version"] = "unavailable"
    return versions


def git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        text=True,
        capture_output=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unavailable"


def main() -> None:
    args = parse_args()
    generation_root = args.generation_root.expanduser().resolve()
    output_root = generation_root / "provenance"
    output_root.mkdir(parents=True, exist_ok=True)
    output_csv = output_root / "artifact_provenance.csv"
    output_json = output_root / "artifact_provenance.json"
    if (output_csv.exists() or output_json.exists()) and not args.overwrite:
        raise SystemExit("Provenance output exists; use --overwrite intentionally.")

    checkpoint_rows = read_csv(args.checkpoint_registry.expanduser().resolve())
    checkpoints = {
        (row["dataset"], row["variant"], int(row["seed"])): row
        for row in checkpoint_rows
        if row["status"] == "PASS"
    }
    deprecated_args = (
        args.deprecated_checkpoint_registry,
        args.deprecated_fair_summary,
        args.deprecated_prediction_root,
    )
    if any(deprecated_args) and not all(deprecated_args):
        raise SystemExit("Deprecated provenance requires registry, fair summary, and prediction root together.")
    deprecated_rows = read_csv(args.deprecated_checkpoint_registry.expanduser().resolve()) if all(deprecated_args) else []
    deprecated_checkpoints = {
        (row["dataset"], row["variant"], int(row["seed"])): row
        for row in deprecated_rows
        if row["status"] == "PASS"
    }
    manifest_paths = {
        "vnwoodknot": args.vn_manifest.expanduser().resolve(),
        "vsb_rarefirst": args.vsb_manifest.expanduser().resolve(),
        "vsb_strict_clean": args.vsb_clean_manifest.expanduser().resolve(),
    }
    for path in manifest_paths.values():
        if not path.exists():
            raise SystemExit(f"Missing authoritative manifest: {path}")
    manifest_hashes = {dataset: sha256(path) for dataset, path in manifest_paths.items()}
    manifest_copies: dict[str, Path] = {}
    manifest_inventory: list[dict[str, str]] = []
    for dataset, source in manifest_paths.items():
        target = output_root / "manifests" / f"{dataset}{source.suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest_copies[dataset] = target
        manifest_inventory.append({"role": dataset, "source": str(source), "copy": str(target), "sha256": sha256(target)})
    for source_arg in args.extra_manifest:
        source = source_arg.expanduser().resolve()
        if not source.exists():
            raise SystemExit(f"Missing extra manifest: {source}")
        target = output_root / "manifests" / "extra" / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() and sha256(target) != sha256(source):
            raise SystemExit(f"Conflicting extra manifest basename: {source}")
        shutil.copy2(source, target)
        manifest_inventory.append({"role": "extra", "source": str(source), "copy": str(target), "sha256": sha256(target)})
    (output_root / "manifest_inventory.json").write_text(json.dumps(manifest_inventory, indent=2) + "\n", encoding="utf-8")
    env = environment()
    commit = git_commit()
    timestamp = datetime.now(timezone.utc).isoformat()
    pretrained = args.pretrained_weights.expanduser().resolve()
    if not pretrained.exists():
        raise SystemExit(f"Missing pretrained weights used by new runs: {pretrained}")
    env["pretrained_weights"] = str(pretrained)
    env["pretrained_sha256"] = sha256(pretrained)
    records: list[dict[str, object]] = []

    fair_summary = args.fair_summary.expanduser().resolve()
    for row in read_csv(fair_summary):
        key = (row["dataset"], row["variant"], int(row["seed"]))
        checkpoint = checkpoints.get(key)
        if not checkpoint:
            raise SystemExit(f"Fair result has no registered checkpoint: {key}")
        yaml_source = Path(row.get("data_yaml", ""))
        if not yaml_source.exists():
            map_path = fair_summary.parent / "corrected_eval_dataset_map.csv"
            mapping = {(x["dataset"], x["variant"]): Path(x["data_yaml"]) for x in read_csv(map_path)}
            yaml_source = mapping[(row["dataset"], row["variant"])]
        yaml_copy, yaml_hash = copy_yaml(yaml_source.resolve(), output_root, row["dataset"], row["variant"])
        records.append(
            {
                "artifact_type": "fair_metric_row",
                "artifact_path": str(fair_summary),
                "artifact_sha256": sha256(fair_summary),
                "dataset": row["dataset"],
                "variant": row["variant"],
                "seed": int(row["seed"]),
                "split": row["split"],
                "checkpoint_path": checkpoint["best_path"],
                "checkpoint_sha256": checkpoint["best_sha256"],
                "checkpoint_role": checkpoint["source_type"],
                "manifest_path": str(manifest_paths[row["dataset"]]),
                "manifest_sha256": manifest_hashes[row["dataset"]],
                "manifest_copy": str(manifest_copies[row["dataset"]]),
                "evaluation_yaml_source": str(yaml_source.resolve()),
                "evaluation_yaml_copy": str(yaml_copy),
                "evaluation_yaml_sha256": yaml_hash,
                "prediction_sha256": "",
            }
        )

    for prediction in sorted(args.prediction_root.expanduser().resolve().rglob("*_predictions.json")):
        payload = json.loads(prediction.read_text(encoding="utf-8"))
        dataset = str(payload["dataset"])
        variant = str(payload["variant"])
        seed = int(payload["seed"])
        checkpoint = checkpoints.get(("vsb_rarefirst" if dataset == "vsb_strict_clean" else dataset, variant, seed))
        if not checkpoint:
            raise SystemExit(f"Prediction has no registered checkpoint: {prediction}")
        yaml_source = Path(str(payload["dataset_yaml"])).expanduser().resolve()
        if payload.get("checkpoint_sha256") != checkpoint["best_sha256"]:
            raise SystemExit(f"Prediction checkpoint hash differs from registry: {prediction}")
        if payload.get("dataset_yaml_sha256") != sha256(yaml_source):
            raise SystemExit(f"Prediction dataset YAML hash differs from current YAML: {prediction}")
        yaml_copy, yaml_hash = copy_yaml(yaml_source, output_root, dataset, variant)
        prediction_hash = sha256(prediction)
        records.append(
            {
                "artifact_type": "prediction_export",
                "artifact_path": str(prediction),
                "artifact_sha256": prediction_hash,
                "dataset": dataset,
                "variant": variant,
                "seed": seed,
                "split": str(payload["split"]),
                "checkpoint_path": checkpoint["best_path"],
                "checkpoint_sha256": checkpoint["best_sha256"],
                "checkpoint_role": checkpoint["source_type"],
                "manifest_path": str(manifest_paths[dataset]),
                "manifest_sha256": manifest_hashes[dataset],
                "manifest_copy": str(manifest_copies[dataset]),
                "evaluation_yaml_source": str(yaml_source),
                "evaluation_yaml_copy": str(yaml_copy),
                "evaluation_yaml_sha256": yaml_hash,
                "prediction_sha256": prediction_hash,
            }
        )

    if deprecated_checkpoints:
        deprecated_fair = args.deprecated_fair_summary.expanduser().resolve()
        for row in read_csv(deprecated_fair):
            key = (row["dataset"], row["variant"], int(row["seed"]))
            checkpoint = deprecated_checkpoints.get(key)
            if not checkpoint:
                raise SystemExit(f"Deprecated fair result has no registered checkpoint: {key}")
            yaml_source = Path(row["data_yaml"]).expanduser().resolve()
            yaml_copy, yaml_hash = copy_yaml(yaml_source, output_root, row["dataset"], row["variant"])
            records.append(
                {
                    "artifact_type": "DEPRECATED_fair_metric_row",
                    "artifact_path": str(deprecated_fair),
                    "artifact_sha256": sha256(deprecated_fair),
                    "dataset": row["dataset"],
                    "variant": row["variant"],
                    "seed": int(row["seed"]),
                    "split": row["split"],
                    "checkpoint_path": checkpoint["best_path"],
                    "checkpoint_sha256": checkpoint["best_sha256"],
                    "checkpoint_role": checkpoint["checkpoint_role"],
                    "manifest_path": str(manifest_paths[row["dataset"]]),
                    "manifest_sha256": manifest_hashes[row["dataset"]],
                    "manifest_copy": str(manifest_copies[row["dataset"]]),
                    "evaluation_yaml_source": str(yaml_source),
                    "evaluation_yaml_copy": str(yaml_copy),
                    "evaluation_yaml_sha256": yaml_hash,
                    "prediction_sha256": "",
                }
            )

        for prediction in sorted(args.deprecated_prediction_root.expanduser().resolve().rglob("*_predictions.json")):
            payload = json.loads(prediction.read_text(encoding="utf-8"))
            dataset = str(payload["dataset"])
            variant = str(payload["variant"])
            seed = int(payload["seed"])
            checkpoint = deprecated_checkpoints.get((dataset, variant, seed))
            if not checkpoint:
                raise SystemExit(f"Deprecated prediction has no registered checkpoint: {prediction}")
            yaml_source = Path(str(payload["dataset_yaml"])).expanduser().resolve()
            if payload.get("checkpoint_sha256") != checkpoint["best_sha256"]:
                raise SystemExit(f"Deprecated prediction checkpoint hash differs from registry: {prediction}")
            if payload.get("dataset_yaml_sha256") != sha256(yaml_source):
                raise SystemExit(f"Deprecated prediction dataset YAML hash differs from current YAML: {prediction}")
            yaml_copy, yaml_hash = copy_yaml(yaml_source, output_root, dataset, variant)
            prediction_hash = sha256(prediction)
            records.append(
                {
                    "artifact_type": "DEPRECATED_prediction_export",
                    "artifact_path": str(prediction),
                    "artifact_sha256": prediction_hash,
                    "dataset": dataset,
                    "variant": variant,
                    "seed": seed,
                    "split": str(payload["split"]),
                    "checkpoint_path": checkpoint["best_path"],
                    "checkpoint_sha256": checkpoint["best_sha256"],
                    "checkpoint_role": checkpoint["checkpoint_role"],
                    "manifest_path": str(manifest_paths[dataset]),
                    "manifest_sha256": manifest_hashes[dataset],
                    "manifest_copy": str(manifest_copies[dataset]),
                    "evaluation_yaml_source": str(yaml_source),
                    "evaluation_yaml_copy": str(yaml_copy),
                    "evaluation_yaml_sha256": yaml_hash,
                    "prediction_sha256": prediction_hash,
                }
            )

    common = {"git_commit": commit, "timestamp_utc": timestamp, **env}
    records = [{**record, **common} for record in records]
    if not records:
        raise SystemExit("No provenance artifacts found.")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    output_json.write_text(json.dumps({"environment": common, "artifacts": records}, indent=2) + "\n", encoding="utf-8")
    (output_root / "environment.yaml").write_text(yaml.safe_dump(common, sort_keys=False), encoding="utf-8")
    checksum_paths = {fair_summary, args.checkpoint_registry.expanduser().resolve()}
    checksum_paths.update(args.prediction_root.expanduser().resolve().rglob("*_predictions.json"))
    checksum_paths.update(output_root.glob("dataset_yamls/*/*/dataset.yaml"))
    checksum_paths.update(path for path in output_root.glob("manifests/**/*") if path.is_file())
    checksum_paths.add(output_root / "manifest_inventory.json")
    for row in checkpoint_rows:
        for key in ("best_path", "last_path"):
            candidate = Path(row[key])
            if candidate.exists():
                checksum_paths.add(candidate.resolve())
    if deprecated_checkpoints:
        checksum_paths.add(args.deprecated_fair_summary.expanduser().resolve())
        checksum_paths.add(args.deprecated_checkpoint_registry.expanduser().resolve())
        checksum_paths.update(args.deprecated_prediction_root.expanduser().resolve().rglob("*_predictions.json"))
        for row in deprecated_rows:
            candidate = Path(row["best_path"])
            if candidate.exists():
                checksum_paths.add(candidate.resolve())
    checksum_lines = []
    for path in sorted(checksum_paths):
        try:
            relative = path.relative_to(generation_root)
        except ValueError:
            continue
        checksum_lines.append(f"{sha256(path)}  {relative}")
    (output_root / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n", encoding="utf-8")
    print(f"Wrote: {output_csv}")
    print(f"Wrote: {output_json}")
    print(f"Wrote: {output_root / 'SHA256SUMS'}")
    print(f"PROVENANCE: PASS ({len(records)} artifact records)")


if __name__ == "__main__":
    main()
