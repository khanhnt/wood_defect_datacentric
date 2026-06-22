#!/usr/bin/env python3
"""Audit corrected common evaluation artifacts for data/path confounds.

The script is intentionally read-only. It checks that standard-metric and
negative-aware artifacts use the expected common evaluation datasets, checkpoint
paths, image memberships, and split counts. Run it on Vast for filesystem checks;
it also works locally on downloaded CSV/JSON files, with path-existence checks
reported as warnings when `/workspace` is unavailable.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


EXPECTED_SEEDS = (42, 43, 44)
VN_VARIANTS = ("baseline", "p2_illumination", "a1_crop", "a2_colorjitter", "p4_a4_combined")
VSB_VARIANTS = ("baseline", "p1_clahe", "p2_illumination", "p3_unsharp", "a1_crop", "a2_colorjitter", "p4_a4_combined")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
RAW_EVAL_VARIANTS = {"baseline", "a1_crop", "a2_colorjitter"}
PREPROC_BY_VARIANT = {
    "p1_clahe": "P1_CLAHE_luminance",
    "p2_illumination": "P2_illumination_normalization",
    "p3_unsharp": "P3_mild_unsharp",
    "p4_a4_combined": "P4_combined_safe",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--test-best-summary", type=Path, default=Path("results/corrected_common_eval/test_best_summary.csv"))
    parser.add_argument("--eval-map", type=Path, default=Path("results/corrected_common_eval/corrected_eval_dataset_map.csv"))
    parser.add_argument("--corrected-predictions-dir", type=Path, default=Path("results/negative_aware_corrected/predictions"))
    parser.add_argument("--corrected-raw-data", type=Path, default=Path("results/negative_aware_corrected/threshold_sweep/raw_data.csv"))
    parser.add_argument("--corrected-summary", type=Path, default=Path("results/negative_aware_corrected/threshold_sweep/summary_aggregated.csv"))
    parser.add_argument("--old-predictions-dir", type=Path, default=None)
    parser.add_argument("--old-summary", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("results/corrected_common_eval/integrity_audit"))
    parser.add_argument("--hash-checkpoints", action="store_true", help="Compute SHA256 for best.pt files if present.")
    parser.add_argument("--hash-images", action="store_true", help="Compute SHA256 for evaluation images if present; can be slow.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = AuditReport()

    test_df = read_csv(args.test_best_summary, report, "test_best_summary")
    map_df = read_csv(args.eval_map, report, "corrected_eval_dataset_map")
    raw_df = read_csv(args.corrected_raw_data, report, "corrected raw_data")
    summary_df = read_csv(args.corrected_summary, report, "corrected summary_aggregated")

    if test_df is not None:
        audit_test_best_summary(test_df, report)
    if map_df is not None:
        audit_eval_map(map_df, report)
        audit_dataset_yamls(map_df, report, args.output_dir, hash_images=args.hash_images)
    if raw_df is not None:
        audit_raw_data(raw_df, report)
    if summary_df is not None:
        audit_summary(summary_df, report)

    corrected_sets = audit_prediction_jsons(
        args.corrected_predictions_dir,
        report,
        args.output_dir / "corrected_prediction_image_membership.csv",
        results_root=args.results_root,
        hash_checkpoints=args.hash_checkpoints,
    )
    if args.old_predictions_dir:
        old_sets = audit_prediction_jsons(
            args.old_predictions_dir,
            report,
            args.output_dir / "old_prediction_image_membership.csv",
            results_root=args.results_root,
            hash_checkpoints=False,
            namespace="old",
        )
        compare_old_new_memberships(old_sets, corrected_sets, report, args.output_dir / "old_vs_corrected_membership_diff.csv")
    if args.old_summary and summary_df is not None:
        old_summary = read_csv(args.old_summary, report, "old summary_aggregated")
        if old_summary is not None:
            compare_summaries(old_summary, summary_df, report, args.output_dir / "old_vs_corrected_threshold_delta.csv")

    report_path = args.output_dir / "integrity_audit_report.md"
    report_path.write_text(report.render(), encoding="utf-8")
    print(report.render())
    print(f"Wrote: {report_path}")


class AuditReport:
    def __init__(self) -> None:
        self.lines: list[str] = ["# Corrected Evaluation Integrity Audit", ""]
        self.errors: list[str] = []
        self.warnings: list[str] = []

    def section(self, title: str) -> None:
        self.lines.extend(["", f"## {title}", ""])

    def info(self, text: str) -> None:
        self.lines.append(f"- {text}")

    def warn(self, text: str) -> None:
        self.warnings.append(text)
        self.lines.append(f"- WARNING: {text}")

    def error(self, text: str) -> None:
        self.errors.append(text)
        self.lines.append(f"- ERROR: {text}")

    def render(self) -> str:
        status = "FAIL" if self.errors else ("WARN" if self.warnings else "PASS")
        prefix = [
            f"Status: **{status}**",
            "",
            f"- Errors: {len(self.errors)}",
            f"- Warnings: {len(self.warnings)}",
            "",
        ]
        if self.errors:
            prefix.extend(["## Errors", "", *[f"- {item}" for item in self.errors], ""])
        if self.warnings:
            prefix.extend(["## Warnings", "", *[f"- {item}" for item in self.warnings], ""])
        return "\n".join(prefix + self.lines) + "\n"


def read_csv(path: Path, report: AuditReport, label: str) -> pd.DataFrame | None:
    path = path.expanduser()
    if not path.exists():
        report.warn(f"Missing {label}: {path}")
        return None
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        report.error(f"Could not read {label}: {path}: {exc}")
        return None
    report.section(label)
    report.info(f"{path} rows={len(df)} columns={list(df.columns)}")
    return df


def audit_test_best_summary(df: pd.DataFrame, report: AuditReport) -> None:
    report.section("Standard Val/Test Summary")
    expected_rows = 2 * len(EXPECTED_SEEDS) * (len(VN_VARIANTS) + len(VSB_VARIANTS))
    if len(df) != expected_rows:
        report.error(f"Expected {expected_rows} rows in test_best_summary, found {len(df)}")
    required = {"dataset", "run", "variant", "seed", "split", "n_images", "n_instances", "precision", "recall", "mAP50", "mAP50_95"}
    missing = required - set(df.columns)
    if missing:
        report.error(f"test_best_summary missing columns: {sorted(missing)}")
        return
    duplicate_count = int(df.duplicated(["dataset", "variant", "seed", "split"]).sum())
    if duplicate_count:
        report.error(f"Duplicate dataset/variant/seed/split rows: {duplicate_count}")
    expected = {
        "vnwoodknot": set(VN_VARIANTS),
        "vsb_rarefirst": set(VSB_VARIANTS),
    }
    for dataset, variants in expected.items():
        for variant in variants:
            for seed in EXPECTED_SEEDS:
                for split in ("val", "test"):
                    mask = (df["dataset"] == dataset) & (df["variant"] == variant) & (df["seed"] == seed) & (df["split"] == split)
                    if int(mask.sum()) != 1:
                        report.error(f"Missing or duplicated row: {dataset}/{variant}/seed{seed}/{split}")
    counts = df.groupby(["dataset", "split"])[["n_images", "n_instances"]].agg(["min", "max"]).reset_index()
    report.info("Split count ranges:")
    for _, row in counts.iterrows():
        report.info(
            f"{row[('dataset', '')]} {row[('split', '')]} images={row[('n_images', 'min')]}..{row[('n_images', 'max')]} "
            f"instances={row[('n_instances', 'min')]}..{row[('n_instances', 'max')]}"
        )
    for metric in ("precision", "recall", "mAP50", "mAP50_95"):
        bad = df[(df[metric] < 0) | (df[metric] > 1)]
        if len(bad):
            report.error(f"{metric} has values outside [0,1]: {len(bad)} rows")


def audit_eval_map(df: pd.DataFrame, report: AuditReport) -> None:
    report.section("Corrected Eval Dataset Map")
    required = {"dataset", "variant", "data_yaml"}
    missing = required - set(df.columns)
    if missing:
        report.error(f"eval map missing columns: {sorted(missing)}")
        return
    expected_rows = len(VN_VARIANTS) + len(VSB_VARIANTS)
    if len(df) != expected_rows:
        report.error(f"Expected {expected_rows} eval-map rows, found {len(df)}")
    for _, row in df.iterrows():
        dataset, variant, data_yaml = str(row["dataset"]), str(row["variant"]), str(row["data_yaml"])
        if variant in RAW_EVAL_VARIANTS:
            if "corrected_common_eval_yolo" in data_yaml or "generated_yolo" in data_yaml:
                report.error(f"{dataset}/{variant} should use raw canonical data, got {data_yaml}")
        elif variant in PREPROC_BY_VARIANT:
            expected_token = PREPROC_BY_VARIANT[variant]
            if expected_token not in data_yaml:
                report.error(f"{dataset}/{variant} should use {expected_token}, got {data_yaml}")
            if "corrected_common_eval_yolo" not in data_yaml:
                report.warn(f"{dataset}/{variant} preprocessing eval YAML is not under corrected_common_eval_yolo: {data_yaml}")
        else:
            report.warn(f"Unknown variant in eval map: {dataset}/{variant}")
        if not Path(data_yaml).exists():
            report.warn(f"Eval YAML path not accessible from this machine: {data_yaml}")
    report.info("Eval-map variant routing checks complete.")


def audit_dataset_yamls(df: pd.DataFrame, report: AuditReport, output_dir: Path, *, hash_images: bool) -> None:
    report.section("Dataset YAML Split Membership")
    rows: list[dict[str, Any]] = []
    seen_yamls = sorted({str(item) for item in df["data_yaml"].dropna()})
    for yaml_text in seen_yamls:
        data_yaml = Path(yaml_text)
        if not data_yaml.exists():
            continue
        data = yaml.safe_load(data_yaml.read_text(encoding="utf-8")) or {}
        root = resolve_dataset_root(data_yaml, data)
        for split in ("val", "test"):
            image_count = 0
            instance_count = 0
            empty_count = 0
            empty_non_empty_class_path: list[str] = []
            for split_dir in resolve_split_dirs(data_yaml, data, split):
                for image_path in sorted(path for path in split_dir.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS):
                    image_count += 1
                    label_path = label_for_image(image_path, dataset_root=root, split=split, split_dir=split_dir)
                    labels = []
                    if label_path.exists():
                        labels = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                    instance_count += len(labels)
                    if not labels:
                        empty_count += 1
                        if "knot_free" not in image_path.as_posix().lower():
                            empty_non_empty_class_path.append(image_path.as_posix())
                    rows.append(
                        {
                            "data_yaml": str(data_yaml),
                            "split": split,
                            "image": image_path.as_posix(),
                            "label": label_path.as_posix(),
                            "num_labels": len(labels),
                            "image_sha256": sha256_file(image_path) if hash_images else "",
                        }
                    )
            report.info(f"{data_yaml} {split}: images={image_count} instances={instance_count} empty_labels={empty_count}")
            if empty_non_empty_class_path:
                report.warn(
                    f"{data_yaml} {split} has {len(empty_non_empty_class_path)} empty-label image(s) outside knot_free; "
                    f"examples: {empty_non_empty_class_path[:5]}"
                )
    if rows:
        write_rows(output_dir / "dataset_membership.csv", rows)
        report.info(f"Wrote dataset membership: {output_dir / 'dataset_membership.csv'}")


def audit_raw_data(df: pd.DataFrame, report: AuditReport) -> None:
    report.section("Corrected Threshold Raw Data")
    required = {
        "variant",
        "seed",
        "threshold",
        "ap50",
        "precision",
        "recall",
        "fp_image_rate",
        "num_positive_images",
        "num_knotfree_images",
        "knotfree_fp_images",
    }
    missing = required - set(df.columns)
    if missing:
        report.error(f"raw_data missing columns: {sorted(missing)}")
        return
    expected_rows = len(VN_VARIANTS) * len(EXPECTED_SEEDS) * 19
    if len(df) != expected_rows:
        report.error(f"Expected {expected_rows} raw threshold rows, found {len(df)}")
    thresholds = sorted(float(item) for item in df["threshold"].unique())
    report.info(f"Thresholds: n={len(thresholds)} min={min(thresholds):.2f} max={max(thresholds):.2f}")
    for variant in VN_VARIANTS:
        seeds = sorted(int(item) for item in df[df["variant"] == variant]["seed"].unique())
        if seeds != list(EXPECTED_SEEDS):
            report.error(f"Variant {variant} has seeds {seeds}, expected {list(EXPECTED_SEEDS)}")
    counts = df.groupby(["variant", "seed"])[["num_positive_images", "num_knotfree_images"]].first()
    report.info(f"Positive/knot-free count combinations: {counts.value_counts().to_dict()}")


def audit_summary(df: pd.DataFrame, report: AuditReport) -> None:
    report.section("Corrected Threshold Summary")
    if len(df) != len(VN_VARIANTS) * 19:
        report.error(f"Expected {len(VN_VARIANTS) * 19} summary rows, found {len(df)}")
    if "n_seeds" in df.columns:
        bad = df[df["n_seeds"] != len(EXPECTED_SEEDS)]
        if len(bad):
            report.error(f"summary_aggregated has rows with n_seeds != 3: {len(bad)}")


def audit_prediction_jsons(
    predictions_dir: Path,
    report: AuditReport,
    membership_csv: Path,
    *,
    results_root: Path,
    hash_checkpoints: bool,
    namespace: str = "corrected",
) -> dict[str, tuple[set[str], set[str], set[str]]]:
    report.section(f"{namespace.title()} Prediction JSONs")
    predictions_dir = predictions_dir.expanduser()
    files = sorted(predictions_dir.glob("*_predictions.json"))
    if len(files) != len(VN_VARIANTS) * len(EXPECTED_SEEDS):
        report.warn(f"{namespace} predictions: expected 15 files, found {len(files)} in {predictions_dir}")
    sets: dict[str, tuple[set[str], set[str], set[str]]] = {}
    rows: list[dict[str, Any]] = []
    meta_rows: list[dict[str, Any]] = []
    for path in files:
        payload = json.loads(path.read_text(encoding="utf-8"))
        variant = str(payload.get("variant"))
        seed = int(payload.get("seed"))
        key = f"{variant}_seed{seed}"
        images = payload.get("images") or []
        all_ids = {str(item.get("canonical_id")) for item in images}
        empty_ids = {str(item.get("canonical_id")) for item in images if bool(item.get("is_knot_free"))}
        positive_ids = all_ids - empty_ids
        sets[key] = (all_ids, empty_ids, positive_ids)
        checkpoint = str(payload.get("checkpoint_path", ""))
        expected_checkpoint = results_root / "multiseed" / "vnwoodknot" / "per_seed" / "runs" / key / "ultralytics" / "train" / "weights" / "best.pt"
        checkpoint_exists = Path(checkpoint).exists()
        expected_exists = expected_checkpoint.exists()
        if checkpoint and checkpoint != str(expected_checkpoint) and expected_exists:
            report.warn(f"{namespace} {key} checkpoint path differs from expected local path: {checkpoint} vs {expected_checkpoint}")
        if not checkpoint_exists:
            report.warn(f"{namespace} {key} checkpoint path not accessible from this machine: {checkpoint}")
        data_yaml = str(payload.get("dataset_yaml", ""))
        if not Path(data_yaml).exists():
            report.warn(f"{namespace} {key} dataset_yaml not accessible from this machine: {data_yaml}")
        meta_rows.append(
            {
                "namespace": namespace,
                "file": path.as_posix(),
                "variant": variant,
                "seed": seed,
                "dataset_yaml": data_yaml,
                "checkpoint_path": checkpoint,
                "checkpoint_sha256": sha256_file(Path(checkpoint)) if hash_checkpoints and Path(checkpoint).exists() else "",
                "base_confidence_threshold": payload.get("base_confidence_threshold"),
                "num_images": len(images),
                "num_empty": len(empty_ids),
                "num_positive": len(positive_ids),
                "num_gt_boxes": sum(len(item.get("gt_boxes") or []) for item in images),
            }
        )
        for item in images:
            rows.append(
                {
                    "namespace": namespace,
                    "variant": variant,
                    "seed": seed,
                    "canonical_id": item.get("canonical_id"),
                    "image": item.get("image"),
                    "image_path": item.get("image_path"),
                    "is_knot_free": bool(item.get("is_knot_free")),
                    "num_gt_boxes": len(item.get("gt_boxes") or []),
                    "num_predictions_exported": len(item.get("predictions") or []),
                }
            )
    if meta_rows:
        write_rows(membership_csv.parent / f"{namespace}_prediction_metadata.csv", meta_rows)
        write_rows(membership_csv, rows)
        report.info(f"Wrote {namespace} prediction metadata: {membership_csv.parent / f'{namespace}_prediction_metadata.csv'}")
        report.info(f"Wrote {namespace} prediction membership: {membership_csv}")
    if sets:
        first_key = sorted(sets)[0]
        ref = sets[first_key]
        report.info(f"Reference prediction membership {first_key}: all={len(ref[0])} empty={len(ref[1])} positive={len(ref[2])}")
        for key, value in sorted(sets.items()):
            if value[0] != ref[0] or value[1] != ref[1] or value[2] != ref[2]:
                report.error(
                    f"{namespace} prediction membership differs for {key}: "
                    f"all_diff={len(value[0] ^ ref[0])} empty_diff={len(value[1] ^ ref[1])} positive_diff={len(value[2] ^ ref[2])}"
                )
        empty_non_knotfree = sorted(item for item in ref[1] if "knot_free" not in item.lower())
        if empty_non_knotfree:
            report.warn(f"{namespace} empty-label images outside knot_free: {empty_non_knotfree[:10]} (n={len(empty_non_knotfree)})")
    return sets


def compare_old_new_memberships(
    old_sets: dict[str, tuple[set[str], set[str], set[str]]],
    new_sets: dict[str, tuple[set[str], set[str], set[str]]],
    report: AuditReport,
    output_csv: Path,
) -> None:
    report.section("Old vs Corrected Prediction Membership")
    rows: list[dict[str, Any]] = []
    for key in sorted(set(old_sets) & set(new_sets)):
        old_all, old_empty, old_positive = old_sets[key]
        new_all, new_empty, new_positive = new_sets[key]
        only_old = sorted(old_all - new_all)
        only_new = sorted(new_all - old_all)
        rows.append(
            {
                "run": key,
                "old_all": len(old_all),
                "new_all": len(new_all),
                "old_empty": len(old_empty),
                "new_empty": len(new_empty),
                "old_positive": len(old_positive),
                "new_positive": len(new_positive),
                "only_old": " ".join(only_old[:20]),
                "only_new": " ".join(only_new[:20]),
            }
        )
        if only_old or only_new:
            report.warn(f"{key}: old/new image membership differs; only_old={only_old[:3]} only_new={only_new[:3]}")
    write_rows(output_csv, rows)
    report.info(f"Wrote old-vs-corrected membership diff: {output_csv}")


def compare_summaries(old: pd.DataFrame, new: pd.DataFrame, report: AuditReport, output_csv: Path) -> None:
    report.section("Old vs Corrected Threshold Summary")
    key_cols = ["variant", "threshold"]
    merged = old.merge(new, on=key_cols, suffixes=("_old", "_new"))
    rows: list[dict[str, Any]] = []
    for _, row in merged.iterrows():
        out = {"variant": row["variant"], "threshold": row["threshold"]}
        for metric in ("ap50_mean", "precision_mean", "recall_mean", "fp_image_rate_mean", "knotfree_fp_images_mean"):
            old_value = float(row[f"{metric}_old"])
            new_value = float(row[f"{metric}_new"])
            out[f"{metric}_old"] = f"{old_value:.6f}"
            out[f"{metric}_new"] = f"{new_value:.6f}"
            out[f"{metric}_delta"] = f"{new_value - old_value:.6f}"
        rows.append(out)
    write_rows(output_csv, rows)
    report.info(f"Wrote old-vs-corrected threshold deltas: {output_csv}")


def resolve_dataset_root(data_yaml: Path, data: dict[str, Any]) -> Path:
    root = Path(str(data.get("path") or data_yaml.parent)).expanduser()
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.resolve()


def resolve_split_dirs(data_yaml: Path, data: dict[str, Any], split: str) -> list[Path]:
    value = data.get(split)
    values = value if isinstance(value, list) else [value]
    root = resolve_dataset_root(data_yaml, data)
    dirs = []
    for item in values:
        if not item:
            continue
        path = Path(str(item)).expanduser()
        dirs.append(path.resolve() if path.is_absolute() else (root / path).resolve())
    return dirs


def label_for_image(image_path: Path, *, dataset_root: Path, split: str, split_dir: Path) -> Path:
    try:
        rel = image_path.relative_to(split_dir)
        return (dataset_root / "labels" / split / rel).with_suffix(".txt")
    except ValueError:
        pass
    try:
        rel = image_path.relative_to(dataset_root)
        parts = list(rel.parts)
        if "images" in parts:
            parts[parts.index("images")] = "labels"
            return (dataset_root / Path(*parts)).with_suffix(".txt")
    except ValueError:
        pass
    return image_path.with_suffix(".txt")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
