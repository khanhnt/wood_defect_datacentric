#!/usr/bin/env python3
"""Validate the lightweight public artifacts for the revised manuscript."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES = PROJECT_ROOT / "results" / "tables"
YOLO = TABLES / "yolov8s"
FRCNN = TABLES / "fasterrcnn"


def main() -> None:
    checks = [
        check_required_artifacts(),
        check_cardinalities(),
        check_paper_values(),
        check_vnwoodknot_manifest(),
        check_vsb_rarefirst_manifest(),
        check_vsb_clean_manifest(),
        check_vsb_clean_partition(),
        check_fasterrcnn_provenance(),
        check_packaged_checksums(),
        check_deprecated_note(),
        check_secret_scan(),
    ]
    print("\nRELEASE INTEGRITY REPORT")
    for name, passed, detail in checks:
        print(f"- {'PASS' if passed else 'FAIL'}: {name} -- {detail}")
    if not all(passed for _, passed, _ in checks):
        raise SystemExit(1)


def check_required_artifacts() -> tuple[str, bool, str]:
    paths = (
        YOLO / "fair_metrics_per_seed.csv",
        YOLO / "fair_metrics_summary.csv",
        YOLO / "validation_threshold_selection.csv",
        YOLO / "locked_test_operating_points_per_seed.csv",
        YOLO / "locked_test_operating_points_summary.csv",
        YOLO / "locked_test_sensitivity_summary.csv",
        FRCNN / "standard" / "per_seed_metrics.csv",
        FRCNN / "standard" / "summary.csv",
        FRCNN / "negative_aware" / "test_operating_metrics_per_seed.csv",
        FRCNN / "negative_aware" / "test_operating_summary.csv",
        FRCNN / "provenance" / "runtime_preflight.json",
        TABLES / "SHA256SUMS",
        PROJECT_ROOT / "data" / "vsb_clean_manifest" / "tile_geometry_summary.json",
    )
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in paths if not path.is_file()]
    return "required release artifacts", not missing, "all present" if not missing else ", ".join(missing)


def check_cardinalities() -> tuple[str, bool, str]:
    targets = (
        (YOLO / "fair_metrics_per_seed.csv", 84),
        (YOLO / "fair_metrics_summary.csv", 28),
        (YOLO / "validation_threshold_selection.csv", 56),
        (YOLO / "locked_test_operating_points_per_seed.csv", 168),
        (YOLO / "locked_test_operating_points_summary.csv", 56),
        (YOLO / "locked_test_sensitivity_summary.csv", 56),
        (FRCNN / "standard" / "per_seed_metrics.csv", 18),
        (FRCNN / "standard" / "summary.csv", 6),
        (FRCNN / "negative_aware" / "test_operating_metrics_per_seed.csv", 36),
        (FRCNN / "negative_aware" / "test_operating_summary.csv", 12),
    )
    observed = [(path, len(read_csv(path)), expected) for path, expected in targets]
    passed = all(count == expected for _, count, expected in observed)
    detail = "; ".join(f"{path.name}={count}/{expected}" for path, count, expected in observed)
    return "result cardinalities", passed, detail


def check_paper_values() -> tuple[str, bool, str]:
    yolo_standard = keyed(
        YOLO / "fair_metrics_summary.csv", ("dataset", "variant", "split")
    )
    yolo_operating = keyed(
        YOLO / "locked_test_operating_points_summary.csv",
        ("dataset", "variant", "epsilon"),
    )
    frcnn_standard = keyed(
        FRCNN / "standard" / "summary.csv", ("variant", "split")
    )
    frcnn_operating = keyed(
        FRCNN / "negative_aware" / "test_operating_summary.csv",
        ("variant", "epsilon"),
    )
    targets = (
        ("YOLO VN baseline mAP50", yolo_standard[("vnwoodknot", "baseline", "test")]["mAP50_mean"], 0.820),
        ("YOLO VN A1 mAP50-95", yolo_standard[("vnwoodknot", "a1_crop", "test")]["mAP50_95_mean"], 0.404),
        ("YOLO VSB baseline mAP50", yolo_standard[("vsb_rarefirst", "baseline", "test")]["mAP50_mean"], 0.810),
        ("YOLO VN strict baseline AP50", yolo_operating[("vnwoodknot", "baseline", "0.0")]["mAP50_mean"], 0.803),
        ("YOLO VSB strict P2 AP50", yolo_operating[("vsb_rarefirst", "p2_illumination", "0.0")]["mAP50_mean"], 0.340),
        ("FRCNN baseline mAP50", frcnn_standard[("baseline", "test")]["mAP50_mean"], 0.866),
        ("FRCNN A1 mAP50", frcnn_standard[("a1_crop", "test")]["mAP50_mean"], 0.695),
        ("FRCNN A2 mAP50", frcnn_standard[("a2_colorjitter", "test")]["mAP50_mean"], 0.886),
        ("FRCNN strict baseline AP50", frcnn_operating[("baseline", "0.00")]["retained_AP50_mean"], 0.797),
        ("FRCNN eps=.01 A2 AP50", frcnn_operating[("a2_colorjitter", "0.01")]["retained_AP50_mean"], 0.821),
    )
    deltas = [abs(float(value) - expected) for _, value, expected in targets]
    passed = all(delta <= 0.0007 for delta in deltas)
    detail = "; ".join(f"{name}={float(value):.3f}" for name, value, _ in targets)
    return "key revised-manuscript values", passed, detail


def check_vnwoodknot_manifest() -> tuple[str, bool, str]:
    path = PROJECT_ROOT / "data" / "vnwoodknot_split" / "manifest.jsonl"
    counts: Counter[str] = Counter()
    defective: Counter[str] = Counter()
    empty: Counter[str] = Counter()
    boxes: Counter[str] = Counter()
    excluded = None

    for row in read_jsonl(path):
        split = "val" if row["split"] == "validation" else str(row["split"])
        annotations = row.get("annotations") or []
        counts[split] += 1
        defective[split] += int(bool(annotations))
        empty[split] += int(not annotations)
        boxes[split] += len(annotations)
        if row["image_id"] == "train/2/img_3671":
            excluded = row

    passed = (
        dict(counts) == {"train": 1060, "val": 226, "test": 229}
        and dict(defective) == {"train": 710, "val": 151, "test": 154}
        and dict(empty) == {"train": 350, "val": 75, "test": 75}
        and dict(boxes) == {"train": 715, "val": 151, "test": 155}
        and excluded is not None
        and excluded["image_path"] == "train/2/img_3671.jpg"
    )
    detail = (
        f"images={dict(counts)}, defective={dict(defective)}, "
        f"empty={dict(empty)}, boxes={dict(boxes)}, "
        "explicit training exclusion=train/2/img_3671"
    )
    return "VNWoodKnot manifest", passed, detail


def check_vsb_rarefirst_manifest() -> tuple[str, bool, str]:
    path = PROJECT_ROOT / "data" / "vsb_rarefirst_split" / "manifest.jsonl"
    counts: dict[str, int] = {}
    empty: dict[str, int] = {}
    boxes: dict[str, int] = {}
    sources: set[str] = set()

    for row in read_jsonl(path):
        split = str(row["split"])
        annotations = row.get("annotations") or []
        sources.add(str(row["source_image_id"]))
        counts[split] = counts.get(split, 0) + 1
        boxes[split] = boxes.get(split, 0) + len(annotations)
        empty[split] = empty.get(split, 0) + int(not annotations)

    passed = (
        counts == {"train": 7679, "val": 977, "test": 972}
        and empty == {"train": 2297, "val": 276, "test": 276}
        and boxes == {"train": 9346, "val": 1146, "test": 1173}
        and len(sources) == 3600
    )
    detail = (
        f"sources={len(sources)}, images={counts}, "
        f"empty={empty}, boxes={boxes}"
    )
    return "VSB rare-first manifest", passed, detail


def check_vsb_clean_manifest() -> tuple[str, bool, str]:
    manifest = (
        PROJECT_ROOT / "data" / "vsb_clean_manifest"
        / "clean_tile_manifest.csv"
    )
    summary_path = (
        PROJECT_ROOT / "data" / "vsb_clean_manifest"
        / "tile_geometry_summary.json"
    )
    rows = read_csv(manifest)
    sources = {row["source_id"] for row in rows}
    labels_empty = all(
        row.get("annotations", "[]").strip() == "[]" for row in rows
    )

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["source_id"]].append(row)

    geometry_ok = True
    widths: Counter[int] = Counter()

    for source_id, source_rows in grouped.items():
        ordered = sorted(source_rows, key=lambda row: int(row["tile_index"]))
        if len(ordered) != 3:
            geometry_ok = False
            continue

        source_width = int(ordered[0]["source_width"])
        widths[source_width] += 1
        expected_x = [0, 896, source_width - 1024]
        actual_x = [int(row["x"]) for row in ordered]
        expected_overlap = [
            "",
            "128",
            str(1024 - (expected_x[2] - expected_x[1])),
        ]
        actual_overlap = [
            row["overlap_with_previous_x"] for row in ordered
        ]
        expected_ids = [
            f"{source_id}__x{x:04d}_y0000" for x in expected_x
        ]
        actual_ids = [row["tile_id"] for row in ordered]

        geometry_ok &= (
            actual_x == expected_x
            and actual_overlap == expected_overlap
            and actual_ids == expected_ids
            and all(int(row["width"]) == 1024 for row in ordered)
            and all(int(row["height"]) == 1024 for row in ordered)
            and all(int(row["source_height"]) == 1024 for row in ordered)
        )

    nonstandard = sum(
        count for width, count in widths.items() if width != 2800
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    passed = (
        len(rows) == 5976
        and len(sources) == 1992
        and labels_empty
        and geometry_ok
        and widths[2800] == 1980
        and nonstandard == 12
        and summary.get("status") == "PASS"
        and summary.get("num_source_images") == 1992
        and summary.get("num_tiles") == 5976
        and summary.get("nonstandard_source_count") == 12
    )
    detail = (
        f"sources={len(sources)}, tiles={len(rows)}, "
        f"all_empty={labels_empty}, geometry_ok={geometry_ok}, "
        f"width_2800={widths[2800]}, "
        f"nonstandard_width={nonstandard}"
    )
    return "VSB strict-clean manifest", passed, detail


def check_vsb_clean_partition() -> tuple[str, bool, str]:
    path = PROJECT_ROOT / "docs" / "runbook_test_evidence" / "vsb_clean_partition_report.json"
    report = json.loads(path.read_text(encoding="utf-8"))
    passed = (
        report.get("status") == "PASS"
        and report.get("selection_source_count") == 996
        and report.get("final_test_source_count") == 996
        and report.get("source_overlap") == 0
        and report.get("selection_tiles_per_variant") == 2988
        and report.get("final_test_tiles_per_variant") == 2988
    )
    detail = (
        f"selection={report.get('selection_source_count')} sources/{report.get('selection_tiles_per_variant')} tiles, "
        f"test={report.get('final_test_source_count')} sources/{report.get('final_test_tiles_per_variant')} tiles, "
        f"overlap={report.get('source_overlap')}"
    )
    return "VSB source-disjoint clean partition", passed, detail


def check_fasterrcnn_provenance() -> tuple[str, bool, str]:
    finalization = json.loads(
        (FRCNN / "standard" / "finalization_report.json").read_text(encoding="utf-8")
    )
    runtime = json.loads(
        (FRCNN / "provenance" / "runtime_preflight.json").read_text(encoding="utf-8")
    )
    registry = read_csv(FRCNN / "standard" / "checkpoint_registry.csv")
    passed = (
        finalization.get("status") == "PASS"
        and finalization.get("checkpoint_runs") == 9
        and finalization.get("metric_rows") == 18
        and finalization.get("prediction_exports") == 18
        and runtime.get("status") == "PASS"
        and len(registry) == 9
    )
    detail = (
        f"runtime={runtime.get('status')}, checkpoints={len(registry)}, "
        f"metric_rows={finalization.get('metric_rows')}, exports={finalization.get('prediction_exports')}"
    )
    return "Faster R-CNN provenance", passed, detail


def check_packaged_checksums() -> tuple[str, bool, str]:
    checksum_file = TABLES / "SHA256SUMS"
    failures: list[str] = []
    listed: set[str] = set()
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        digest, relative = line.split("  ", 1)
        path = TABLES / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            failures.append(relative)
        listed.add(relative)
    actual = {
        path.relative_to(TABLES).as_posix()
        for path in TABLES.rglob("*")
        if path.is_file() and path != checksum_file
    }
    unlisted = sorted(actual - listed)
    stale = sorted(listed - actual)
    passed = not failures and not unlisted and not stale
    detail = (
        f"checked={len(listed)}, failures={len(failures)}, "
        f"unlisted={len(unlisted)}, stale={len(stale)}"
    )
    return "packaged table checksums", passed, detail


def check_deprecated_note() -> tuple[str, bool, str]:
    paths = (
        PROJECT_ROOT / "results" / "_deprecated" / "README.md",
        PROJECT_ROOT / "results" / "_deprecated" / "pre_revision_tables" / "README.md",
    )
    text = "\n".join(path.read_text(encoding="utf-8") for path in paths if path.is_file())
    passed = "6252" in text and "5976" in text and "supersed" in text.lower()
    return "deprecated-output documentation", passed, "6,252-tile output is separated from the 5,976-tile strict-clean source set"


def check_secret_scan() -> tuple[str, bool, str]:
    patterns = (
        re.compile(r"/Users/[A-Za-z0-9._-]+"),
        re.compile(r"/Volumes/[A-Za-z0-9._ -]+"),
        re.compile(r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
        re.compile(r"BEGIN (RSA|OPENSSH|PRIVATE) KEY"),
    )
    bad: list[str] = []
    for path in release_files():
        if path.name == ".env":
            bad.append(str(path.relative_to(PROJECT_ROOT)))
            continue
        if not path.is_file() or path.stat().st_size > 50_000_000:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if any(pattern.search(text) for pattern in patterns):
            bad.append(str(path.relative_to(PROJECT_ROOT)))
    detail = "no personal paths or credential-like strings found" if not bad else ", ".join(bad[:10])
    return "secret and personal-path scan", not bad, detail


def keyed(path: Path, columns: tuple[str, ...]) -> dict[tuple[str, ...], dict[str, str]]:
    return {tuple(row[column] for column in columns): row for row in read_csv(path)}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def release_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    names = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    extra_roots = (
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "docs" / "RELEASE_NOTES_REVISION.md",
        TABLES,
        PROJECT_ROOT / "results" / "_deprecated",
    )
    for root in extra_roots:
        if root.is_file():
            names.add(str(root.relative_to(PROJECT_ROOT)))
            continue
        if root.exists():
            for dirpath, _, filenames in os.walk(root):
                for filename in filenames:
                    names.add(str((Path(dirpath) / filename).relative_to(PROJECT_ROOT)))
    return [PROJECT_ROOT / name for name in sorted(names)]


if __name__ == "__main__":
    main()
