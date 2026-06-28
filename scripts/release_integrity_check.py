#!/usr/bin/env python3
"""Run release integrity checks for the public paper artifact bundle."""

from __future__ import annotations

import csv
import json
import os
import re
import subprocess
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TABLES = PROJECT_ROOT / "results" / "tables"


def main() -> None:
    checks = [
        check_table_numbers(),
        check_vsb_rarefirst_manifest(),
        check_vsb_clean_denominator(),
        check_leakage_report(),
        check_deprecated_note(),
        check_secret_scan(),
    ]
    print("\nRELEASE INTEGRITY REPORT")
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        print(f"- {status}: {name} -- {detail}")
    if not all(passed for _, passed, _ in checks):
        raise SystemExit(1)


def check_table_numbers() -> tuple[str, bool, str]:
    vn = read_operational(TABLES / "vnwoodknot_operational_selection.csv")
    vsb = read_operational(TABLES / "vsb_clean_operational_selection.csv")
    targets = [
        ("VN Baseline recall", vn["baseline"]["retained_recall_mean"], 0.357),
        ("VN Baseline AP50", vn["baseline"]["retained_AP50_mean"], 0.357),
        ("VSB A1 recall", vsb["a1_crop"]["retained_recall_mean"], 0.201),
        ("VSB A1 AP50", vsb["a1_crop"]["retained_AP50_mean"], 0.221),
        ("VSB P4+A4 recall", vsb["p4_a4_combined"]["retained_recall_mean"], 0.021),
        ("VSB P4+A4 AP50", vsb["p4_a4_combined"]["retained_AP50_mean"], 0.017),
    ]
    deltas = [abs(float(value) - expected) for _, value, expected in targets]
    passed = all(delta <= 0.0007 for delta in deltas)
    detail = "; ".join(f"{name}={float(value):.3f}" for name, value, _ in targets)
    return "paper table values within rounding", passed, detail


def check_vsb_clean_denominator() -> tuple[str, bool, str]:
    path = TABLES / "vsb_clean_threshold_sweep_summary.csv"
    rows = read_csv(path)
    values = {round(float(row["num_knotfree_images_mean"])) for row in rows}
    passed = values == {5976}
    return "strict VSB clean denominator", passed, f"num_clean_tiles={sorted(values)}"


def check_vsb_rarefirst_manifest() -> tuple[str, bool, str]:
    path = PROJECT_ROOT / "data" / "vsb_rarefirst_split" / "manifest.jsonl"
    if not path.exists():
        return "VSB rare-first tile manifest", False, "missing data/vsb_rarefirst_split/manifest.jsonl"
    counts: dict[str, int] = {}
    boxes: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            split = str(row.get("split"))
            counts[split] = counts.get(split, 0) + 1
            boxes[split] = boxes.get(split, 0) + len(row.get("annotations") or [])
    expected = {"train": 7679, "val": 977, "test": 972}
    passed = counts == expected
    detail = f"splits={counts}, boxes={boxes}"
    return "VSB rare-first tile manifest", passed, detail


def check_leakage_report() -> tuple[str, bool, str]:
    path = TABLES / "vsb_clean_set_report.json"
    report = json.loads(path.read_text())
    leakage = report["leakage"]
    expected_zero = [
        "train_val_source_overlap",
        "train_val_tile_overlap",
        "test_source_overlap",
        "test_tile_overlap",
    ]
    passed = all(int(leakage.get(key, -1)) == 0 for key in expected_zero)
    detail = ", ".join(f"{key}={leakage.get(key)}" for key in expected_zero)
    detail += f", clean_sources={report.get('num_clean_source_images')}, clean_tiles={report.get('num_clean_tiles')}"
    return "VSB clean leakage report", passed, detail


def check_deprecated_note() -> tuple[str, bool, str]:
    note = PROJECT_ROOT / "results" / "_deprecated" / "README.md"
    if not note.exists():
        return "deprecated VSB clean note", False, "missing results/_deprecated/README.md"
    text = note.read_text(encoding="utf-8")
    passed = "6252" in text and "5976" in text
    return "deprecated VSB clean note", passed, "README explains 6252 denominator exclusion"


def check_secret_scan() -> tuple[str, bool, str]:
    files = tracked_files()
    users_prefix = "/" + "Users" + "/" + "ntkhanh"
    volumes_prefix = "/" + "Volumes" + "/" + "Data"
    patterns = [
        re.compile(re.escape(users_prefix)),
        re.compile(re.escape(volumes_prefix)),
        re.compile(r"(?i)(api[_-]?key|secret[_-]?key|access[_-]?token)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{16,}"),
        re.compile(r"BEGIN (RSA|OPENSSH|PRIVATE) KEY"),
    ]
    bad: list[str] = []
    for path in files:
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
    passed = not bad
    detail = "no personal paths or credential-like strings found" if passed else ", ".join(bad[:10])
    return "secret and personal-path scan", passed, detail


def read_operational(path: Path) -> dict[str, dict[str, str]]:
    return {row["variant"]: row for row in read_csv(path)}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def tracked_files() -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files"],
        cwd=PROJECT_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    extra_roots = [
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "REPRODUCE.md",
        PROJECT_ROOT / "CITATION.cff",
        PROJECT_ROOT / "LICENSE",
        PROJECT_ROOT / "analysis",
        PROJECT_ROOT / "scripts",
        PROJECT_ROOT / "data" / "README.md",
        PROJECT_ROOT / "data" / "prepare_vsb.py",
        PROJECT_ROOT / "data" / "prepare_vnwoodknot.py",
        PROJECT_ROOT / "data" / "processed",
        PROJECT_ROOT / "data" / "vsb_rarefirst_split",
        PROJECT_ROOT / "data" / "vsb_clean_manifest",
        PROJECT_ROOT / "data" / "vnwoodknot_split",
        PROJECT_ROOT / "results" / "tables",
        PROJECT_ROOT / "results" / "_deprecated",
    ]
    names = {line.strip() for line in result.stdout.splitlines() if line.strip()}
    for root in extra_roots:
        if root.is_file():
            names.add(str(root.relative_to(PROJECT_ROOT)))
        elif root.exists():
            for dirpath, _, filenames in os.walk(root):
                for filename in filenames:
                    names.add(str((Path(dirpath) / filename).relative_to(PROJECT_ROOT)))
    return [PROJECT_ROOT / name for name in sorted(names)]


if __name__ == "__main__":
    main()
