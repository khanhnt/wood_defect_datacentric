#!/usr/bin/env python3
"""Generate LaTeX tables from multiseed summary CSV files."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


METRICS = (
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("mAP50", "mAP50"),
    ("mAP50_95", "mAP50--95"),
)

DATASET_LABELS = {
    "vnwoodknot": "VNWoodKnot",
    "vsb_rarefirst": "VSB rare-first",
}

DATASET_CAPTIONS = {
    "vnwoodknot": "Multiseed validation results on VNWoodKnot.",
    "vsb_rarefirst": "Multiseed validation results on the VSB rare-first curated benchmark.",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV from aggregate_results.py.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .tex path. Defaults to <input parent>/multiseed_latex_tables.tex.",
    )
    parser.add_argument("--caption-prefix", default="", help="Optional text prepended to each table caption.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        raise SystemExit(f"Input summary CSV does not exist: {input_path}")

    rows = read_rows(input_path)
    if not rows:
        raise SystemExit(f"No rows found in summary CSV: {input_path}")

    output_path = args.output.expanduser().resolve() if args.output else input_path.parent / "multiseed_latex_tables.tex"
    latex = render_all_tables(rows, caption_prefix=args.caption_prefix)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(latex + "\n", encoding="utf-8")

    print(latex)
    print(f"\nWrote: {output_path}")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def render_all_tables(rows: list[dict[str, str]], *, caption_prefix: str) -> str:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)

    tables = []
    for dataset in ("vnwoodknot", "vsb_rarefirst"):
        dataset_rows = grouped.get(dataset, [])
        if dataset_rows:
            tables.append(render_table(dataset, dataset_rows, caption_prefix=caption_prefix))
    for dataset in sorted(set(grouped) - {"vnwoodknot", "vsb_rarefirst"}):
        tables.append(render_table(dataset, grouped[dataset], caption_prefix=caption_prefix))
    return "\n\n".join(tables)


def render_table(dataset: str, rows: list[dict[str, str]], *, caption_prefix: str) -> str:
    best_values = best_means(rows)
    caption = DATASET_CAPTIONS.get(dataset, f"Multiseed validation results on {DATASET_LABELS.get(dataset, dataset)}.")
    if caption_prefix:
        caption = f"{caption_prefix.rstrip()} {caption}"
    label = f"tab:{latex_label(dataset)}_multiseed"

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{{latex_escape(caption)}}}",
        rf"\label{{{label}}}",
        r"\begin{tabular}{lccccc}",
        r"\toprule",
        r"Variant & Seeds & Precision & Recall & mAP50 & mAP50--95 \\",
        r"\midrule",
    ]
    for row in rows:
        cells = [
            latex_escape(row.get("variant_label") or row["variant"]),
            latex_escape(row.get("seeds", "")),
        ]
        for metric, _ in METRICS:
            cells.append(format_metric_cell(row, metric, best_values.get(metric)))
        lines.append(" & ".join(cells) + r" \\")
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )
    return "\n".join(lines)


def best_means(rows: list[dict[str, str]]) -> dict[str, float]:
    values: dict[str, float] = {}
    for metric, _ in METRICS:
        metric_values = [parse_float(row.get(f"{metric}_mean", "")) for row in rows]
        metric_values = [value for value in metric_values if value == value]
        if metric_values:
            values[metric] = max(metric_values)
    return values


def format_metric_cell(row: dict[str, str], metric: str, best_value: float | None) -> str:
    mean = parse_float(row.get(f"{metric}_mean", ""))
    std = parse_float(row.get(f"{metric}_std", ""))
    if mean != mean:
        return "--"
    text = f"{mean:.3f} {{\\pm}} {std:.3f}"
    if best_value is not None and abs(mean - best_value) < 5e-7:
        return rf"\textbf{{{text}}}"
    return text


def parse_float(value: str | float | int) -> float:
    try:
        return float(str(value).strip())
    except Exception:
        return float("nan")


def latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in str(value))


def latex_label(value: str) -> str:
    return "".join(char if char.isalnum() else "_" for char in value.lower()).strip("_")


if __name__ == "__main__":
    main()
