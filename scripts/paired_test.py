#!/usr/bin/env python3
"""
Compute supporting statistics for the multi-seed comparison, using the
per-seed FINAL-EPOCH metrics already produced by training (no re-training).

For each variant vs. the baseline (per dataset) it reports:
  - mean difference
  - standardised effect size (Cohen's d, pooled SD)
  - per-seed sign consistency (all three seeds agree?)
  - paired t-test p-value (reported as SUPPORTING evidence only; n=3 is low power)

Reads per-seed final-epoch mAP50 from:
  results/multiseed/{dataset}/per_seed/runs/{variant}_seed{seed}/ultralytics/train/results.csv
Use the SAME column the paper tables use (final-epoch metrics/mAP50(B)).

Fill the [TODO] p-values in main.tex from this script's output.
"""
import os, glob, csv, math, argparse
from itertools import combinations

SEEDS = [42, 43, 44]
DATASETS = {
    "vnwoodknot": ["baseline","p2_illumination","a1_crop","a2_colorjitter","p4_a4_combined"],
    "vsb_rarefirst": ["baseline","p1_clahe","p2_illumination","p3_unsharp",
                      "a1_crop","a2_colorjitter","p4_a4_combined"],
}
# Ultralytics column name for mAP50 (verify in your results.csv header)
MAP50_COL_CANDIDATES = ["metrics/mAP50(B)", "metrics/mAP_0.5", "metrics/mAP50"]

def read_final_map50(results_csv):
    with open(results_csv) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty: {results_csv}")
    header = {k.strip(): k for k in rows[-1].keys()}
    col = next((header[c] for c in MAP50_COL_CANDIDATES if c in header), None)
    if col is None:
        raise KeyError(f"mAP50 column not found in {results_csv}; "
                       f"headers={list(header)}")
    return float(rows[-1][col.strip()] if col.strip() in rows[-1] else rows[-1][col])

def per_seed_values(root, dataset, variant):
    vals = []
    for sd in SEEDS:
        pat = os.path.join(root, "multiseed", dataset, "per_seed", "runs",
                           f"{variant}_seed{sd}", "ultralytics", "train", "results.csv")
        hits = glob.glob(pat)
        if not hits:
            raise FileNotFoundError(pat)
        vals.append(read_final_map50(hits[0]))
    return vals

def paired_t(diffs):
    n = len(diffs); m = sum(diffs)/n
    var = sum((d-m)**2 for d in diffs)/(n-1) if n > 1 else 0.0
    sd = math.sqrt(var)
    if sd == 0:
        return float("inf"), 0.0  # identical-direction, zero variance
    t = m/(sd/math.sqrt(n))
    # two-sided p from t with df=n-1, via survival function approx (use scipy if available)
    try:
        from scipy import stats
        p = 2*stats.t.sf(abs(t), df=n-1)
    except Exception:
        p = float("nan")  # install scipy for exact p; t-value still reported
    return t, p

def cohend(a, b):
    ma, mb = sum(a)/len(a), sum(b)/len(b)
    va = sum((x-ma)**2 for x in a)/(len(a)-1)
    vb = sum((x-mb)**2 for x in b)/(len(b)-1)
    sp = math.sqrt((va+vb)/2) if (va+vb) > 0 else 0.0
    return (ma-mb)/sp if sp > 0 else float("inf"), ma-mb

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="./results")
    args = ap.parse_args()
    for ds, variants in DATASETS.items():
        print(f"\n=== {ds}: variant vs baseline (mAP50, final epoch, n=3 seeds) ===")
        base = per_seed_values(args.results_dir, ds, "baseline")
        for v in variants:
            if v == "baseline":
                continue
            vv = per_seed_values(args.results_dir, ds, v)
            diffs = [x-y for x, y in zip(vv, base)]
            d, md = cohend(vv, base)
            t, p = paired_t(diffs)
            consistent = all(x > 0 for x in diffs) or all(x < 0 for x in diffs)
            pstr = f"{p:.4f}" if p == p else "install scipy"
            print(f"{v:18s} diff={md:+.4f}  d={d:+.2f}  "
                  f"per-seed consistent={'yes' if consistent else 'NO'}  "
                  f"paired-t p={pstr}")

if __name__ == "__main__":
    main()
