#!/usr/bin/env python3
"""Generate final quantitative revision figures from a frozen YOLOv8s generation."""

import argparse
import csv
import glob
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.analyze_generation import evaluate_detection_payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation-root",
        type=Path,
        default=PROJECT_ROOT / "revised" / "generations" / "access_r1_g2",
        help="Frozen access_r1_g2 generation containing prediction exports and training results.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "figures",
        help="Directory for PDF and 300-DPI PNG outputs.",
    )
    return parser.parse_args()


ARGS = parse_args()
GEN = str(ARGS.generation_root.resolve())
OUT = str(ARGS.output_dir.resolve())
os.makedirs(OUT, exist_ok=True)

SEEDS = [42, 43, 44]
GRID = [round(0.05 * i, 2) for i in range(1, 20)]   # 0.05 .. 0.95

VARIANTS = ["baseline", "p1_clahe", "p2_illumination", "p3_unsharp",
            "a1_crop", "a2_colorjitter", "p4_a4_combined"]
LABEL = {"baseline": "Baseline", "p1_clahe": "P1 CLAHE",
         "p2_illumination": "P2 illumination", "p3_unsharp": "P3 unsharp",
         "a1_crop": "A1 crop", "a2_colorjitter": "A2 colour jitter",
         "p4_a4_combined": "P4+A4 combined"}
# Colour-blind-safe Okabe-Ito palette.
COLOR = {"baseline": "#000000", "p1_clahe": "#E69F00", "p2_illumination": "#56B4E9",
         "p3_unsharp": "#009E73", "a1_crop": "#D55E00", "a2_colorjitter": "#CC79A7",
         "p4_a4_combined": "#0072B2"}
MARKER = {"baseline": "o", "p1_clahe": "s", "p2_illumination": "^", "p3_unsharp": "v",
          "a1_crop": "D", "a2_colorjitter": "P", "p4_a4_combined": "X"}

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8, "axes.linewidth": 0.7, "pdf.fonttype": 42,
    "legend.frameon": True, "legend.framealpha": 0.92, "legend.fontsize": 6.5,
})

_cache = {}

def load(dataset, split, variant, seed):
    key = (dataset, split, variant, seed)
    if key in _cache:
        return _cache[key]
    pat = f"{GEN}/predictions/{dataset}/{split}/{variant}_seed{seed}_*.json"
    hits = glob.glob(pat)
    if not hits:
        _cache[key] = None
        return None
    _cache[key] = json.load(open(hits[0]))
    return _cache[key]


def ap50_at(data, tau):
    return float(evaluate_detection_payload(data, threshold=tau, iou=0.50)["mAP50"])


def recall_at(data, tau):
    return float(evaluate_detection_payload(data, threshold=tau, iou=0.50)["recall"])


def precision_at(data, tau):
    return float(evaluate_detection_payload(data, threshold=tau, iou=0.50)["precision"])


def clean_fp_at(data, tau):
    """Return clean-image false-positive rate and false positives per image."""
    clean = [im for im in data["images"] if im.get("is_knot_free")]
    if not clean:
        return 0.0, 0.0
    hit = sum(1 for im in clean
              if any(p["conf"] >= tau for p in im.get("predictions", [])))
    nfp = sum(sum(1 for p in im.get("predictions", []) if p["conf"] >= tau)
              for im in clean)
    return hit / len(clean), nfp / len(clean)


def clean_maxconf(data):
    return [max((p["conf"] for p in im.get("predictions", [])), default=0.0)
            for im in data["images"] if im.get("is_knot_free")]


def select_tau(dataset, clean_split_ds, clean_split, eps=0.0):
    """Select the lowest grid threshold whose mean validation FP rate is at most eps."""
    out = {}
    for v in VARIANTS:
        chosen = GRID[-1]
        for tau in GRID:
            rates = []
            for s in SEEDS:
                d = load(clean_split_ds, clean_split, v, s)
                if d is None: rates = None; break
                rates.append(clean_fp_at(d, tau)[0])
            if rates and (sum(rates)/len(rates)) <= eps + 1e-12:
                chosen = tau; break
        out[v] = chosen
    return out


def style(ax, x0=None, x1=None, y0=0.0, y1=None):
    ax.grid(alpha=0.22, lw=0.5)
    for sp in ax.spines.values():
        sp.set_linewidth(0.7)
    ax.margins(x=0, y=0)
    if x0 is not None: ax.set_xlim(x0, x1)
    if y1 is not None: ax.set_ylim(y0, y1)


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"    -> {OUT}/{name}.pdf")


# =============================================================================
print("=" * 74)
print("VALIDATION-SELECTED THRESHOLDS (epsilon = 0)")
print("=" * 74)
TAU_VN  = select_tau("vnwoodknot", "vnwoodknot", "val", 0.0)
TAU_VSB = select_tau("vsb_rarefirst", "vsb_strict_clean", "val", 0.0)
print(f"{'variant':20} {'tau VN':>8} {'tau VSB':>9}")
for v in VARIANTS:
    print(f"  {LABEL[v]:20} {TAU_VN[v]:>6}   {TAU_VSB[v]:>7}")

# =============================================================================
print("\n" + "=" * 74)
print("TEST RETAINED RECALL / AP50 AT VALIDATION-SELECTED THRESHOLDS")
print("=" * 74)
print(f"{'variant':20} {'tau':>5} {'recall':>16} {'AP50':>16}")
for v in VARIANTS:
    rs, aps = [], []
    for s in SEEDS:
        d = load("vnwoodknot", "test", v, s)
        rs.append(recall_at(d, TAU_VN[v])); aps.append(ap50_at(d, TAU_VN[v]))
    print(f"  {LABEL[v]:20} {TAU_VN[v]:>5} "
          f"{np.mean(rs):>8.3f}+/-{np.std(rs,ddof=1):<6.3f} "
          f"{np.mean(aps):>8.3f}+/-{np.std(aps,ddof=1):<6.3f}")

# =============================================================================
print("\n" + "=" * 74); print("GENERATING FIGURES"); print("=" * 74)

# Detection performance by threshold.
print("  [2] detection performance vs threshold")
fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.3))
for ax, (fn, ti) in zip(axes, [(ap50_at, "AP50"), (precision_at, "Precision"),
                               (recall_at, "Recall")]):
    for v in VARIANTS:
        ys = np.array([[fn(load("vnwoodknot", "test", v, s), t) for t in GRID] for s in SEEDS])
        m, sd = ys.mean(0), ys.std(0, ddof=1)
        ax.plot(GRID, m, color=COLOR[v], lw=1.2, label=LABEL[v])
        ax.fill_between(GRID, m - sd, m + sd, color=COLOR[v], alpha=0.11, lw=0)
    ax.set_xlabel("Confidence threshold"); ax.set_title(ti, fontsize=8.5)
    style(ax, 0.05, 0.95, 0.0, 1.0)
axes[0].legend(loc="lower left", ncol=1)
fig.tight_layout(pad=0.5); save(fig, "detection_performance_vs_threshold")

# Absolute clean-image false-alarm counts with an exact-binomial reference band.
print("  [3] clean false alarms, absolute counts")
from scipy.stats import beta as _beta
def cp_hi(k, n, a=0.05):
    return 1.0 if k >= n else float(_beta.ppf(1 - a/2, k + 1, n - k))

fig, ax = plt.subplots(figsize=(3.6, 2.8))
N = 75
for v in VARIANTS:
    cnt = np.array([[round(clean_fp_at(load("vnwoodknot","test",v,s), t)[0] * N)
                     for t in GRID] for s in SEEDS]).mean(0)
    ax.plot(GRID, cnt, color=COLOR[v], lw=1.2, marker=MARKER[v], ms=3, label=LABEL[v])
# Clopper-Pearson reference for 0/75.
ax.axhspan(0, cp_hi(0, N) * N, color="0.6", alpha=0.16, lw=0, zorder=0)
ax.axhline(cp_hi(0, N) * N, color="0.45", lw=0.8, ls=":", zorder=1)
ax.set_xlabel("Confidence threshold")
ax.set_ylabel("Clean images flagged (of 75)")
ax.set_ylim(0, 4.4)
ax.legend(loc="upper right", ncol=1, fontsize=6)
style(ax)
fig.tight_layout(pad=0.4); save(fig, "false_positive_behavior_vs_threshold")

# Retained recall with validation-selected operating points.
print("  [5] operational selection")
fig, ax = plt.subplots(figsize=(3.6, 2.8))
for v in VARIANTS:
    r = np.array([[recall_at(load("vnwoodknot","test",v,s), t) for t in GRID]
                  for s in SEEDS]).mean(0)
    ax.plot(GRID, r, color=COLOR[v], lw=1.2, label=LABEL[v], zorder=2)
    j = GRID.index(TAU_VN[v])
    ax.plot(TAU_VN[v], r[j], marker=MARKER[v], ms=8, color=COLOR[v],
            markeredgecolor="k", markeredgewidth=0.9, zorder=6, clip_on=False)
ax.set_xlabel("Confidence threshold")
ax.set_ylabel("Retained recall")
ax.legend(loc="upper right", ncol=1, fontsize=6)
style(ax, 0.05, 0.95, 0.0, 1.0)
fig.tight_layout(pad=0.4); save(fig, "operational_selection_recall_fp_tradeoff")

# Clean-image maximum-confidence CDF.
print("  [7] clean max-confidence CDF")
fig, ax = plt.subplots(figsize=(3.5, 2.6))
for v in VARIANTS:
    vals = sorted(x for s in SEEDS for x in clean_maxconf(load("vnwoodknot", "test", v, s)))
    ax.step(vals, np.arange(1, len(vals) + 1) / len(vals), where="post",
            color=COLOR[v], lw=1.2, label=LABEL[v])
ax.set_xlabel("Per-clean-image maximum confidence"); ax.set_ylabel("CDF")
ax.axvline(0.10, color="0.55", lw=0.7, ls=":", zorder=1)
ax.legend(loc="lower right"); style(ax, 0.0, 0.10, 0.0, 1.005)
fig.tight_layout(pad=0.4); save(fig, "clean_max_confidence_cdf")

# Reliability diagram.
print("  [8] reliability diagram")
fig, ax = plt.subplots(figsize=(3.5, 2.9))
ax.plot([0, 1], [0, 1], ls="--", lw=0.8, color="0.45", label="Perfect calibration")
for v in VARIANTS:
    xs, ys = [], []
    for b in range(10):
        lo, hi = b / 10, (b + 1) / 10
        c, t = [], []
        for s in SEEDS:
            for im in load("vnwoodknot", "test", v, s)["images"]:
                if im.get("is_knot_free"): continue
                for p in im.get("predictions", []):
                    if lo <= p["conf"] < hi:
                        c.append(p["conf"]); t.append(int(p.get("validator_tp_mask", 0)) & 1)
        if len(c) >= 10:
            xs.append(np.mean(c)); ys.append(np.mean(t))
    ax.plot(xs, ys, color=COLOR[v], lw=1.1, marker=MARKER[v], ms=3, label=LABEL[v])
ax.set_xlabel("Mean confidence"); ax.set_ylabel("Empirical precision")
ax.legend(loc="upper left"); style(ax, 0.0, 1.0, 0.0, 1.0)
fig.tight_layout(pad=0.4); save(fig, "reliability_curve")

# Retained AP50 by tolerated clean false-alarm rate.
print("  [9] AP50 vs tolerance (both datasets)")
EPS = [0.0, 0.01, 0.02, 0.05]
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.7), sharey=False)
for ax, (ds, cds, csp, ttl) in zip(axes, [
        ("vnwoodknot", "vnwoodknot", "val", "VNWoodKnot (production line)"),
        ("vsb_rarefirst", "vsb_strict_clean", "val", "VSB (curated benchmark)")]):
    for v in VARIANTS:
        ys = []
        for e in EPS:
            tau = select_tau(ds, cds, csp, e)[v]
            ys.append(np.mean([ap50_at(load(ds, "test", v, s), tau) for s in SEEDS]))
        ax.plot(range(len(EPS)), ys, color=COLOR[v], lw=1.3,
                marker=MARKER[v], ms=4, label=LABEL[v])
    ax.set_xticks(range(len(EPS))); ax.set_xticklabels([f"{e:g}" for e in EPS])
    ax.set_xlabel(r"clean false-alarm tolerance $\epsilon$")
    ax.set_title(ttl, fontsize=8.5); style(ax, 0, len(EPS)-1, 0.0, 1.0)
    ax.set_clip_on(False)
    for ln in ax.get_lines(): ln.set_clip_on(False)
axes[0].set_ylabel("retained AP50")
axes[1].legend(loc="lower right", ncol=1)
fig.tight_layout(pad=0.5); save(fig, "ap50_vs_tolerance_vnwk_vsb")

# Training convergence.
print("  [A] convergence curves")
fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.6), sharey=True)
for ax, ds, ttl in zip(axes, ["vnwoodknot", "vsb_rarefirst"],
                       ["VNWoodKnot", "VSB rare-first"]):
    for v in VARIANTS:
        cur = []
        for s in SEEDS:
            hits = glob.glob(f"{GEN}/multiseed/{ds}/per_seed/runs/{v}_seed{s}/**/results.csv",
                             recursive=True)
            if not hits: continue
            rows = list(csv.DictReader(open(hits[0])))
            col = next((c for c in rows[0] if "mAP50" in c and "50-95" not in c), None)
            if col: cur.append([float(r[col]) for r in rows if r.get(col, "").strip()])
        if cur:
            n = min(len(c) for c in cur)
            ax.plot(range(1, n + 1), np.array([c[:n] for c in cur]).mean(0),
                    color=COLOR[v], lw=1.1, label=LABEL[v])
    ax.axvline(40, color="0.5", ls=":", lw=0.8)
    ax.text(40.8, 0.05, "mosaic off", fontsize=6, color="0.4", rotation=90)
    ax.set_xlabel("Epoch"); ax.set_title(ttl, fontsize=8.5)
    style(ax, 1, 50, 0.0, 0.9)
axes[0].set_ylabel("Validation mAP50")
axes[1].legend(loc="lower right", ncol=1)
fig.tight_layout(pad=0.5); save(fig, "convergence_map50")

# Selected threshold by clean source-set size.
print("  [B] threshold vs clean-set size (subsampling)")
rng = random.Random(20260817)
SIZES = [25, 50, 75, 150, 300, 600, 996]
NDRAW = 200
fig, ax = plt.subplots(figsize=(3.6, 2.7))
for v in ["baseline", "p2_illumination", "a1_crop"]:
    per_seed_src = {}
    for s in SEEDS:
        d = load("vsb_strict_clean", "val", v, s)
        src = defaultdict(float)
        for im in d["images"]:
            if not im.get("is_knot_free"): continue
            sid = os.path.basename(im["canonical_id"]).split("__")[0]
            src[sid] = max(src[sid], max((p["conf"] for p in im.get("predictions", [])), default=0.0))
        per_seed_src[s] = src
    ids = sorted(per_seed_src[SEEDS[0]].keys())
    lo, md, hi = [], [], []
    for n in SIZES:
        taus = []
        for _ in range(NDRAW):
            sub = rng.sample(ids, min(n, len(ids)))
            mx = max(max(per_seed_src[s].get(i, 0.0) for i in sub) for s in SEEDS)
            taus.append(min([t for t in GRID if t > mx] or [GRID[-1]]))
        lo.append(np.percentile(taus, 2.5)); md.append(np.median(taus)); hi.append(np.percentile(taus, 97.5))
    ax.plot(SIZES, md, color=COLOR[v], lw=1.3, marker=MARKER[v], ms=3.5, label=LABEL[v])
    ax.fill_between(SIZES, lo, hi, color=COLOR[v], alpha=0.14, lw=0)
ax.set_xscale("log"); ax.set_xticks(SIZES); ax.set_xticklabels(SIZES)
ax.set_xlabel("Clean source images used for selection")
ax.set_ylabel(r"selected $\tau$  (median, 95% range)")
ax.legend(loc="lower right"); style(ax)
ax.set_ylim(0.4, 1.0)
fig.tight_layout(pad=0.4); save(fig, "threshold_vs_cleanset_size")

print("\n" + "=" * 74)
print(f"Completed. Figures written to: {OUT}/")
print("=" * 74)
