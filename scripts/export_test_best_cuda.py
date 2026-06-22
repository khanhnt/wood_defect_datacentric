#!/usr/bin/env python3
"""
export_test_best_cuda.py — Vast.ai RTX 3090 version.

For every run, evaluate weights/best.pt on the VAL and TEST splits and write
test_best_summary.csv.  Inference only (no training).

Robustness: instead of guessing from folder names, it reads `seed` and the dataset
`data:` path straight from each run's Ultralytics args.yaml (saved next to weights/).
The variant label is parsed from the run-folder name via VARIANT_TOKENS.

Run on the same box you trained on so val mAP matches your best_by_map50 numbers.
"""
import glob, csv, re
from pathlib import Path
import yaml
from ultralytics import YOLO

# ----------------------------- EDIT IF NEEDED -----------------------------
RUNS_GLOB = "results*/**/weights/best.pt"     # adjust to your runs location
DEVICE = "0"                                   # 3090; "0,1" not needed for val
IMGSZ, BATCH, CONF, IOU = 1024, 32, 0.001, 0.5 # 24 GB handles batch 32 at 1024
OUT = "test_best_summary.csv"
VARIANT_TOKENS = {                             # substring in run name -> label
    "p4_a4": "P4+A4 combined", "p4a4": "P4+A4 combined", "combined": "P4+A4 combined",
    "a1_crop": "A1 crop", "a1": "A1 crop",
    "a2_color": "A2 colour jitter", "a2": "A2 colour jitter",
    "p1": "P1 CLAHE", "p2": "P2 illumination", "p3": "P3 unsharp",
    "baseline": "Baseline",
}
# --------------------------------------------------------------------------


def label_variant(name: str) -> str:
    n = name.lower()
    for tok, lab in VARIANT_TOKENS.items():     # longer/more specific tokens first
        if tok in n:
            return lab
    return name


def read_args(ckpt: str):
    """best.pt is at <train_dir>/weights/best.pt; args.yaml sits in <train_dir>."""
    train_dir = Path(ckpt).parents[1]
    args_path = train_dir / "args.yaml"
    seed, data = None, None
    if args_path.exists():
        with open(args_path) as f:
            a = yaml.safe_load(f)
        seed = a.get("seed")
        data = a.get("data")
    return seed, data, train_dir.name


def run_val(ckpt, data, split):
    m = YOLO(ckpt)
    return m.val(data=data, split=split, imgsz=IMGSZ, batch=BATCH,
                 conf=CONF, iou=IOU, device=DEVICE, plots=False, verbose=False)


rows = []
ckpts = sorted(glob.glob(RUNS_GLOB, recursive=True))
print(f"found {len(ckpts)} checkpoints\n")
for ckpt in ckpts:
    seed, data, name = read_args(ckpt)
    variant = label_variant(name)
    ds = "vsb_rarefirst" if "vsb" in name.lower() else "vnwoodknot"
    if data is None:
        print(f"[skip] no data path in args.yaml: {name}")
        continue
    for split in ("val", "test"):
        try:
            r = run_val(ckpt, data, split)
        except Exception as e:
            print(f"[error] {name} ({split}): {e}")
            continue
        rows.append(dict(dataset=ds, run=name, variant=variant, seed=seed, split=split,
                         precision=round(float(r.box.mp), 4),
                         recall=round(float(r.box.mr), 4),
                         mAP50=round(float(r.box.map50), 4),
                         mAP50_95=round(float(r.box.map), 4)))
        print(f"[ok] {name:42s} {split:4s} seed={seed} {variant:18s} mAP50={r.box.map50:.4f}")

if not rows:
    raise SystemExit("No checkpoints matched RUNS_GLOB — check the path.")
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader(); w.writerows(rows)
print(f"\nwrote {OUT} ({len(rows)} rows = {len(rows)//2} runs x 2 splits)")
