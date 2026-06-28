# Figure Artifacts

This folder mirrors the final PDFs and PNGs used in the paper. The final PDF copies are also placed in the top-level `figures/` folder for manuscript upload.

Regenerate the AP50-vs-tolerance figure with:

```bash
python analysis/plot_ap50_vs_tolerance.py
```

Other figures are copied from the synchronized analysis output folder by:

```bash
python scripts/build_release_artifacts.py
```
