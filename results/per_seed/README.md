# Per-Seed Prediction Archive

Large per-seed prediction JSON files are not stored in GitHub.

Expected archive layout after download:

```text
results/per_seed/
  vnwoodknot/
    baseline_seed42_predictions.json
    ...
    p4_a4_combined_seed44_predictions.json
  vsb_clean/
    baseline_seed42_predictions.json
    ...
    p4_a4_combined_seed44_predictions.json
```

The GitHub release contains the derived CSV/JSON tables under `results/tables/`. Those files are enough to inspect all reported table values and regenerate the paper figures. Download the per-seed archive only if you want to rerun threshold matching and bootstrap analysis from raw detections.

Zenodo DOI: `TBD`

After minting the archive DOI, update this file, `README.md`, and `CITATION.cff`.
