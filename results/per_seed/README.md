# Per-Seed Prediction Archive

Large per-seed prediction JSON files are not stored in GitHub.

The lightweight GitHub release does not require this archive. To repeat matching,
threshold selection, or bootstrap calculations from detections, restore the frozen
generation exports and either preserve their original layout or point each analysis
script at the restored location.

The YOLOv8s generation uses this layout:

```text
predictions/
  vnwoodknot/{val,test}/
  vsb_rarefirst/{val,test}/
  vsb_strict_clean/{val,test}/
```

Each directory contains one JSON export per variant and seed. The Faster R-CNN archive
uses `predictions/vnwoodknot/{val,test}/` for its nine runs.

The GitHub release contains the derived CSV/JSON tables under `results/tables/`. Those
files are enough to inspect every reported table value. Download the prediction archive
only to rerun analyses from raw detections.

The external archive DOI will be added here after deposition. Until then, the
aggregate artifacts in `results/tables/` remain the citable GitHub release content.
