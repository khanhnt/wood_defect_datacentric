# Figure Artifacts

This folder mirrors the final PDFs and PNGs used in the accompanying manuscript. The top-level `figures/` directory contains the same manuscript-ready assets.

Regenerate the final quantitative figures from a restored frozen YOLOv8s generation with:

```bash
python scripts/generate_manuscript_figures.py \
  --generation-root /path/to/yolo_generation_v1 \
  --output-dir /path/to/output_figures
```

Dataset-sample and detection-scenario figures are generated separately by
`scripts/fig_dataset_samples.py` and `scripts/fig_detection_scenarios.py` because
they require the source images and checkpoints.
