# YOLOv8s frozen-generation artifacts

Source generation: `access_r1_g2`.

The seven variants are Baseline, P1 CLAHE, P2 illumination normalization, P3
unsharp masking, A1 defect-preserving crop, A2 texture-aware colour jitter, and
P4+A4 combined. All reported standard metrics use common non-augmented evaluation
splits. Negative-aware thresholds are selected on validation clean material and
applied unchanged to the held-out test clean material.

The `deprecated_vs_corrected_*.csv` files quantify the effect of replacing the old
augmentation-materialized validation protocol. They are audit artifacts and are not
the primary result tables.
