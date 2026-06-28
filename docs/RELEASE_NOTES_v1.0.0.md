# v1.0.0 Release Notes

Initial public reproducibility release for the IEEE Access paper "Beyond mAP: Negative-Aware Evaluation of Data-Centric Pipelines for Wood Knot Detection".

Included:

- YOLOv8s training and evaluation scripts for the fixed-detector data-centric study.
- Preprocessing and augmentation configs for baseline, P1, P2, P3, A1, A2, and P4+A4 variants.
- Portable split and tiling manifests for VNWoodKnot, VSB rare-first, and VSB strict clean-wood evaluation.
- CSV/JSON files behind the reported tables.
- Final figure PDFs/PNGs, including AP50-vs-tolerated-clean-FP-rate.
- Release integrity checker with table-value, VSB clean-denominator, leakage, deprecated-output, and secret/path checks.

Not included:

- Raw dataset images.
- Materialized YOLO image folders.
- Model checkpoints.
- Large per-seed prediction JSON files. These should be archived separately and linked via Zenodo DOI.

Before publishing:

1. Upload the large per-seed prediction archive to Zenodo.
2. Update `results/per_seed/README.md`, `README.md`, and `CITATION.cff` with the Zenodo DOI.
3. Run `python scripts/release_integrity_check.py`.
4. Tag the reviewed commit as `v1.0.0`.
