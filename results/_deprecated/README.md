# Deprecated Local Outputs

Early VSB clean-wood analysis outputs used 6,252 clean tiles (`6252`) because 276 rare-first empty tiles were accidentally mixed with the strict defect-free source-image pool. Those CSVs are not released as paper evidence.

The revised analysis uses only the 1,992 source images with empty annotation files and yields exactly 5,976 clean tiles (`5976`). These are partitioned by source ID into disjoint 2,988-tile validation and test halves. Use:

- `data/vsb_clean_manifest/clean_tile_manifest.csv`
- `docs/runbook_test_evidence/vsb_clean_partition_report.json`
- `results/tables/yolov8s/clean_fp_sweep_summary.csv`
- `results/tables/yolov8s/locked_test_operating_points_summary.csv`
- `results/tables/yolov8s/locked_test_sensitivity_summary.csv`

No deprecated 6,252-denominator CSV is required for reproduction.
