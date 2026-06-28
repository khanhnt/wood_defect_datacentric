# Deprecated Local Outputs

Early VSB clean-wood analysis outputs used 6,252 clean tiles (`6252`) because 276 rare-first empty tiles were accidentally mixed with the strict defect-free source-image pool. Those CSVs are not released as paper evidence.

The released VSB clean-wood analysis uses only the 1,992 source images with empty annotation files and yields exactly 5,976 clean tiles (`5976`). Use:

- `results/tables/vsb_clean_set_report.json`
- `results/tables/vsb_clean_threshold_sweep_summary.csv`
- `results/tables/vsb_clean_operational_selection.csv`
- `results/tables/vsb_clean_sensitivity.csv`

No deprecated 6,252-denominator CSV is required for reproduction.
