# Notebook Debug Log

One entry per fix, newest at the bottom. Format:
`## [YYYY-MM-DD HH:MM] NBnn cell N — error` with Cause / Fix / Assumptions / Verified.

## [2026-06-12 02:20] NB05 cells 10 & 18 — never executed; per-group feature parquets missing
- **Cause:** cells 10 (injury history features) and 18 (save individual feature group parquets) were added to the notebook after its last execution, so `data/processed/features/*.parquet` were never written even though `feature_matrix.parquet` existed.
- **Fix:** no code change needed — re-ran the notebook end-to-end with a fresh kernel (`run_notebooks.py --only 05 --fail-fast`, 14s). All 13 code cells executed; all 5 per-group parquets plus feature matrix and train/val/test splits written.
- **Assumptions/limitations:** none.
- **Verified:** `run_notebooks.py --only 05` passed; `verify_outputs.py --only 05` passed.
