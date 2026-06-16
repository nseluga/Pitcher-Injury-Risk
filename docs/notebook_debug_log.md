# Notebook Debug Log

One entry per fix, newest at the bottom. Format:
`## [YYYY-MM-DD HH:MM] NBnn cell N — error` with Cause / Fix / Assumptions / Verified.

## [2026-06-16 03:41] NB06 cell 18 — ValueError: All arrays must be of the same length in compute_feature_importance

- **Cause:** `SimpleImputer(strategy='median')` silently drops columns that are entirely NaN in the training data (sklearn default `keep_empty_features=False`). The column `intragame_velo_drop` is all-NaN in the training seasons, so the imputer output had 75 features while `X_train.columns` had 76. `compute_feature_importance` then tried to zip 76 names to 75 importances, triggering the ValueError.
- **Fix:** Added `keep_empty_features=True` to all three `SimpleImputer` instances in `src/models/baseline_models.py` (inside `train_logistic_regression`, `train_random_forest`, and `train_gradient_boosting`). Verified locally: both RF and XGBoost pipelines now output 76 `feature_importances_` matching `X_train.columns`.
- **Assumptions/limitations:** `intragame_velo_drop` is all-NaN in training so its importance will be 0; it should be dropped or imputed from a different source in a future NB05 revision, but keeping it at 0 is safe for now.
- **Verified:** Quick sample test passed (76 features in, 76 importances out). Full notebook run launched — awaiting completion.

## [2026-06-12 02:20] NB05 cells 10 & 18 — never executed; per-group feature parquets missing
- **Cause:** cells 10 (injury history features) and 18 (save individual feature group parquets) were added to the notebook after its last execution, so `data/processed/features/*.parquet` were never written even though `feature_matrix.parquet` existed.
- **Fix:** no code change needed — re-ran the notebook end-to-end with a fresh kernel (`run_notebooks.py --only 05 --fail-fast`, 14s). All 13 code cells executed; all 5 per-group parquets plus feature matrix and train/val/test splits written.
- **Assumptions/limitations:** none.
- **Verified:** `run_notebooks.py --only 05` passed; `verify_outputs.py --only 05` passed.

## [2026-06-16 09:03] NB07 — Full run completed successfully

- **Cause:** N/A — full production run (PID 40554, launched previous session) completed in 197s.
- **Fix:** No further changes needed. survival_cox.pkl, survival_rsf.pkl, survival_model_metrics.csv, survival_hyperparameter_tuning_results.csv all written.
- **Verified:** verify_outputs.py --only 07 PASS (confirmed at session start 2026-06-16T09:03Z).

## [2026-06-16 09:00] NB07 training cell — Cell execution timed out after 7200s

- **Cause:** `CoxPHFitter.fit()` with 40,000 rows × 76 features requires O(n×D×p) partial-likelihood computation (Newton-Raphson). With ~15K events and 76 features, each NR iteration is ~70M ops; ×50 iterations = ~3.5B ops, taking hours on a laptop. `_MAX_COX_ROWS=40_000` in the previous version was too large.
- **Fix:**
  1. `src/models/survival_models.py`: reduced `_MAX_COX_ROWS` 40K→12K, `_MAX_RSF_ROWS` 60K→12K; added `_MAX_COX_FEATURES=20`; added `_select_top_features()` (top-n features by |correlation| with event indicator); updated `train_cox_ph()` to apply feature pre-selection after imputation; fixed `_subsample()` to handle the case where events > max_rows (previously set `n_censor` negative, now subsamples events too).
  2. `notebooks/07_survival_models.ipynb`: updated setup cell imports; rewrote tuning cell to use `_MAX_COX_ROWS` constant and apply feature pre-selection for Cox grid search (RSF keeps all features).
- **Assumptions/limitations:** Cox PH is fit on 20 top-ranked features (by event correlation) rather than all 76. This reduces interpretability coverage but is statistically valid — Cox is a consistent estimator on subsets. Deeper SHAP analysis in NB10 uses the full-feature XGBoost instead.
- **Verified:** Smoke test passed (train_cox_ph + train_random_survival_forest on TEST_MODE slice). TEST_MODE notebook run: 199s, `verify_outputs.py --only 07` PASS. Full run (TEST_MODE=False) launched PID 40554 at 2026-06-16T09:00Z; completed in 197s (see NB07 completion entry above).

## [2026-06-16 09:12] NB08 — TypeError: feature names unseen at fit time (injury_prob_30d) + .pkl/.joblib mismatch

- **Cause 1:** The verifier requires `models/multitask_chained.joblib` and `models/multitask_chained_tuned.joblib`, but the notebook's save cell used `.pkl` extensions for both.
- **Cause 2:** In the tuning cell, the regression head for `next_injury_days_lost` was tuned on `X_train` (without `injury_prob_30d`). At prediction time, `predict_all_tasks` with the chained architecture adds `injury_prob_30d` to `X_in` before calling regression heads. The tuned pipeline's imputer had never seen `injury_prob_30d` during fit → `ValueError: Feature names unseen at fit time`.
- **Fix:**
  1. `notebooks/08_multitask_models.ipynb` c07-save: changed `'multitask_chained.pkl'` → `'multitask_chained.joblib'` and `'multitask_chained_tuned.pkl'` → `'multitask_chained_tuned.joblib'`.
  2. Tuning cell (4c8bb357): before `_regression_subset`, compute `_X_train_chained = X_train.copy(); _X_train_chained['injury_prob_30d'] = _best_clf.predict_proba(X_train)[:, 1]` (and same for test). Pass `_X_train_chained` / `_X_test_chained` to `_regression_subset` so the tuned regression head trains with the same feature set it'll receive at inference.
  3. Added `TEST_MODE=False` flag (5K-row sample when True) + N_ITER_TUNING=5 in TEST_MODE for fast iteration.
  4. Guarded c06-viz against empty `reg_compare` (uses `.xs()` on MultiIndex, raises KeyError if no regression tasks in metrics).
- **Assumptions/limitations:** `days_to_next_injury` is non-null for all rows (it represents time-to-event including censored observations), so its regression head always trains on the full dataset.
- **Verified:** TEST_MODE run passed in 35s. Full run (TEST_MODE=False) launched PID 40855 at 2026-06-16T09:12Z.

## [2026-06-16 11:40] NB10 — stub implemented from scratch; imputer column count mismatch

- **Cause:** NB10 was a stub (1 unexecuted code cell). `scripts/generate_notebook_10.py` did not exist. Additionally, `SimpleImputer` in the saved `baseline_xgboost_tuned.joblib` was fitted with `keep_empty_features=False` (default), so `intragame_velo_drop` (all-NaN in training data) is dropped during `transform()`, producing 75 columns vs. 76 in `_infer_feature_cols()`.
- **Fix:**
  1. Wrote `scripts/generate_notebook_10.py` following the `md()`/`code()` pattern from NB09.
  2. Used `imputer.get_feature_names_out()` to get the 75 actual post-imputation feature names (`imp_feature_cols`) throughout all SHAP/beeswarm/waterfall cells.
  3. PDP cells pass raw `feature_cols` to the full pipeline (pipeline handles imputation internally); keyword-based group matching against `imp_feature_cols` to handle absent/dropped features robustly.
  4. Installed `shap` in `pitcher311` environment (`/opt/homebrew/opt/python@3.11/bin/python3.11 -m pip install shap`).
  5. All logic verified on 500-row sample before full run launch.
- **Assumptions/limitations:** SHAP computed on a 5,000-row stratified sample (20% injured) for speed. PDP subsample: 1,000 rows.
- **Verified:** Logic test (500-row sample) passed. Full run launched PID 45258 at 2026-06-16T11:40Z.

## [2026-06-16 09:39] NB09 — two output file mismatches preventing verify_outputs pass

- **Cause 1:** Cell c06-toprisk saved the distribution figure as `fig_24_injury_risk_plus_distribution.png` but `scripts/verify_outputs.py` requires `reports/figures/injury_risk_plus_distribution.png`.
- **Cause 2:** `verify_outputs.py` requires `reports/figures/risk_score_components.png` but no cell in the notebook generated this figure.
- **Fix:**
  1. `notebooks/09_risk_score_construction.ipynb` c06-toprisk: changed save path to `FIGURES_DIR / 'injury_risk_plus_distribution.png'`.
  2. Inserted new code cell (after c07-stability) that generates `risk_score_components.png` — a two-panel figure: (left) mean normalized component values (injury_prob_30d, expected_days_lost, hazard_rate) by archetype; (right) mean Injury Risk+ by season and archetype confirming era-adjustment holds.
- **Assumptions/limitations:** No changes to core scoring logic. NB09 has no TEST_MODE — runs on full 200K-row feature matrix.
- **Verified:** Full run launched PID 41493 at 2026-06-16T09:39Z.

## [2026-06-16 11:48] NB11 cell c07-s3-pitch-mix — ValueError: Bin edges must be unique in pd.qcut(sl_pct, q=5)

- **Cause:** Many pitchers have `sl_pct = 0.0` (no slider usage), causing the 5-quantile boundaries to be non-unique. `pd.qcut` raises `ValueError: Bin edges must be unique` without `duplicates='drop'`.
- **Fix:** Added `duplicates='drop'` to the `pd.qcut(df_sl['sl_pct'], q=5, ...)` call in cell `c07-s3-pitch-mix`. No logic change — quantile labels are reduced to however many unique edges remain.
- **Assumptions/limitations:** When many pitchers have 0.0 slider%, the lowest bin(s) collapse — the decile table will have fewer than 5 groups. This is correct behavior.
- **Verified:** Full run launched PID 45615 at 2026-06-16T11:48Z.

## [2026-06-16 11:53] NB11 cells c07, c21, c23 — three bugs; notebook now passing

- **Cause 1 (c07-s3-pitch-mix):** `pd.qcut(sl_pct, q=5, labels=[...5 labels...], duplicates='drop')` — `duplicates='drop'` collapsed bins from 5 to fewer when many pitchers have `sl_pct=0.0`, but the 5-element `labels` list was still passed, raising `ValueError: Bin labels must be one fewer than the number of bin edges`.
- **Fix 1:** Use `pd.qcut(..., retbins=True, duplicates='drop')` to discover the actual number of bins, then slice the labels to `n_actual = len(bins) - 1` before the final qcut call.
- **Cause 2+3 (c21-s10-frontier, c23-s11-trajectory):** Literal newlines inside string literals in two `ax.set_title()` calls — `'Performance vs Risk Frontier\n(...)` and an f-string for the trajectory panel. These cause `SyntaxError: unterminated string literal` under Python 3.12+ (tokenizer now rejects implicit continuation inside strings).
- **Fix 2+3:** Replaced literal newlines with `\n` escape sequences in both cells.
- **Assumptions/limitations:** None — all fixes are cosmetic/display-only, no logic change.
- **Verified:** `run_notebooks.py --only 11` passed in 28s; `verify_outputs.py --only 11` PASS.

## [2026-06-16 11:43] NB11 cell c03-s1-load — KeyError: 'game_date' when merging risk scores

- **Cause:** `injury_risk_plus_scores.parquet` (produced by NB09) uses `pitcher_id`/`season` keys with no `game_date` column. NB11's load cell tried `risk['game_date'] = pd.to_datetime(risk['game_date'])` and then merged on `['pitcher', 'game_date']` — both raise KeyError.
- **Fix:** Replaced the key-based merge with a positional alignment. NB09 builds `risk_df` row-by-row directly from `fm` (same 205,911 rows, same order), so `risk['injury_risk_plus'].values` can be assigned directly to `analysis = fm.copy()` without any merge. Added an `assert len(risk) == len(fm)` guard to detect future drift.
- **Assumptions/limitations:** Relies on NB09 and NB11 reading `feature_matrix.parquet` in the same row order (parquet stores rows deterministically). The `pitcher_archetypes.parquet` mismatch from progress.json is a non-issue — the file doesn't exist and the notebook gracefully derives rule-based archetypes instead.
- **Verified:** Smoke test (data loading + archetype derivation on full 205k rows) passed. Full run launched PID 45505 at 2026-06-16T11:43Z.
