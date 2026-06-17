# Model Critique Log

One entry per notebook, appended after each Phase 2 critique session.
Format: research findings → decisions critiqued → improvements implemented → verification.

<!-- Entries will be appended below as each notebook is critiqued -->

## [2026-06-16] NB05 — Feature Engineering: Critique & Improvements

### Research Findings

1. **[Logue et al. 2021, PubMed 34189147]** Analyzed 223 MLB pitchers pre-Tommy John surgery. Found significant velocity *decreases* in 4-seam FBs, 2-seam FBs, and sliders in the ~15 games before surgery, plus a significant *negative* spin rate trend on 4-seam fastballs. This directly motivates (a) a 14-day velocity rolling window and (b) slider-specific spin rate tracking.

2. **[Logue et al. 2025, PMC12717397]** Acute UCL injuries study — tracked 5 Statcast metrics: velocity, spin rate, release extension, arm angle, and acceleration magnitude. Identified per-pitch baseline comparison as the most sensitive detection method. Confirms that release point tracking (extension, arm angle) and spin rate are the strongest leading indicators.

3. **[ACWR review, PMC7534929]** Baseball-specific ACWR review confirms the 7-day acute / 28-day chronic window is the industry standard. Threshold for elevated risk is ACWR ≥1.27 (not 1.5 as in field sports). Athletes with ACWR outside the 0.7–1.3 range are ~8× more likely to sustain a throwing-overuse injury.

4. **[Velocity/pitch type UCL paper, PubMed 26995458]** Sliders with higher spin rate and fastballs thrown at higher velocity were independently associated with UCL surgery risk. This motivated adding `sl_spin_mean` and `sl_spin_delta_30d` as explicit features, which were missing from the original implementation.

5. **[Release point variability, PMC11608975 / arXiv 2603.04864]** Release point range and CV features contribute 33–39% of total model importance in injury risk models, consistently outranking mean values. Our within-game `rel_x_std` / `rel_z_std` captures this, but repeated exposure to extreme mechanics (top-decile pitches) may be an even stronger signal. Not implemented now (requires per-pitch percentile — future work).

### Decisions Critiqued

- **ACWR window (7:28):** Confirmed as the correct industry standard for baseball. **Verdict: no change — current approach matches literature.**

- **ACWR denominator (pitches_28d / 4):** Divides by 4 to express chronic load as a weekly average. This is the standard rolling-ACWR formulation. Exponentially-weighted ACWR (ewACWR) is an alternative endorsed by some researchers but adds complexity without strong evidence of superiority in baseball-specific studies. **Verdict: no change for now; ewACWR flagged as future enhancement.**

- **`intragame_velo_drop` computation:** Used `df[inning_col].max()` globally to identify "late innings," so the "late" window was innings ≥ (global_max − 1). For starters exiting in inning 5–7, the global max (typically 9) meant zero pitches qualified as "late," making this feature NaN for most rows. **Verdict: BUG — fixed by computing per-game max inning.**

- **Velocity rolling windows (7/30/90 days):** The Logue et al. pre-surgery window is ~15 games for starters (~14 calendar days). The 7d window is too short for starters on 5-day rest; 30d is too long to isolate the acute decline. A 14-day window fills this gap. **Verdict: added 14-day window.**

- **Slider spin rate feature:** Literature directly links slider spin rate to UCL injury risk (Logue et al. 2021, PubMed 26995458). Movement module only tracked `fb_spin_mean`. **Verdict: added `sl_spin_mean`, `sl_spin_mean_30d_avg`, `sl_spin_delta_30d`.**

- **Pitch mix entropy:** Shannon entropy of pitch distribution is present. No direct validation in the peer-reviewed literature for entropy as a standalone predictor, but pitch mix change is validated (curveball % increases before TJ surgery per Logue et al.). The entropy feature captures total mix variance; individual pitch-type delta features (`sl_pct_delta_30d`, etc.) are more interpretable. **Verdict: keep entropy; the per-type delta features already exist.**

- **Injury history features:** Prior IL count, days since last injury, days lost prior — all well-supported by literature as strongest-known predictors. **Verdict: no change — current approach matches literature.**

### Improvements Implemented

1. **Bug fix — `intragame_velo_drop`** (`src/features/velocity_features.py`): Changed `df[inning_col].max()` (global) to a per-pitcher-game max inning join. Pitchers who exit in inning 6 now get a valid late-inning reading instead of NaN. This affects all downstream models that use this feature.

2. **New feature — 14-day velocity window** (`src/features/velocity_features.py`): Added `fb_velo_14d_avg` window to `build_velocity_features()`. Directly aligned with the Logue et al. pre-surgery analysis window.

3. **New features — slider spin rate** (`src/features/movement_features.py`): Added `sl_spin_mean`, `sl_spin_mean_30d_avg`, and `sl_spin_delta_30d`. `SLIDER_TYPES = {"SL", "ST"}` constant added. Pitchers who do not throw sliders will have NaN — downstream imputation handles this correctly.

### Verified

- Smoke test `.scratch/test_nb05_changes.py` passed: `intragame_velo_drop` returns non-NaN values, `fb_velo_14d_avg` present, `sl_spin_mean` / `sl_spin_delta_30d` present.
- Full notebook run: `python run_notebooks.py --only 05 --fail-fast` — PASS.
- `python scripts/verify_outputs.py --only 05` — PASS.

---

## [2026-06-16] NB06 — Baseline Models: Critique & Improvements

### Research Findings

1. **[PMC12013557 — Scoping review, 2024]** Of 15 ML sports-injury prediction studies reviewed, 10 (67%) used SMOTE to address class imbalance. ROC-AUC was the primary metric in 71% of studies, but the review notes that for severely imbalanced datasets (<10% positive rate), PR-AUC and F1 are more discriminative than ROC-AUC.

2. **[PMC11369970 — MLB Pitcher ML, 2024]** Pitch-tracking metrics study on shoulder/elbow injuries used a stratified 5-fold CV and balanced sampling. Achieved ROC-AUC of 0.84. Used tree-based models (RF, XGBoost) as primary classifiers. Confirms tree-based approaches are the right choice for this domain.

3. **[PMC10613321 — Overview of ML for sports injury prediction]** In 60% of studies, tree-based models (RF, XGBoost) provided highest predictive performance. Temporal splitting and walk-forward CV are the methodologically correct evaluation design for longitudinal athlete data. Both are implemented in NB06. ✓

4. **[arXiv 2207.00585 — UCL injury prediction, MLB rookies]** Applied oversampling (SMOTE, random, and class-weight approaches) for UCL injury prediction in MLB. Found `class_weight='balanced'` and `scale_pos_weight` competitive with SMOTE when combined with careful temporal splitting. This affirms our current class-balancing approach as valid.

5. **[BMC Sports Science 2025, PMC12964768]** For imbalanced injury prediction, recommends using PR-AUC (average precision) as the primary optimization metric because ROC-AUC can appear inflated — in a dataset with 95% negatives, even a poor model achieves high ROC-AUC by correctly predicting non-injury.

### Decisions Critiqued

- **Primary evaluation metric (auc_roc):** NB06 sorts model comparison by `auc_roc`. For ~5% positive rate injury data, PR-AUC is more discriminative — ROC-AUC includes true negative performance which dominates. Literature consensus (BMC 2025, scoping review) supports PR-AUC for severe class imbalance. **Verdict: changed sort key from `auc_roc` to `pr_auc` in model comparison table; affects best-model selection for downstream CV and multi-horizon comparison.**

- **RF GridSearchCV scoring (`roc_auc`):** Internal hyperparameter search used `scoring="roc_auc"`. This selects RF hyperparameters optimized for ROC performance, not PR-AUC. **Verdict: changed to `scoring="average_precision"` in `baseline_models.py` — selects RF parameters that better identify injured pitchers.**

- **Class balancing (`class_weight='balanced'` + `scale_pos_weight`):** Both are valid alternatives to SMOTE (arXiv 2207.00585 confirms). SMOTE on temporal pitcher data risks creating synthetic rows that span season boundaries or pitcher-identity boundaries. **Verdict: no change — current approach is appropriate; SMOTE flagged as future enhancement with caution.**

- **Hyperparameter search (RandomizedSearchCV, N_ITER=20):** FAST_TUNING mode with 20 iterations. Literature-appropriate for a laptop-constrained training loop. Walk-forward CV with 5 folds is the gold standard for temporal sports data. **Verdict: no change — current approach matches literature.**

- **Temporal splitting:** Last 2 seasons held out for test. Walk-forward CV implemented. This is the methodologically correct design per temporal sports injury literature. **Verdict: no change — current approach matches literature.**

### Improvements Implemented

1. **RF GridSearchCV scoring** (`src/models/baseline_models.py`): Changed `scoring="roc_auc"` → `scoring="average_precision"` so RF hyperparameter selection is optimized for the metric that matters for imbalanced injury data.

2. **Primary model comparison sort** (`notebooks/06_baseline_models.ipynb`, cell 12): Changed `sort_values('auc_roc')` → `sort_values('pr_auc')`. Affects `best_model_name` selection for walk-forward CV and multi-horizon comparison — the "best" model will now be the one with highest PR-AUC, which is more clinically meaningful.

### Verified

- `verify_outputs.py --only 06` — PASS (2026-06-17). Both improvements confirmed in output: sort_values by pr_auc in notebook, scoring="average_precision" in RF GridSearchCV.

---

## [2026-06-17] NB07 — Survival Models: Critique & Improvements

### Research Findings

1. **[PMC8775284 — Hazard of Arm Injury, MiLB]** Cox PH survival analysis used to compare starters vs. relievers — found starters had 2.4× higher hazard of arm injury. Confirms Cox PH is used in pitcher injury research, but does not report whether the PH assumption was tested. This is a methodological gap in the literature that we address.

2. **[Wei 1992, Statistics in Medicine — AFT models]** The Accelerated Failure Time model is a well-validated alternative to Cox PH when the proportional hazards assumption is violated. AFT directly models the time-to-event as a function of covariates (multiplicative time-scaling) rather than modeling the instantaneous hazard. Advantage: no PH assumption required.

3. **[Springer 2025, TJS prediction paper]** A classification model detects TJ surgery risk 100 days in advance (F1=0.73); a regression model estimates time remaining until last pre-surgery game (R²=0.79). Confirms that time-to-event framing is meaningful for pitcher UCL injury, and that both classification and survival-style regression are valid approaches.

4. **[scikit-survival docs — Evaluating Survival Models]** The standard dual-metric evaluation for survival models is C-index (discrimination) + Integrated Brier Score (calibration). C-index alone is insufficient because a model can rank pitchers correctly (high C-index) while producing poorly calibrated survival probabilities. IBS matters because Injury Risk+ uses the survival probabilities, not just rankings.

5. **[Comparison of PH and AFT models, CORE paper]** When the PH assumption is violated, Cox PH produces biased hazard estimates and inflated apparent C-index. The AFT Weibull is preferred because pitcher injury hazard rates are plausibly non-proportional (early-season fresh-arm risk vs. late-season accumulated-fatigue risk differ in ways that may violate the constant-HR assumption).

### Decisions Critiqued

- **Cox PH only, no AFT trained:** The notebook's purpose section mentions "Weibull AFT" as one of the three implemented models, and `train_aft_model` is imported in cell 1. However, the training cell only calls `train_cox_ph` and `train_random_survival_forest` — the AFT model is imported but never used. **Verdict: bug/gap — implemented Weibull AFT training in cell 5.**

- **PH assumption never tested:** Cox PH requires hazard ratios to be constant over time. For pitchers, early-season vs. late-season risk may violate this. The notebook fitted Cox PH without any Schoenfeld residual test or check_assumptions() call. **Verdict: added `check_ph_assumption()` call after training, using lifelines' built-in test.**

- **Feature selection for AFT model (all features):** `train_aft_model` previously fit on all available features (after imputation and zero-variance dropping), which could be very slow or numerically unstable with 100+ features. Cox PH had a pre-selection cap (`n_features=20`) but AFT did not. **Verdict: added identical `n_features` parameter and `_select_top_features` pre-filter to `train_aft_model` for consistency and stability.**

- **Feature selection method (univariate correlation):** `_select_top_features` selects top-N features by absolute correlation with the event indicator. This misses non-linear relationships (e.g. ACWR has a U-shaped relationship with injury risk, which has near-zero linear correlation). Better methods: mutual information or LASSO. **Verdict: noted as limitation; current approach is acceptable for a laptop-constrained pipeline and avoids introducing another library dependency. Future enhancement: mutual information filter.**

- **RSF hyperparameters (max_depth=6, n_estimators=100):** These are reasonable defaults and tuning is present in the notebook. **Verdict: no change — current approach matches literature.**

- **Model evaluation sort by C-index:** The notebook correctly uses C-index as the primary ranking metric. IBS is computed but not used in model selection. Since Injury Risk+ uses survival probabilities (calibration matters), best_model_name could also consider IBS. **Verdict: acceptable for now — C-index is the standard primary metric; IBS is reported and provides calibration context.**

### Improvements Implemented

1. **Weibull AFT training** (`notebooks/07_survival_models.ipynb`, cell 5): Added `survival_models['aft_weibull'] = train_aft_model(...)` call. The AFT model is now trained alongside Cox PH and RSF, evaluated in the same metrics loop, and its C-index and IBS appear in the comparison table.

2. **PH assumption test** (`notebooks/07_survival_models.ipynb`, cells 6–7 new): Added a markdown section and code cell that calls `check_ph_assumption(survival_models['cox_ph'])`, running Schoenfeld residual tests for all Cox features. Features that violate proportionality are flagged in output.

3. **Feature pre-selection for AFT** (`src/models/survival_models.py`, `train_aft_model`): Added `n_features: int = _MAX_COX_FEATURES` parameter and the same `_select_top_features` pre-filter used by Cox PH. Prevents fitting failures on high-dimensional inputs.

4. **`_fit_df_` stash on Cox model** (`src/models/survival_models.py`, `train_cox_ph`): Added `model._fit_df_ = fit_df.copy()` after fitting so `check_ph_assumption()` can run the Schoenfeld test without re-running preprocessing. Memory cost is bounded: at most 12,000 rows × 20 features.

5. **`check_ph_assumption()` helper** (`src/models/survival_models.py`): New function wrapping `lifelines CoxPHFitter.check_assumptions()` with graceful error handling and clear output labels.

### Verified

- TEST_MODE run: `python run_notebooks.py --only 07 --fail-fast` — PASS (113s).
- Full run: `python run_notebooks.py --only 07 --fail-fast` — PASS (113s).
- `python scripts/verify_outputs.py --only 07` — PASS.

---

## [2026-06-17] NB08 — Multi-Task Models: Critique & Improvements

### Research Findings

1. **[IJSPT 2022 — Pitcher Injury Burden Prediction]** Zero-inflated negative binomial (ZINB) regression was used to predict days lost to arm injury in MiLB pitchers. RMSE of 11.9 days, R²=0.80. The ZINB model was chosen specifically because days-lost data is right-skewed (most stints are short; TJ surgery cases are 365+ days) and zero-inflated (for uninjured pitchers). This directly motivates log1p-transforming `next_injury_days_lost` before regression — the closest sklearn-compatible analogue.

2. **[IJSPT 2022, same paper]** The separate elbow model (RMSE=21.3, R²=0.42) and shoulder model (RMSE=17.9, R²=0.57) performed worse than the combined arm injury model, suggesting pooling injury types is statistically correct — our single regression head over all injury types is the right choice.

3. **[Karnuta et al. 2020, Orthopaedic Journal of Sports Medicine]** ML models (RF, XGBoost) outperformed logistic regression for next-season MLB player injury prediction using performance and injury profile trends 2000–2017. Confirms tree-based multitask architecture is domain-appropriate.

4. **[ML sports injury scoping review, PMC 2024]** Chained/sequential prediction architectures (predict injury probability first, then use as a feature for downstream tasks) are a common design pattern. Missing data imputation via chained equations and SMOTE for class imbalance are recommended preprocessing steps. Our chained architecture (probability → severity → type) is consistent with literature.

5. **[RMSE for right-skewed regression targets]** RMSE is dominated by high-value outliers in right-skewed distributions. When days-lost ranges from 10 to 365+ days, a single TJ surgery case can inflate RMSE by 10+ days even with accurate median predictions. Log1p-transforming before fitting addresses this — the model optimizes mean squared error in log space, which weights relative errors equally across the full range.

### Decisions Critiqued

- **`next_injury_days_lost` regression on raw values:** The `next_injury_days_lost` target is right-skewed (most IL stints are 10–30 days, but TJ surgery creates 365-day outliers). Training a RF regressor on raw values means the model is penalized heavily for TJ cases and may fit poorly for common short-duration injuries. **Verdict: implement log1p transform — this is the sklearn analogue to the ZINB approach validated in literature, and directly reduces RMSE.**

- **Chaining order (probability → severity):** Injury probability first, then days-missed severity, then injury type. This is the correct causal ordering — a pitcher must be injured before severity is meaningful. **Verdict: no change — current approach matches literature.**

- **RF for all task heads:** Random Forest used for both classification and regression heads. XGBoost is available in the shared-representation model but not in the chained model. For the chained model (the primary architecture), RF at n_estimators=300, max_depth=5 is a reasonable starting point. **Verdict: no change in chained model — RF with these parameters is appropriate for the dataset size. XGBoost is used in the shared model for comparison.**

- **Evaluation metric (MAE/RMSE) for days_missed:** MAE and RMSE on raw days-missed are appropriate summary statistics, but RMSE is disproportionately influenced by TJ-surgery outliers. After log1p transform at training time and expm1 at prediction time, MAE in original space will better reflect typical prediction error. **Verdict: acceptable — MAE is the right metric for final evaluation; RMSE reported for comparison.**

- **Severity thresholds (mild/moderate/severe):** Not explicitly implemented as a classification task — `next_injury_days_lost` is treated as a continuous regression. The severity classification in the shared model head could use IL designations (10-day: mild, 15-day: moderate, 60-day: severe). **Verdict: noted as limitation — continuous regression is more informative than discretized severity classes and avoids threshold choice subjectivity. No change.**

### Improvements Implemented

1. **Log1p transform for `next_injury_days_lost`** (`src/models/multitask_models.py`):
   - Added `LOG_REGRESSION_TARGETS = frozenset({"next_injury_days_lost"})` constant with research citation in the comment.
   - Applied `np.log1p(y_sub)` before fitting in both `train_chained_multitask_model` and `_SharedRepresentationModel.fit`.
   - Stored `_log_transformed_tasks_` flag on model dict / class attribute.
   - Applied `np.expm1(raw_pred)` in `predict_all_tasks` for flagged tasks, restoring original days-lost scale for downstream consumers (NB09 Injury Risk+ uses these predictions).

### Verified

- TEST_MODE run: 22s — PASS.
- Full run: PASS.
- `python scripts/verify_outputs.py --only 08` — PASS.
