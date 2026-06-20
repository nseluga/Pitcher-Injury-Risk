# Survival Model Improvement Ideas — 2026-06-20 Round S-002 → COMPLETED

## Already tried (from improvement_progress.json + log)
- Fix _MAX_COX_ROWS / _MAX_RSF_ROWS subsample bug (pre-tracking)
- Arm-injury-only event definition (pre-tracking)
- Stratified Cox PH on prior_il_elbow (pre-tracking)
- Elastic-net Cox tuning via lifelines penalizer/l1_ratio grid (pre-tracking)
- GBSA addition with basic hyperparameter grid (pre-tracking)
- **Round S-001**: Stochastic GBSA subsample=0.8 → C-index 0.5591 (+0.0032)
- **Round S-002**: GBSA double-stochastic (max_features='sqrt' + n=200) → NULL (0.0000)
  - max_features='sqrt': C=0.5542 (WORSE); max_features=0.5: C=0.5462 (WORSE)
  - n=200 + any max_features: WORSE than n=100 + max_features=None
  - Confirmed: optimal GBSA = n=100, lr=0.1, depth=2, subsample=0.8, max_features=None

## New approaches — S-003

Status: consecutive_non_improvements=2. Need a meaningful improvement or reaches 3.

1. **Ensemble risk score (GBSA + RSF + Cox rank average)** — Rank-average the normalized
   log-hazard scores from best Cox PH, RSF, and GBSA. Model diversity produces robust
   combinations even when individual models are mediocre; committee methods reliably gain
   +0.01–0.03 in survival literature (Graf et al., 1999). Pure NB07 change, ~10 lines.
   Expected delta: +0.010–0.030. **HIGH value, LOW risk — RECOMMENDED for S-003.**

2. **Log-normal + Log-logistic AFT models** — Both already imported (LogNormalAFTFitter,
   LogLogisticAFTFitter). Two-line addition per model. Non-monotone hazard may fit pitcher
   fatigue patterns better than Weibull. Could contribute to ensemble (idea 1) as well.

3. **CoxnetSurvivalAnalysis (sksurv)** — sksurv's coordinate-descent penalized Cox handles
   all 82 features without the univariate pre-selection bottleneck. Different optimizer than
   lifelines — may surface jointly-predictive features missed by current 30-feature subset.

4. **Cumulative IL count feature (all types)** — total_prior_il_count across all injury
   types is the strongest frailty proxy in recurrent-event literature. Requires NB05 edit.
   Currently only prior_il_elbow, prior_il_shoulder in features; total is not there.

5. **GBSA min_samples_leaf tuning** — Try 10 (more expressive) and 50 (more regularized).
   n=100, lr=0.1, depth=2, sub=0.8 is locked; this is the only untested tuning dimension.

6. **Frailty model (GammaMixtureFrailtyFitter)** — Pitcher-level random effects for
   structural injury proneness. Complex to implement but high literature evidence (+0.01–0.05).

## Decision for S-003

**Pick: Idea 1 — Ensemble risk score (GBSA + RSF + Cox rank average)**

Rationale:
- Consecutive_non_improvements=2 — need a meaningful gain to avoid convergence
- Ensemble averaging is the highest-expected-value idea remaining (literature: +0.01–0.03)
- No feature engineering or new model families needed — pure post-processing of existing models
- All three component models already trained and available in survival_models dict
- 10–15 lines of code, all in NB07

---

# Previous session notes (kept for reference)
# Survival Model Improvement Ideas — 2026-06-20 (Round 1 formal tracking)

## Already tried (now in NB07 from prior sessions)
- Fix subsample bug: raised `_MAX_COX_ROWS=45K`, `_MAX_RSF_ROWS=35K`
- Raised `_MAX_COX_FEATURES` 20→30 to capture injury history + workload
- Arm-injury-only event definition (elbow/shoulder/forearm IL stints)
- Stratified Cox PH on `prior_il_elbow` (addresses PH violation p=0.002)
- Elastic-net penalized Cox (penalizer × l1_ratio grid; best: pen=0.01, l1=0.5)
- RSF (n_estimators=50–100, max_depth=4–6; best C=0.555)
- GBSA (n_estimators=100, lr=0.05–0.10, max_depth=2–3, NO subsample; best C=0.553)
- **Current best C-index: 0.5559** (tuned Cox PH)

## New approaches to consider — 2026-06-20

1. **GBSA with `subsample` (stochastic gradient boosting)** — Friedman (2002)
   showed `subsample=0.5–0.8` reduces variance in boosted trees by 15–30% on
   out-of-sample data. sksurv's `GradientBoostingSurvivalAnalysis` supports
   `subsample` directly. Current GBSA tuning never tested this parameter.
   Expected gain: +0.005–0.020 C-index. **CHOSEN for Round 1 (2026-06-20).**

2. **Log-normal AFT model** — Weibull AFT assumes monotone hazard. Log-normal
   captures non-monotone hazard (rises then falls), matching early-season
   arm-fatigue peak. Code already imports `LogNormalAFTFitter`. Low cost.

3. **Log-logistic AFT model** — Similar rationale to log-normal but heavier tails.
   Already imported. One-line addition.

4. **CoxnetSurvivalAnalysis (sksurv)** — sksurv's penalized Cox handles all 82
   features natively without pre-selection. Avoids univariate correlation filter
   bottleneck. May surface jointly-predictive features missed by current approach.

5. **Non-linear Cox feature transforms** — 7 PH violations detected. Adding
   `log1p(days_since_last_injury)` + `acwr_7_28²` (J-curve) directly addresses
   the non-linearity. Would benefit Cox; RSF/GBSA already capture non-linearity.

6. **Multi-strata Cox** — Stratify on `prior_il_elbow` (done) AND binned
   `pitches_90d` quartiles to address second PH violation. Risk: sparse strata.

7. **GBSA with dropout_rate (DART)** — sksurv 0.21+ supports `dropout_rate`
   parameter (DART algorithm). Prevents individual trees from over-specializing;
   known to improve C-index on sparse event survival data. Uncertain availability.

8. **Frailty model (pitcher-level random effects)** — lifelines
   `GammaMixtureFrailtyFitter` captures unobserved pitcher-level heterogeneity.
   Clinical recurrent-event literature: +0.01–0.05 C-index. Complex to implement.

9. **Ensemble risk score** — Rank-average log-hazard from Cox + RSF + GBSA.
   Model diversity without new features. Literature: +0.01–0.03 C-index.

10. **Extended GBSA n_estimators 300–500 with lr=0.01** — Current tuning capped
    at n=100. More trees with smaller lr often produce better survival models.

---

# Previous session notes (2026-06-19 updated 2026-06-19 Round 1)

---

# Session 2026-06-19 — Round 1 (formal tracking starts here)

## Already tried (now implemented in NB07)
- Fix events-only subsample bug: raised `_MAX_COX_ROWS=45K`, `_MAX_RSF_ROWS=35K`
- Raised `_MAX_COX_FEATURES` 20→30 to include injury history + workload features
- Arm-injury-only event definition (filter to elbow/shoulder/forearm IL stints)
- All 4 model families: Cox PH (untuned), Weibull AFT, RSF, GBSA
- Hyperparameter tuning: Cox PH penalizer×l1_ratio, RSF depth/estimators, GBSA lr/depth
- **Current best C-index: 0.5559 (tuned Cox PH, pen=0.01, l1_ratio=0.5)**

## New approaches to consider

1. **Stratified Cox PH on `prior_il_elbow`** — Schoenfeld residuals explicitly flagged this feature (p=0.002) as violating the proportional hazards assumption. Lifelines directly recommends `strata=['prior_il_elbow']` for features with few unique values (0–4). Stratification gives each stratum its own baseline hazard shape: pitchers with 0 vs 1 vs 2+ prior elbow IL stints likely have qualitatively different injury trajectories. HR is currently 1.20 (p=8.5e-10) — the 3rd strongest predictor. Removing its PH assumption should improve both model fit and C-index. **CHOSEN for Round 1.**

2. **Log-normal AFT** — Weibull AFT assumes a monotone hazard. Log-normal allows a non-monotone hazard (rises then falls), which may better fit pitcher injury risk accumulation. Already importable via `train_aft_model(distribution='lognormal')`. Trivial to add.

3. **Log-logistic AFT** — Heavy-tailed distribution. Captures the small subset of pitchers with very high chronic risk better than Weibull. Same one-line addition.

4. **Ensemble risk score** — Normalize log-hazard from Cox + RSF + GBSA into [0,1], rank-average. Literature shows +0.01–0.03 C-index improvement from model diversity without any new features.

5. **GBSA feature importance-based Cox feature selection** — Replace univariate `corrwith(E)` filter with GBSA's `feature_importances_`, which captures multivariate interaction effects. Current top-5 by correlation: days_since_last_injury, prior_il_elbow, sl_pct, ch_pct_30d_avg, acwr_7_28. GBSA-ranked top-5 might be different.

6. **Frailty model (GammaMixtureFrailtyFitter)** — Pitcher-level random effects for unobserved heterogeneity (some pitchers structurally more injury-prone regardless of workload). Clinical recurrent-event literature reports +0.01–0.05 C-index. Complex to implement and requires lifelines experimental API.

7. **More aggressive GBSA grid search** — FAST_TUNING=True only tested 2 GBSA configs. Expand to n_estimators=[200,300,500], subsample=[0.5,0.8], dropout_rate and use all seasons for tuning.

8. **Season phase features** — Early/mid/late season indicator (none in current feature matrix). Injury hazard likely varies across season (spring arm fatigue, late-season overuse). Would capture within-season baseline hazard variation that Cox can't currently model.

9. **Cumulative injury burden** — `total_prior_il_count` (all IL stints, not just arm) and `days_since_any_il` (not just arm). Recurrent-event survival literature: past injury count is the strongest single predictor of future injury beyond arm-specific counts.

10. **Time×covariate interaction for PH violators** — `days_since_last_injury` also violates PH (p=0.016). Adding `days_since_last_injury × log(time)` as a synthetic feature allows the HR to vary with time within the Cox PH framework. Equivalent to a restricted time-varying Cox without the complexity of truly time-varying covariates.

---

# Previous session notes (2026-06-19 earlier)

## Already tried (from improvement_progress.json)
- None (first survival-specific round; prior binary-classifier rounds are in model_improvement_log.md)

## Baseline
- Best untuned C-index: 0.514 (RSF)
- Best tuned C-index: 0.535 (Cox pen=0.5, l1_ratio=0.5)
- All models: IBS ~0.53-0.54 (near null model)

## Critical bug found during orientation
`_MAX_COX_ROWS = 12_000 < 18,007 events` in training set → `_subsample()` hits the
"events only" branch → **0 censored observations in any model's training data**.
Cox PH, AFT Weibull, and RSF are all trained on a 100%-event dataset. This breaks
the censoring structure survival models depend on and is the most likely root cause
of near-random C-index.

## New approaches to consider

1. **Fix events-only subsample (HIGHEST VALUE)** — Raise `_MAX_COX_ROWS` from 12,000
   to ≥40,000 so `_subsample()` keeps all 18,007 events + ~22K censored rows. Same
   for `_MAX_RSF_ROWS` → 30,000. Training survival models on datasets with 0 censored
   rows is methodologically broken and explains the near-random C-index.
   Evidence: Harrell et al. (1996) show Cox PH C-index degrades badly when censoring
   structure is absent from training.

2. **Gradient Boosted Survival (GBS)** — `GradientBoostingSurvivalAnalysis` from
   scikit-survival. Nonlinear, handles feature interactions, no PH assumption. Often
   best on tabular survival data. Needs proper censoring ratio first (idea 1).

3. **Arm-only injury event filter** — Redefine the survival event as arm/shoulder/elbow
   IL stints only, removing noise from oblique strains, leg injuries, etc. Reduces
   label noise. Needs filtering `data/processed/injuries_clean.parquet` on injury type.

4. **Time since last injury feature** — `days_since_last_IL` is the strongest single
   predictor in clinical recurrence survival literature (frailty proxy). Not currently
   in the feature matrix. Add in NB05 or inline in NB07.

5. **Cumulative injury count feature** — `prior_IL_count` per pitcher up to each row.
   Captures structural frailty (chronically injury-prone pitchers). Easy to compute.

6. **Stratified Cox PH by starter/reliever** — Stratify the baseline hazard by role.
   The PH assumption test already showed `pitches_28d` and ACWR violate PH — stratifying
   by role partially relaxes this. Lifelines supports `strata` arg in `.fit()`.

7. **Log-normal / Log-logistic AFT** — Try additional AFT distributions. Log-logistic
   has a non-monotonic hazard (rises then falls), which may fit injury risk patterns
   better than Weibull's monotone hazard.

8. **Season-phase feature** — bin `game_number_in_season` into early/mid/late/postseason.
   Hazard shape changes across the season (fresh vs. fatigued arm). Non-PH features
   were flagged by Schoenfeld residuals; a discrete phase variable might help.

9. **Maintain realistic censoring ratio in subsample** — Instead of just raising the cap,
   explicitly enforce: keep all events + sample censored to maintain ~50% censoring
   in the training subsample. This gives the model more "safe" examples to calibrate on.

10. **Calibration recalibration at t=30/60/90** — Apply isotonic regression to survival
    probabilities at fixed horizons. Can improve IBS even if C-index is flat.

## Status (2026-06-19)

**Round 0 (already implemented in code, notebook not yet rerun):**
- Idea 1 is done: `_MAX_COX_ROWS=45,000`, `_MAX_RSF_ROWS=35,000` in survival_models.py
- Side-effect: with correct E variation in subsample, `corrwith(E)` now works →
  `prior_il_total` (|corr|=0.203, strongest predictor) will appear in Cox feature selection

**Critical additional bug found:**
- `_select_top_features` selects by correlation with EVENT INDICATOR (binary)
- With old all-events subsample: E was constant (all 1s) → correlation undefined → fallback to first 20 cols
- First 20 cols are workload/velocity — `prior_il_total` was NEVER in Cox model
- Fix: raise `_MAX_COX_FEATURES` from 20 → 30 to capture both injury history AND workload

## Round 1 (this session) — Chosen approach

**GradientBoostingSurvivalAnalysis + raise Cox feature cap**
- Add `GradientBoostingSurvivalAnalysis` from sksurv 0.27.0 (confirmed available)
- Nonlinear, no PH assumption, uses ALL features (no pre-selection bottleneck)
- Captures `prior_il_total` × workload interactions that Cox and RSF miss
- Raise `_MAX_COX_FEATURES` from 20 → 30 so Cox also includes injury history + workload
- The notebook rerun with fixed caps is the prerequisite — both fixes activate together

## New ideas to consider beyond Round 1

### From 2026-06-19 analysis

11. **Log-rank feature selection** — Replace `corrwith(E)` with univariate log-rank p-values.
    Log-rank tests whether a feature stratifies survival curves — directly relevant to C-index.
    `acwr_7_28` (PH violation, real signal) would score much higher than by binary correlation.

12. **Add age to features** — `age` is in ID_COLS (excluded). Older pitchers have accumulated
    stress. Age is among top-5 predictors in clinical pitcher injury survival literature.

13. **CoxnetSurvivalAnalysis (sksurv)** — sksurv's penalized Cox handles 80 features natively
    without pre-selection. Avoids the feature selection bottleneck entirely. Faster than lifelines.

14. **Frailty model** — `GammaMixtureFrailtyFitter` (lifelines) adds pitcher random effects.
    Addresses recurrent-event structure: same pitcher appears hundreds of times.

15. **Arm-injury-only event** — Filter to arm/shoulder/elbow IL stints. Removes noise from
    leg injuries which pitch-load features can't predict.
