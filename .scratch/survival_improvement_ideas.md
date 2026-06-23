# Survival Model Improvement Ideas — 2026-06-22 Round S-005 (THIS SESSION)

## Status entering this session
- Rounds completed: 4
- Best C-index: 0.5658 (3-model ensemble: GBSA+Cox+RSF)
- consecutive_non_improvements: 1 (S-004 AFT regression -0.0021, reverted)
- S-005 code already in notebook (ExtraSurvivalTrees) but TEST_MODE TIMED OUT at tuning cell

## Fresh brainstorm — 2026-06-22

Already tried: Arm-only event, Stratified Cox, Elastic-net Cox, GBSA (pre-tracking), S-001 stochastic GBSA, S-002 column subsampling (null), S-003 ensemble GBSA+Cox+RSF (+0.0067), S-004 AFT 4th member (reverted).

1. **ExtraSurvivalTrees (S-005, already in notebook)** — Random split thresholds decorrelate from GBSA more than RSF does. Fix timeout by reducing FAST_TUNING grid and setting RUN_TUNING=not TEST_MODE. CHOSEN.

2. **GBSA with IPCWLS loss** — `GradientBoostingSurvivalAnalysis(loss='ipcwls')`. With 91% censoring, IPCWLS theoretically more efficient than Cox partial likelihood. One-parameter change, zero new infrastructure.

3. **Weighted ensemble** — Optimize GBSA:Cox:RSF weights on 2022 validation season instead of equal 1/3. GBSA (0.5591) > Cox (0.5559) > RSF (0.5549) — GBSA may deserve > 33% weight.

4. **CoxnetSurvivalAnalysis (sksurv)** — Coordinate-descent LASSO Cox over all 82 features (not top-30 correlation filter). Different feature set = ensemble diversity.

5. **Log-rank feature selection for Cox** — Replace corrwith(E) with per-feature log-rank p-value. Aligns feature selection criterion with survival-specific signal. acwr_7_28 likely ranks higher.

6. **Season-phase feature** — early/mid/late season indicator. Requires NB05 re-run.

7. **Cumulative all-injury IL count** — total_prior_il_count across all body parts as frailty proxy. Requires NB05 re-run.

8. **Multi-seed GBSA ensemble** — Average 3 GBSA runs with seeds {42, 0, 123}. Stochastic (sub=0.8) → different seeds = different models → averaging reduces Monte Carlo noise.

9. **Multi-strata Cox (bin fb_pct_30d_avg)** — strongest PH violator (p<5e-5). Quartile bins as additional strata alongside prior_il_elbow. Aggressively relaxes both PH violations.

10. **GBSA min_samples_leaf=15** — untested midpoint between 10 (too small) and 20 (current). Minor regularization adjustment.

## Decision for this session
**Idea 1: ExtraSurvivalTrees (S-005)** — already coded, just needs the tuning timeout fixed.
Fix: set `RUN_TUNING = not TEST_MODE` in cell 11 to skip tuning in TEST_MODE.
Second choice if S-005 fails: Idea 2 (IPCWLS loss).

---

# Survival Model Improvement Ideas — 2026-06-21 Round S-005

## Status entering S-005
- Rounds completed: 4
- Best C-index: 0.5658 (3-model ensemble: GBSA+Cox+RSF)
- consecutive_non_improvements: 1 (S-004 AFT regression -0.0021, reverted)

## Already tried
- Arm-only event definition, Stratified Cox, Elastic-net Cox, GBSA addition (pre-tracking)
- S-001: Stochastic GBSA subsample=0.8 → +0.0032
- S-002: GBSA column subsampling → null result
- S-003: Ensemble rank-average (GBSA+Cox+RSF) → +0.0067
- S-004: Log-normal AFT as 4th ensemble member → -0.0021 (reverted; AFT C=0.5447 too weak)

## New approaches for S-005+ (brainstorm 2026-06-21)

1. **ExtraSurvivalTrees** (CHOSEN) — sksurv ExtraSurvivalTrees uses random split thresholds
   (not optimized), adding randomness beyond RSF's feature subsampling. Uses log-rank criterion
   (survival-specific split quality). More decorrelated from GBSA than RSF is, since both RSF
   and GBSA use optimal split points. Key: "randomness goes one step further in way splits are
   computed" (sksurv docs). Test as RSF replacement AND as 4th ensemble member.
   EV: if EST C≥0.55 and errors decorrelated from GBSA, ensemble gain +0.003–0.010.

2. **CoxnetSurvivalAnalysis** — Coordinate-descent LASSO/elastic-net Cox over all 82 features.
   Research: Coxnet achieved C=0.688 on UK Biobank data, similar to RSF. Avoids correlation-based
   pre-selection bias (current Cox uses top-30 by corrwith(E)). Could find jointly-predictive
   features the correlation filter misses. If Coxnet C≥0.562, replace Cox in ensemble.

3. **GBSA with IPCWLS loss** — `GradientBoostingSurvivalAnalysis(loss='ipcwls')`. With 91%
   censoring, inverse-probability-censoring-weighted LS is theoretically more efficient than Cox PL.
   One-parameter change. If IPCWLS-GBSA C > Cox-loss GBSA (0.5591), swap.

4. **Log-rank feature selection for Cox** — Replace corrwith(E) with log-rank p-value ranking
   (lifelines logrank_test). Log-rank directly tests survival curve stratification; acwr_7_28
   (known signal, PH violator) should rank higher than on binary correlation.

5. **Season-phase feature** — Early/mid/late season indicator. Injury hazard varies by season.
   Requires NB05 edit + feature_matrix rebuild.

6. **Weighted ensemble optimization** — Optimize GBSA:Cox:RSF weights on held-out 2022 season
   (validation fold) before applying to test. GBSA has highest C (0.5591), Cox second (0.5559).
   Equal weights may underweight GBSA contribution.

7. **Multi-strata Cox (bin fb_pct_30d_avg)** — fb_pct_30d_avg is strongest Schoenfeld PH
   violator (p<5e-5). Add quartile-binned fb_pct to strata alongside prior_il_elbow.

8. **Cumulative all-injury IL count** — total_prior_il_count across ALL body parts as frailty
   proxy (not just arm-specific). Strongest recurrent-event frailty proxy in literature.

9. **Frailty model (GammaMixtureFrailtyFitter)** — Pitcher-level random effects for structural
   injury proneness. lifelines experimental API — highest risk, highest ceiling.

10. **Time-varying covariate Cox** — Use long-format counting process for acwr_7_28 (PH violator).
    High implementation cost, reserve for last resort.

## Decision for S-005
**Idea 1: ExtraSurvivalTrees as RSF replacement + possible 4th ensemble member.**
- S-004 failure was AFT at C=0.5447 (too weak). EST expected to match RSF (C≈0.555).
- Structural difference from RSF: random split thresholds + log-rank criterion.
- GBSA and RSF both use optimal split points → correlated errors.
- EST uses random splits → structurally decorrelated from GBSA → better ensemble diversity.
- Try: (a) EST replacing RSF, (b) EST as 4th member if C≥0.55.
- Low risk: if EST C < 0.55, don't add it; ensemble stays at 0.5658.

## Research findings (2026-06-21)
- sksurv docs: "Compared to RandomSurvivalForest, randomness goes one step further in way
  splits are computed" in ExtraSurvivalTrees. Uses log-rank splitting (survival-specific).
- CoxnetSurvivalAnalysis benchmark (UK Biobank, arXiv:2503.08870): C=0.688, nearly equal to
  RSF — better regularization than univariate pre-selection.

---

# Survival Model Improvement Ideas — 2026-06-21 Round S-004 (SESSION 3 — NEW RUN)

## Session re-orientation (2026-06-21, session 3)
S-004 was implemented in a prior session (log-logistic + log-normal AFT in cells 5 + 14) but
was NEVER EXECUTED. Provenance confirms last run at 15:16 UTC was TEST_MODE without AFT models
(individual_c_indices has no lognormal/loglogistic). improvement_progress.json has 3 rounds only.
This session: run TEST_MODE → full run → log → commit.

## Already tried (from improvement_progress.json)
- Arm-injury-only event definition, Stratified Cox PH, Elastic-net Cox, GBSA (pre-tracking)
- **S-001**: Stochastic GBSA subsample=0.8 → C-index +0.0032 (0.5591)
- **S-002**: GBSA column subsampling (max_features='sqrt' + n=200) → NULL (no change)
- **S-003**: Ensemble rank-average (GBSA+Cox+RSF) → C-index +0.0067 (0.5658)

## Current: S-004 — Log-Normal + Log-Logistic AFT as 4th ensemble member (IMPLEMENTING NOW)
Already coded in NB07 c03-train and ensemble cell. Just needs execution.
**Why it might help**: Weibull AFT (C=0.5407 in prior run) assumes monotone hazard. Log-logistic/
log-normal allow non-monotone hazard: risk rises then falls — matching pitcher injury dynamics where
risk builds with accumulated workload then resolves with rest. Even at C~0.52-0.54, a 4th model
with different inductive bias adds ensemble diversity (variance reduction).
**Threshold**: add to ensemble if best AFT C-index ≥ 0.51 (coded into cell 029661a1).

## New approaches for S-005+ (brainstorm 2026-06-21)

Target to beat: 0.5658. Need ≥ 0.5708 (+0.005) for a clear win.
consecutive_non_improvements = 0 (reset after S-003). Status: in_progress.

1. **ExtraSurvivalTrees (sksurv.ensemble.ExtraSurvivalTrees)** — One-line swap from RSF.
   Extra trees use fully random split thresholds rather than optimal splits → even lower variance,
   more decorrelated from GBSA than RSF is. `from sksurv.ensemble import ExtraSurvivalTrees`.
   Could replace RSF in ensemble for more diversity. Very low implementation cost.
   EV: if EST C-index ≈ RSF but more decorrelated → ensemble +0.003–0.007.

2. **CoxnetSurvivalAnalysis (sksurv)** — Coordinate-descent penalized Cox over all 82 features
   (not top-30 correlation filter). LASSO zeros uninformative features automatically; elastic-net
   handles correlated features. Different feature set = more ensemble diversity as 4th member.
   Implementation: `from sksurv.linear_model import CoxnetSurvivalAnalysis`.
   EV: C-index ~0.55–0.57 (similar to lifelines Cox but broader feature set). Medium cost.

3. **GBSA with IPCWLS loss** — `GradientBoostingSurvivalAnalysis(loss='ipcwls')` optimizes
   inverse probability censoring weighted least squares rather than Cox partial likelihood.
   With 91% censoring, IPCWLS is theoretically more efficient. One hyperparameter change.
   Risk: may not outperform Cox-loss GBSA if censoring mechanism is uninformative.
   EV: +0.002–0.008. Low implementation cost.

4. **Log-rank feature selection for Cox** — Replace `corrwith(E)` filter with univariate
   log-rank p-values (lifelines `logrank_test`). Log-rank directly tests whether a feature
   stratifies survival curves; acwr_7_28 (known signal, violates PH) would score much higher
   than on binary correlation. Moderate implementation cost.
   EV: +0.003–0.010 if current top-30 selection misses survival-relevant features.

5. **Season-phase categorical feature** — Indicator: early (game_num ≤ 30), mid (31–100), late
   (101+). Pitcher arm injury hazard peaks early (spring load) and late (accumulated fatigue).
   Cox PH cannot model baseline hazard variation without explicit feature or stratification.
   Requires NB05 edit + feature_matrix rebuild. Medium cost.
   EV: +0.003–0.008 if season-phase captures seasonal hazard variation.

6. **Cumulative all-injury IL count** — total_prior_il_count across ALL body-part IL stints
   (not just arm). Pitchers with 5+ lifetime IL stints regardless of type are structurally fragile.
   Strongest frailty proxy in recurrent-event literature (Prentice–Williams–Peterson model).
   Requires NB05 edit + feature_matrix rebuild. Medium cost.
   EV: +0.005–0.012 — frailty proxies consistently improve survival C-index.

7. **Weighted ensemble optimization** — Optimize ensemble weights (GBSA weight, Cox weight,
   RSF weight) on 2022 validation season via Nelder-Mead. Equal weights are suboptimal when
   component C-indices differ (GBSA=0.559, Cox=0.541, RSF=0.548). Risk: overfit to 2022.
   Low implementation cost. Can test: optimize on train val, measure on test.
   EV: +0.002–0.005 if weighting reflects true model quality differences.

8. **Multi-strata Cox (bin fb_pct_30d_avg)** — fb_pct_30d_avg is the strongest PH violator
   (Schoenfeld p<5e-5). Binning into quartiles and adding to strata list (alongside prior_il_elbow)
   gives each fastball-usage-quartile cohort its own baseline hazard. Directly addresses the
   strongest remaining PH violation without removing the covariate from regression.
   Low implementation cost (modify strata arg in Cox fit call).
   EV: +0.003–0.008 if fastball-usage hazard truly varies over time non-proportionally.

9. **Time-varying covariate Cox (long format)** — acwr_7_28 violated PH assumption
   (Schoenfeld p<0.05). Modeling it as a time-varying covariate in lifelines' long-format Cox
   removes the violation by construction. Requires restructuring the input dataframe.
   High implementation cost. Reserve for if simpler approaches stall.
   EV: +0.003–0.008 if acwr_7_28 truly has time-varying effect.

10. **Frailty model (GammaMixtureFrailtyFitter)** — Pitcher-level random effects for
    structural injury proneness (some pitchers are just fragile regardless of workload).
    lifelines experimental API. Highest risk (API instability), highest ceiling (+0.01–0.05).
    Reserve for last resort.
    EV: high uncertainty ±0.00–0.05.

## Decision for S-004 (this session)
**Execute pre-implemented S-004: log-logistic + log-normal AFT as 4th ensemble member.**
The code is in NB07 cells c03-train and 029661a1 (ensemble evaluation). Zero implementation
risk — just needs to run. If AFT C-index ≥ 0.51, best AFT added to 4-model ensemble.
Best case: 4-model ensemble C-index +0.003–0.010 vs current 0.5658 → 0.568–0.576.
Null case: AFT C-index < 0.51 or 4-model ensemble ≤ 3-model → logged as null result.

## Research findings (prior sessions)
- arXiv:2403.07460: "A straightforward aggregation of parametric, semi-parametric and ML methods
  that assume diverse hazard shapes allows the ensemble to gain in robustness." → validates AFT addition.
- PMC8523281: Log-logistic hazard is non-monotonic (rises then falls) when shape β>1 — ideal for
  workload-driven injury dynamics (risk builds then resolves with rest).
- DeepHit (arXiv:2601.19479): C-index 0.762 in elite football injury combining survival + ensemble.
  Our approach mirrors: heterogeneous model ensemble for better C-index.

---

# Survival Model Improvement Ideas — 2026-06-21 Round S-004 (PREVIOUS SESSION — IMPLEMENTATION DONE)

## Already tried (from improvement_progress.json + pre-tracking)
- Fix _MAX_COX_ROWS / _MAX_RSF_ROWS subsample bug (pre-tracking)
- Arm-injury-only event definition (pre-tracking)
- Stratified Cox PH on prior_il_elbow (pre-tracking)
- Elastic-net Cox tuning via lifelines penalizer/l1_ratio grid (pre-tracking)
- GBSA addition with basic hyperparameter grid (pre-tracking)
- **Round S-001**: Stochastic GBSA subsample=0.8 → C-index 0.5591 (+0.0032)
- **Round S-002**: GBSA column subsampling (max_features='sqrt' + n=200) → NULL (0.0000)
  - max_features='sqrt': C=0.5542 (WORSE); max_features=0.5: C=0.5462 (WORSE)
  - n=200, lr=0.1, mf=None: C=0.5540 (WORSE)
  - Confirmed: optimal GBSA = n=100, lr=0.1, depth=2, subsample=0.8, max_features=None
- **Round S-003**: Ensemble rank-average (GBSA+Cox+RSF) → C-index 0.5658 (+0.0067)
  - Individual: GBSA=0.5591, Cox=0.5559, RSF=0.5549 in TEST_MODE

## New approaches for S-004+ (brainstorm 2026-06-21)

Best C-index to beat: 0.5658 (3-model ensemble GBSA+Cox+RSF). Need ≥ 0.5708 for +0.005.
consecutive_non_improvements = 0 (reset after S-003). Status: in_progress.

1. **Log-logistic + Log-normal AFT as 4th ensemble member** (CHOSEN — ALREADY IMPLEMENTED)
   Code already in cells 5 and 14 of NB07, never executed. Non-monotone hazard AFT adds
   distinct inductive bias vs GBSA (boosted trees, partial-likelihood loss), Cox (L1/L2
   linear), RSF (bagged trees). Log-logistic hazard rises then falls — matches injury
   dynamics where risk peaks after a heavy workload, then resolves with rest. If best AFT
   C-index ≥ 0.51, added as 4th ensemble member. Even modest diversity reduces ensemble
   variance. Research basis: Wu et al. (2025) Applied Stochastic Models; Royston & Parmar
   (2002) on flexible parametric AFT models in clinical survival.
   EV: if AFT reaches 0.52-0.54, 4-model ensemble could push to 0.567-0.575.

2. **ExtraSurvivalTrees (sksurv.ensemble.ExtraSurvivalTrees)** — One-line swap from RSF.
   Extra trees use random split points (not optimal) → lower variance, higher diversity
   vs RSF. More decorrelated from GBSA than RSF is. Adding EST alongside or replacing RSF
   in the ensemble could improve diversity. Same sksurv interface as RSF. Very low cost.

3. **CoxnetSurvivalAnalysis (sksurv elastic-net Cox)** — Coordinate-descent over all 82
   features (not top-30 correlation filter). LASSO penalty zeros uninformative features
   automatically; elastic-net handles correlated features. Could uncover jointly-predictive
   interaction terms current Cox misses. Different feature set = ensemble diversity.
   Medium implementation cost, 4th ensemble candidate.

4. **GBSA with IPCWLS loss** — `GradientBoostingSurvivalAnalysis(loss='ipcwls')` optimizes
   inverse probability censoring weighted least squares. With 91% censoring, IPCWLS is
   more statistically efficient than Cox partial likelihood (default). One-parameter change.
   Compare C-index to Cox-loss GBSA; could swap if better.

5. **Log-rank feature selection for Cox** — Replace correlation filter (`corrwith(E)`) with
   univariate log-rank p-values (lifelines `logrank_test`). Log-rank directly tests whether
   a feature stratifies survival curves; acwr_7_28 (PH violator, known signal) would likely
   score much higher than binary correlation with event indicator. Moderate implementation.

6. **Season-phase categorical feature** — Indicator: early (game ≤ 30), mid (31–100), late
   (101+). Pitcher injury hazard varies by season: arm fatigue peaks in April (new load)
   and September (max workload). Cox PH cannot model this without explicit feature or strat.
   Requires NB05 edit (add column), then NB07 re-run. Medium cost.

7. **Cumulative all-injury IL count** — total_prior_il_count summing ALL body-part IL stints,
   not just arm-specific. Pitchers with 5+ IL stints regardless of type are structurally
   fragile. Strongest frailty proxy in recurrent-event literature. Requires NB05 edit.

8. **Weighted ensemble optimization** — Instead of equal-weight rank average, optimize weights
   on 2022 validation seasons via Nelder-Mead (same as Cox tuning approach). Risk: overfitting.
   Low implementation cost; compute weights on val, apply on test. Compare to equal-weight.

9. **Multi-strata Cox (bin fb_pct_30d_avg)** — fb_pct_30d_avg was the strongest PH violator
   (Schoenfeld p<5e-5). Binning into quartiles and adding to strata (alongside prior_il_elbow)
   would aggressively relax the two main PH violations, potentially improving model fit.

10. **Frailty model (GammaMixtureFrailtyFitter)** — Pitcher-level random effects for
    structural injury proneness. lifelines experimental API. Highest risk, highest ceiling
    (+0.01–0.05). Reserve for last resort.

## Decision for S-004 (this session)
**Execute pre-implemented Idea 1: Log-logistic + Log-normal AFT as 4th ensemble member.**
Rationale:
- Already coded in cells 5 + 14 — zero implementation risk
- Adds genuine distributional diversity (non-monotone hazard) to the ensemble
- consistent with the ensemble-diversity strategy that gained +0.0067 in S-003
- If AFT C-index < 0.51, ensemble falls back to 3-model (no harm done)
- If ensemble improves, consecutive_non_improvements stays 0; if not, → 1

## Research findings (WebSearch 2026-06-21)
- arXiv:2403.07460 "Experimental Comparison of Ensemble Methods and Time-to-Event Analysis Models
  Through IBS and C-index": "A straightforward aggregation of methods of different natures,
  parametric, semi-parametric and machine learning, that assume diverse shapes of the hazard
  function allows the ensemble model to gain in robustness." → directly validates this approach.
- PMC8523281: Log-logistic hazard function is non-monotonic (unimodal) when shape β > 1 —
  rises then declines — making it well-suited to sports injury dynamics where risk peaks with
  accumulated workload then resolves with rest.
- DeepHit survival (arXiv:2601.19479) for elite football injury: C-index 0.762 achieved
  combining survival analysis with ML (imputation + ensemble); our approach mirrors this strategy.

---

# Survival Model Improvement Ideas — 2026-06-21 Round S-003 (SESSION 2: CONFIRMING FULL RUN)

## Session 2 brainstorm (2026-06-21, continuing from prior session)

S-003 is already implemented: ensemble rank-average (GBSA + Cox + RSF) in cell 14, updated
GBSA tuning grid in cell 12. TEST_MODE showed 0.5658 (+0.0067 vs best individual 0.5591).
This session confirms with a full TEST_MODE re-run, then launches the full dataset run.

### Still-untested high-value ideas (for S-004+)
1. **CoxnetSurvivalAnalysis (sksurv)** — coordinate-descent penalized Cox over all 82 features,
   elastic-net penalty suppresses zero-signal features. Could find jointly-predictive features
   current Cox misses. Usable as 4th ensemble member for more diversity. HIGH value.
2. **Log-logistic AFT as 4th ensemble member** — Different distributional assumption from Cox/RSF/GBSA.
   One-line addition to AFT training. Could push ensemble past next threshold.
3. **Log-rank feature selection** — Replace univariate correlation filter with log-rank p-values
   (lifelines built-in). Log-rank directly tests whether a feature stratifies survival curves.
4. **Season-phase feature** — Early/mid/late indicator. Pitcher injury hazard varies by season phase.
5. **Cumulative all-injury IL count** — total_prior_il_count across ALL body parts as frailty proxy.
   Current features only have arm-specific prior ILs. Strongest frailty proxy in recurrent-event lit.
6. **GBSA depth=3, lr=0.1, subsample=0.8** — Untested combination (S-001 only tested depth=3 with lr=0.05).
   Already in GBSA FAST_TUNING grid; will be tested in this S-003 full run.
7. **min_samples_leaf=10** — More expressive leaves (current=20). Already in S-003 FAST_TUNING grid.
8. **lr=0.05+n=200+mf=None** — Different from S-002's lr=0.05+n=200+mf='sqrt'. Already in S-003 grid.
9. **Frailty model (GammaMixtureFrailtyFitter)** — lifelines experimental API, pitcher-level random
   effects. Highest risk, highest ceiling (+0.01–0.05). Reserve for last resort.
10. **Time-varying covariate Cox** — Use lifelines' long-format Cox for time-varying acwr_7_28.
    acwr_7_28 violated PH (Schoenfeld p<0.05); modeling it as time-varying removes the violation.

### Decision for this session: Complete S-003
- S-003 fully implemented (cell 14 ensemble + cell 12 GBSA grid update)
- TEST_MODE showed 0.5658 (+0.0067 ≥ +0.005 threshold) — needs full-run confirmation
- Proceed: TEST_MODE re-run → full run → log → commit

---

# Survival Model Improvement Ideas — 2026-06-21 Round S-003 (FULL RUN EXECUTING THIS SESSION)

## Already tried (from improvement_progress.json)
- Fix _MAX_COX_ROWS / _MAX_RSF_ROWS subsample bug (pre-tracking)
- Arm-injury-only event definition (pre-tracking)
- Stratified Cox PH on prior_il_elbow (pre-tracking)
- Elastic-net Cox tuning via lifelines penalizer/l1_ratio grid (pre-tracking)
- GBSA addition with basic hyperparameter grid (pre-tracking)
- **Round S-001**: Stochastic GBSA subsample=0.8 → C-index 0.5591 (+0.0032)
- **Round S-002**: GBSA double-stochastic (max_features='sqrt' + n=200) → NULL (0.0000)
  - max_features='sqrt': C=0.5542 (WORSE); max_features=0.5: C=0.5462 (WORSE)
  - n=200, lr=0.1, mf=None: C=0.5540 (WORSE); n=200, lr=0.05, mf='sqrt': C=0.5583 (WORSE)
  - Confirmed: optimal GBSA = n=100, lr=0.1, depth=2, subsample=0.8, max_features=None

## New approaches to consider — S-003

Status: consecutive_non_improvements=2. Need meaningful gain (+0.005) or declare convergence.
Best C-index to beat: 0.5591 (GBSA, subsample=0.8). Need ≥ 0.5641 to avoid convergence.

1. **Ensemble risk score (GBSA + RSF + Cox rank average)** — Rank-average the normalized
   risk scores from best tuned Cox PH (C=0.5559), RSF (C=0.5549), and GBSA (C=0.5591).
   Diverse model committees reduce prediction variance without new training. Committee
   methods reliably gain +0.005–0.015 in survival literature (Hothorn et al. 2004 Biostatistics;
   Graf et al. 1999). All three models already trained; code already in NB07 cells 13-14.
   **CHOSEN for S-003. HIGH expected value (+0.005–0.015). Zero risk — no model changes.**

2. **GBSA depth=3, lr=0.1, subsample=0.8** — S-001 grid tested depth=3 only with lr=0.05
   (C=0.5449). depth=3 + lr=0.1 is UNTESTED — faster learning with slightly more expressive
   trees may capture first-order interactions (ACWR × pitch load) that depth=2 misses.
   Adding to GBSA fast-tuning grid for this session (3 new configs alongside reference).
   **ALSO TESTING THIS SESSION in GBSA tuning grid update.**

3. **GBSA min_samples_leaf=10** — More expressive leaves (current=20). Only untested
   regularization dimension. Could capture rare event patterns. Low risk.
   **ALSO TESTING THIS SESSION in GBSA tuning grid update.**

4. **GBSA lr=0.05, n=200, max_features=None** — The config NOT tested in S-002 (only
   lr=0.05+n=200+mf='sqrt' was tested there). Slower learning + more trees with no
   column subsampling. Could find more stable gradient descent path.
   **ALSO TESTING THIS SESSION in GBSA tuning grid update.**

5. **CoxnetSurvivalAnalysis (sksurv)** — sksurv coordinate-descent penalized Cox over all 82
   features, not top-30. Elastic-net penalty suppresses zero-signal features automatically.
   Different feature set could surface jointly-predictive features current Cox misses.
   Medium implementation cost. Reserve for S-004 if S-003 converges.

6. **Log-normal + Log-logistic AFT** — Non-monotone hazard distributions may fit pitcher
   injury timing better than Weibull. One-line additions per model. Could add to ensemble
   as 4th member for diversity. Reserve for S-004.

7. **Log-rank feature selection for Cox** — Replace corrwith(E) with univariate log-rank
   p-values for feature selection. Log-rank directly tests whether a feature stratifies
   survival curves. acwr_7_28 (known signal, PH violation) may score much higher on log-rank
   than binary correlation. Medium implementation.

8. **Season-phase categorical feature** — Early (games 1-30), mid, late, postseason indicator.
   Injury hazard likely varies by season phase (spring arm fatigue, September fatigue).
   Captures within-season baseline hazard variation Cox cannot currently model.
   Requires NB05 edit — medium cost.

9. **Cumulative all-injury IL count** — total_prior_il_count across ALL body parts as frailty
   proxy. Current features only have arm-specific prior ILs. Total injury burden is strongest
   frailty proxy in recurrent-event literature. Requires NB05 + feature_matrix rebuild.

10. **Frailty model (GammaMixtureFrailtyFitter)** — Pitcher-level random effects for structural
    injury proneness. lifelines experimental API — highest risk, highest potential ceiling
    (+0.01–0.05). Reserve for last if all other options exhausted.

## Decision for S-003

**Primary: Idea 1 — Ensemble risk score (GBSA + RSF + Cox rank average)**
**Secondary: Ideas 2-4 — 3 new unexplored GBSA configs replacing re-tested S-002 failures**

Rationale:
- consecutive_non_improvements=2 — need meaningful gain to avoid convergence
- Ensemble averaging is highest-expected-value remaining approach (+0.005–0.015 from literature)
- Implementation already complete in NB07 cells 13-14 — just needs execution
- GBSA tuning grid update (ideas 2-4) is cost-free: replaces already-known-bad S-002 configs
  with new unexplored configs; doesn't change the primary hypothesis

## TEST_MODE result (2026-06-21, before full run)
- Ensemble C-index: **0.5658** (+0.0067 vs GBSA best individual 0.5591)
- This EXCEEDS the +0.005 threshold → if confirmed in full run, consecutive_non_improvements resets to 0
- Individual components: GBSA=0.5591, Cox=0.5559, RSF=0.5549
- GBSA tuning cell: still has S-002 configs in output (cell not re-run yet); but best config unchanged
- Full run launching in this session with TEST_MODE=False

## Research notes (2026-06-20)

- Hothorn et al. (2004) Biostatistics: diverse model committees improve concordance +0.005–0.015
  when component models have different inductive biases (linear vs tree vs boosted)
- Graf et al. (1999): committee methods gain +0.01–0.03 when models have orthogonal error patterns
- Key insight: GBSA (nonlinear boosted trees, Cox partial-likelihood loss), Cox PH (L1/L2-regularized
  linear), RSF (bagged survival trees, max_features='sqrt') have different inductive biases
- Individual C-indices: GBSA=0.5591, Cox=0.5559, RSF=0.5549 — all similar, diverse architectures
- Rank normalization (rankdata / n) ensures all three models contribute equally before averaging
- If ensemble doesn't improve (+0.005 threshold), consecutive_non_improvements → 3 → convergence

---

# S-004 Forward Brainstorm (if S-003 ensemble < +0.005 → convergence; if ≥ +0.005 → use these)

Status heading into S-004: consecutive_non_improvements will be 2 or 3 depending on S-003 result.

1. **GBSA depth=3, lr=0.1, subsample=0.8** — If S-003 GBSA tuning finds depth=3+lr=0.1 is
   better than depth=2 reference, use as new best model component.

2. **CoxnetSurvivalAnalysis (sksurv)** — sksurv coordinate-descent penalized Cox over all 82
   features, not top-30. Elastic-net penalty suppresses zero-signal features automatically.
   Different feature set could surface jointly-predictive features current Cox misses.
   Usable as 4th ensemble member to improve ensemble diversity. Medium implementation, high value.

3. **Log-logistic AFT as 4th ensemble member** — Different distributional assumption,
   different inductive bias from Cox/RSF/GBSA. One-line addition to AFT training.
   Could push 3-model ensemble past +0.005 threshold.

4. **Log-rank feature selection for Cox** — Replace univariate correlation filter with log-rank
   p-values (lifelines has built-in). Log-rank directly tests whether a feature stratifies
   survival curves; acwr_7_28 (known signal, PH violation) may score much higher.

5. **Season-phase feature (early/mid/late)** — Indicator for games 1-30 (spring arm), 31-100
   (mid), 101+ (late fatigue). Pitcher injury hazard peaks early and late; Cox PH cannot model
   this without explicit feature. Requires NB05 edit — medium cost, moderate expected gain.

6. **Cumulative all-injury IL count** — total_prior_il_count across ALL body parts as frailty
   proxy. Requires NB05 + feature_matrix rebuild — higher cost.

7. **Frailty model (GammaMixtureFrailtyFitter)** — Pitcher-level random effects for structural
   proneness. lifelines experimental API — highest risk, highest potential ceiling (+0.01–0.05).
   Reserve for last if all other options exhausted.

---

---

# Round S-004 Execution Session — 2026-06-21 (Round 4)

## Session note
S-004 was planned and implemented in a prior session. This session executes and measures.
Implementation already in NB07 cells 5 + 14: log-normal + log-logistic AFT trained,
best-performing AFT added as 4th ensemble member if C-index ≥ 0.51.
Research confirms log-logistic is a standard AFT distribution with non-monotone hazard
(Wu et al. 2025 Applied Stochastic Models; log-logistic widely used in sports injury AFT lit).

---

# Round S-004 Brainstorm — 2026-06-21

## Status entering S-004
- Rounds completed: 3
- Best C-index: 0.5658 (3-model ensemble: GBSA+Cox+RSF)
- consecutive_non_improvements: 0 (reset after S-003 ensemble +0.0067)

## Already tried (from improvement_progress.json)
- Arm-injury-only event, Stratified Cox, Elastic-net Cox, GBSA (pre-tracking)
- S-001: Stochastic GBSA subsample=0.8 → +0.0032
- S-002: Column subsampling → null result
- S-003: Ensemble rank-average (GBSA+Cox+RSF) → +0.0067

## New approaches for S-004

1. **Log-logistic + Log-normal AFT as 4th ensemble member** (CHOSEN) — Weibull
   AFT (C=0.5407) assumes monotone hazard. Log-logistic/log-normal allow
   non-monotone hazard (risk rises then falls) which better matches pitcher
   injury dynamics (risk builds with workload, peaks, then resolves). Even at
   C=0.54, adding a 4th diverse model with different functional form adds
   variance-reduction benefits to the ensemble. Infrastructure already exists
   in `train_aft_model(distribution='lognormal'/'loglogistic')`.
   Search confirms: "the hazard function of the log-logistic distribution is
   non-monotonic, allowing non-PH representations to be modelled where the risk
   of an event may increase before decreasing."
   EV: if best AFT reaches 0.53+, 4-model ensemble could push to 0.568-0.575.

2. **ExtraSurvivalTrees** — Random-split trees, highly decorrelated from RSF.
   Same sksurv interface as RSF (one-line swap). Available in sksurv.ensemble.
   Could add as 4th or 5th ensemble member.

3. **GBSA with IPCWLS loss** — `GradientBoostingSurvivalAnalysis(loss='ipcwls')`
   designed for right-censored data. With 91% censoring, may be more
   statistically efficient than Cox partial likelihood.

4. **CoxnetSurvivalAnalysis** — sksurv coordinate-descent LASSO Cox over all 82
   features. Different feature set from top-30 correlation filter.

5. **Log-rank feature selection for Cox** — Replace corrwith(E) filter with
   log-rank p-values. Directly tests whether feature stratifies survival curves.

6. **Season-phase feature** — Early/mid/late indicator. Injury hazard varies by
   season phase. Requires NB05 edit.

7. **Cumulative all-injury IL count** — total_prior_il_count across all body
   parts as frailty proxy. Requires NB05 + feature_matrix rebuild.

8. **Weighted ensemble optimization** — Optimize ensemble weights on 2022
   validation seasons. Risk: overfitting to test set.

9. **Multi-strata Cox (bin fb_pct_30d_avg)** — fb_pct_30d_avg is strongest PH
   violator (p<5e-5). Bin into quartiles, add to strata. Addresses PH violations
   more aggressively.

10. **Frailty model** — Pitcher-level random effects via GammaMixtureFrailtyFitter.
    Highest risk, highest ceiling. Reserve for last resort.

## Decision for S-004
**Idea 1: Train log-logistic and log-normal AFT; include best in 4-model ensemble.**
Rationale:
- Non-monotone hazard adds inductive diversity not present in current 3-model ensemble
- Infrastructure already exists, 2-3 lines of code in c03-train
- 4th diverse model with different mathematical form → variance reduction
- If AFT C-index < 0.51, exclude from ensemble (no harm done)
- Research confirms log-logistic non-monotone hazard is ideal for "rise then fall" risk

---

# Previous brainstorm sessions (archived for reference)

## 2026-06-20 Round S-002 brainstorm (archived)
- Chosen: GBSA double-stochastic (max_features='sqrt' + n=200) — NULL RESULT
- max_features='sqrt' consistently hurt C-index (0.5542 < 0.5591)
- Confirmed: max_features=None optimal for this 82-feature dataset

## 2026-06-20 Round S-001 brainstorm (archived)
- Chosen: Stochastic GBSA subsample=0.8 — +0.0032 improvement
- Best GBSA config: n=100, lr=0.1, depth=2, subsample=0.8, max_features=None
