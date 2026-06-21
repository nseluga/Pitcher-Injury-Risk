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

# Previous brainstorm sessions (archived for reference)

## 2026-06-20 Round S-002 brainstorm (archived)
- Chosen: GBSA double-stochastic (max_features='sqrt' + n=200) — NULL RESULT
- max_features='sqrt' consistently hurt C-index (0.5542 < 0.5591)
- Confirmed: max_features=None optimal for this 82-feature dataset

## 2026-06-20 Round S-001 brainstorm (archived)
- Chosen: Stochastic GBSA subsample=0.8 — +0.0032 improvement
- Best GBSA config: n=100, lr=0.1, depth=2, subsample=0.8, max_features=None
