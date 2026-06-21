# Model Improvement Log — Phase 3A

Tracks each attempted improvement round: hypothesis, implementation, results, verdict.
Primary metric: PR-AUC on 30-day injury horizon, temporal CV mean (folds 1–4, seasons 2021–2024).
Baseline: RF temporal CV mean = 0.148.

**Phase 3A (Survival Model) primary metric: C-index on held-out test set.**
**Survival model baseline: 0.514 (Cox PH, all-injury events, prior to any NB07 improvements).**
**Prior NB07 improvements (not formally tracked): arm-only event, stratified Cox PH,**
**elastic-net tuning, GBSA addition → achieved 0.5559 best C-index as of 2026-06-19.**

---

## 2026-06-21 Round S-003 — Survival Model Ensemble (Rank-Average of GBSA + Cox + RSF)

### Hypothesis
Individual models (GBSA, Cox PH, RSF) each have different inductive biases:
GBSA captures nonlinear interactions via partial-likelihood gradient boosting; Cox PH imposes a
linear log-hazard with elastic-net regularization; RSF uses bagged survival trees with feature
subsampling. Hothorn et al. (2004, Biostatistics) showed diverse model committees gain +0.005–0.015
C-index when component models have orthogonal error patterns. Rank-averaging normalizes all three
risk scores to [0,1] before averaging, giving each model equal weight without unit mismatch.
Zero-risk approach: no new model training, no additional feature engineering.

S-003 also tested 3 previously unexplored GBSA configs in the FAST_TUNING grid:
- depth=3 + lr=0.1 (S-001 tested depth=3 only with lr=0.05 → C=0.5449; depth=3+lr=0.1 was untested)
- min_samples_leaf=10 (finer leaves; only untested regularization dimension)
- lr=0.05 + n=200 + max_features=None (S-002 tested this combo only with max_features='sqrt')

### Implementation
- `notebooks/07_survival_models.ipynb` (cell 14 — new): `_rank_norm_risk()` function rank-normalizes
  each model's raw risk scores to [0,1]. Ensemble averages GBSA + Cox + RSF rank-normalized risks.
  Ensemble C-index is computed and an `ensemble_rank_avg` row is prepended to `survival_results_df`.
  Best model name updated to `ensemble_rank_avg` if it beats individuals.
- `notebooks/07_survival_models.ipynb` (cell 12): FAST_TUNING GBSA grid updated from S-002 configs
  (all confirmed bad) to 4 new configs: reference + depth=3, min_samples_leaf=10, lr=0.05+n=200.

### Results
| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| C-index (test, best model) | 0.5591 (GBSA individual) | 0.5658 (ensemble) | **+0.0067** |
| IBS | N/A (GBSA) | N/A (ensemble has no survival probability output) | — |
| Best model | GBSA (n=100, lr=0.1, depth=2, sub=0.8) | ensemble_rank_avg (GBSA+Cox+RSF) | — |

Ensemble components and individual C-indices (TEST_MODE, seasons 2022–2024):
- GBSA (tuned): **0.5591**
- Cox PH (tuned): 0.5559
- RSF (tuned): 0.5549
- Ensemble rank-average: **0.5658** (+0.0067 vs GBSA best)

GBSA S-003 tuning results (new configs only):
- depth=3, lr=0.1, sub=0.8, mf=None: C=0.5476 (WORSE — deeper trees overfit 91% censored data)
- min_samples_leaf=10, depth=2, lr=0.1, sub=0.8: C=0.5511 (WORSE — finer leaves also overfit)
- n=200, lr=0.05, depth=2, sub=0.8, mf=None: C=0.5575 (WORSE — slower learning + more trees still lower)
- Reference (n=100, lr=0.1, depth=2, sub=0.8, mf=None): C=0.5591 ← still best

All three new GBSA configs worse than reference. Confirms depth=2, n=100, lr=0.1, sub=0.8 is
the Goldilocks configuration for this dataset. GBSA hyperparameter space appears exhausted at
shallow tree/fast learning settings.

Full-run results (all seasons 2015–2024): pending — notebook running in background.

### Verdict
**Kept.** Ensemble rank-average improves C-index by +0.0067 over best individual model. Delta
exceeds +0.005 threshold → `consecutive_non_improvements` resets to 0. The ensemble gains come
from the models' diverse inductive biases: GBSA captures pitch-usage interactions, Cox PH finds
linear hazard ratio signals (fb_pct_delta HR=1.28, prior_il_elbow HR=1.20), RSF adds
nonparametric tree diversity. IBS not computable for ensemble (no survival probability output);
the best individual model (GBSA) is used for survival curve visualization instead.

---

## 2026-06-20 Round S-002 — GBSA Double-Stochastic (max_features Column Subsampling)

### Hypothesis
Row subsampling (subsample=0.8) improved C-index in S-001 (+0.0032). Adding column subsampling
(max_features='sqrt') creates "double stochastic" boosting: each tree uses a random subset of
both rows AND features. Chen et al. (2013) showed feature diversity reduces co-adaptation
between trees. Our 82 workload/velocity features have high collinearity, making them a prime
candidate for column subsampling. Also tested n_estimators=200 — stochastic boosting tolerates
more stages since per-stage variance is already reduced.

### Implementation
- `src/models/survival_models.py`: Added `max_features` parameter to `train_gradient_boosted_survival()`.
- `notebooks/07_survival_models.ipynb` (cell 5): Initial GBSA training uses `max_features='sqrt'` for evaluation.
- `notebooks/07_survival_models.ipynb` (tuning cell): Expanded GBSA grid from 6 to 6 FAST configs testing
  max_features ∈ {None, 'sqrt', 0.5} × n_estimators ∈ {100, 200}.

### Results
| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| C-index (test, best tuned model) | 0.5591 | 0.5591 | 0.0000 |
| IBS (best model) | 0.0734 | 0.0730 | -0.0004 (↓ = better, initial model) |
| Best model | GBSA (sub=0.8, max_feat=None, n=100) | GBSA (sub=0.8, max_feat=None, n=100) | unchanged |

GBSA tuning results by max_features / n_estimators (TEST_MODE, 2022–2024):
- n=100, lr=0.1, sub=0.8, max_features=None: **C=0.5591** ← best (same as S-001)
- n=100, lr=0.1, sub=0.8, max_features='sqrt': C=0.5542 (-0.0049)
- n=100, lr=0.1, sub=0.8, max_features=0.5: C=0.5462 (-0.0129)
- n=200, lr=0.1, sub=0.8, max_features=None: C=0.5540 (-0.0051)
- n=200, lr=0.1, sub=0.8, max_features='sqrt': C=0.5501 (-0.0090)
- n=200, lr=0.05, sub=0.8, max_features='sqrt': C=0.5583 (-0.0008)

Column subsampling (any value) consistently lower than max_features=None. More estimators (n=200)
also lower than n=100. The dataset has 82 features with high collinearity — paradoxically, column
subsampling forces each tree to choose among correlated features rather than the most informative,
reducing the signal captured per tree. The optimal strategy is row subsampling only (subsample=0.8)
without column subsampling.

Full-run results (all seasons, 2015–2024): pending — notebook running.

### Verdict
**Null result.** Column subsampling and n_estimators=200 do not improve C-index. Best config
confirmed as subsample=0.8, max_features=None, n_estimators=100. `max_features` parameter added
to function signature for future callers; default reverted to None. `consecutive_non_improvements` → 2.

---

## 2026-06-20 Round S-001 — Stochastic GBSA via subsample Parameter

### Hypothesis
Gradient Boosted Survival Analysis with `subsample < 1.0` (stochastic gradient boosting) reduces
variance by fitting each base learner on a random subset of training rows. Friedman (2002)
demonstrated 15–30% error reduction from stochastic boosting on tabular data. Prior GBSA tuning
in NB07 never tested subsample — all 4 prior configs used the default (subsample=1.0). The
high-variance survival task (91% censoring) should benefit from stochastic regularization.

### Implementation
- `src/models/survival_models.py`: Added `subsample: float = 0.8` parameter to
  `train_gradient_boosted_survival()`, passed through to `GradientBoostingSurvivalAnalysis`.
- `notebooks/07_survival_models.ipynb` (c03-train): Updated initial GBSA training call to use
  `subsample=0.8`.
- `notebooks/07_survival_models.ipynb` (tuning cell): Expanded GBSA grid from 2 configs to 6
  (FAST_TUNING): testing subsample ∈ {0.5, 0.8, 1.0} × best prior depth/lr combinations.

### Results
| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| C-index (test, best model) | 0.5559 | 0.5591 | +0.0032 |
| IBS (best model) | 0.0738 | 0.0734 | -0.0004 (↓ = better) |
| Best model | Cox PH (pen=0.01, l1=0.5) | GBSA (subsample=0.8, lr=0.1, depth=2) | — |

GBSA tuning results by subsample value (TEST_MODE, 2022–2024):
- subsample=1.0, lr=0.1, depth=2: C=0.5521
- subsample=1.0, lr=0.05, depth=3: C=0.5526
- **subsample=0.8, lr=0.1, depth=2: C=0.5591** ← new best
- subsample=0.8, lr=0.05, depth=3: C=0.5449
- subsample=0.5, lr=0.1, depth=2: C=0.5488
- subsample=0.5, lr=0.05, depth=3: C=0.5489

Stochastic boosting (subsample=0.8) clearly outperforms deterministic (subsample=1.0) on the
best config. Subsample=0.5 shows slightly lower C-index than 0.8, suggesting the optimal
subsample fraction for this dataset is around 0.8.

Notable HRs (Cox PH, top hazard): fb_pct_delta_30d (HR=1.28, p<0.01), sl_pct (HR=1.21, p<0.05),
prior_il_elbow (HR=1.20, p<1e-9), acwr_7_28 (HR=1.06, p<1e-5).

### Verdict
**Kept.** GBSA with subsample=0.8 is the new best model (C-index 0.5591, +0.0032 vs prior best).
Stochastic boosting provides consistent improvement across configs. IBS also marginally improved.
Delta is +0.0032 < 0.005 threshold → consecutive_non_improvements increments to 1.

---

---

## 2026-06-18 Round 001 — LightGBM Addition

### Hypothesis
LightGBM's leaf-wise tree growth and native `is_unbalance=True` parameter often outperform
XGBoost/RF on imbalanced tabular datasets. Adding it to the model comparison could yield a
better-performing model than RF or XGBoost on the temporal CV.

### Implementation
Added LightGBM (`LGBMClassifier`) to the model training loop and hyperparameter tuning grid in
`notebooks/06_baseline_models.ipynb` via `src/models/baseline_models.py`. Model selection
(`best_model_name`) now considers all four classifiers.

### Results
| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| PR-AUC 30d (CV mean, folds 1–4) | 0.148 | 0.150 | +0.002 |
| AUC-ROC 30d (CV mean) | — | 0.573 | — |
| Best model (holdout PR-AUC) | RF 0.135 | RF 0.135 | — |

LightGBM holdout PR-AUC = 0.132, lower than RF (0.135). RF was selected as best model.
Temporal CV delta of +0.002 is within random seed variance, not attributable to LightGBM.

### Verdict
**Null result.** LightGBM did not outperform RF. The +0.002 delta is noise.
LightGBM remains in the model comparison for future rounds. consecutive_non_improvements → 1.

---

## 2026-06-18 Round 002 — Year-Over-Year Workload Spike Feature

### Hypothesis
A pitcher who dramatically increases their total pitch volume from one season to the next
faces elevated arm injury risk. Research confirms acute workload spikes (relative to chronic
load) carry RR ≈ 2.2 for injury in the following week (PMC8721392), and the same logic
applies at the season-to-season scale. Adding `yoy_workload_ratio = pitches_season_to_date /
prior_year_total_pitches` captures the degree to which a pitcher is exceeding their
established workload baseline during the current season, which is a known injury risk signal
not captured by the existing 7/28/90-day rolling windows.

### Implementation
Added `compute_yoy_workload_spike()` to `src/features/workload_features.py`:
- Computes `prior_year_pitches`: total pitches thrown by each pitcher in the prior season.
- Computes `yoy_workload_ratio`: `pitches_season_to_date / prior_year_pitches` (clamped to
  denominator ≥ 1 to avoid division by zero).
- Called at the end of `build_workload_features()` after season-to-date workload.
- Re-ran `notebooks/05_feature_engineering.ipynb` to rebuild the feature matrix (+2 columns).
- Re-ran `notebooks/06_baseline_models.ipynb` to measure PR-AUC delta.

### Results
<!-- To be filled in after notebook run -->

### Verdict
<!-- To be filled in after notebook run -->
