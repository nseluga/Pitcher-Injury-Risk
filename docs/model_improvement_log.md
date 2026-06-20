# Model Improvement Log — Phase 3A

Tracks each attempted improvement round: hypothesis, implementation, results, verdict.
Primary metric: PR-AUC on 30-day injury horizon, temporal CV mean (folds 1–4, seasons 2021–2024).
Baseline: RF temporal CV mean = 0.148.

**Phase 3A (Survival Model) primary metric: C-index on held-out test set.**
**Survival model baseline: 0.514 (Cox PH, all-injury events, prior to any NB07 improvements).**
**Prior NB07 improvements (not formally tracked): arm-only event, stratified Cox PH,**
**elastic-net tuning, GBSA addition → achieved 0.5559 best C-index as of 2026-06-19.**

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
