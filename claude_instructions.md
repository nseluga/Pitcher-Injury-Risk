# Claude Instructions: Complete, Debug, Tune, and Validate Notebooks 5–9

You are working on the `Pitcher-Injury-Risk` project.

The goal is to finish, debug, optimize, and validate the following notebooks:

- `05_feature_engineering.ipynb`
- `06_baseline_models.ipynb`
- `07_survival_models.ipynb`
- `08_multitask_models.ipynb`
- `09_risk_score_construction.ipynb`

By the end of your run each of these should be able to run as intended. Do not work on notebooks 10–12 yet.

---

# Primary Objective

Produce a stable, reproducible, restart-safe pipeline that:

1. Runs successfully on the full dataset.
2. Avoids laptop memory crashes.
3. Produces all required intermediate outputs.
4. Trains and evaluates models.
5. Performs appropriate hyperparameter tuning.
6. Constructs the final Injury Risk+ score.
7. Documents all fixes, assumptions, and limitations.

The goal is not just to make notebooks run.

The goal is to create the strongest possible first version of the modeling pipeline.

---

# Critical Context

Notebook 5 previously crashed due to excessive memory usage.

VS Code and Python consumed extremely large amounts of RAM.

Therefore:

## Memory Safety Is Mandatory

Do not load the entire pitch-level dataset into memory if avoidable.

Use:

- batching
- checkpointing
- parquet storage
- memory logging
- garbage collection

The project must be capable of running on a laptop.

---

# General Workflow Requirements

For each notebook:

1. Run notebook top-to-bottom.
2. Fix all errors.
3. Restart kernel.
4. Run again from scratch.
5. Confirm completion.
6. Confirm outputs exist.
7. Confirm restart safety.

Do not patch isolated cells without verifying full notebook execution.

The final notebooks must be restart-safe.

---

# Directory Standards

Use:

```text
data/raw/
data/processed/
data/processed/features/

models/

reports/figures/
reports/tables/

docs/
```

Create missing directories automatically.

Use Parquet for large datasets.

Avoid CSV for large intermediate outputs.

---

# Memory Safety Requirements

Notebook 5 must:

1. Process data in chunks when possible.
2. Aggregate pitch-level data as early as possible.
3. Save intermediate outputs.
4. Delete large objects after use.
5. Run gc.collect().
6. Downcast numeric columns.
7. Convert repeated strings to categories.
8. Avoid unnecessary dataframe copies.
9. Avoid repeated huge merges.
10. Log memory usage after major operations.
11. Include checkpoints.
12. Support:

```python
TEST_MODE = False
FULL_MODE_BATCHED = True
```

The notebook should function as an orchestration notebook rather than one giant in-memory workflow.

---

# Notebook 5: Feature Engineering

## Goal

Build the final model-ready feature matrix.

## Inputs

Use cleaned outputs from notebooks 1–3.

Inspect existing files before assuming filenames.

## Required Outputs

```text
data/processed/features/workload_features.parquet
data/processed/features/velocity_features.parquet
data/processed/features/pitch_mix_features.parquet
data/processed/features/movement_features.parquet
data/processed/features/injury_history_features.parquet

data/processed/feature_matrix.parquet
```

## Required Feature Groups

### Workload

Examples:

- pitches per outing
- appearances last 7 days
- pitches last 7 days
- pitches last 30 days
- acute workload
- chronic workload
- ACWR
- days rest
- short-rest flag
- consecutive appearances

### Velocity

Examples:

- avg fastball velocity
- max fastball velocity
- velocity trends
- velocity loss from peak
- velocity spike flags

### Pitch Mix

Examples:

- fastball %
- slider %
- breaking ball %
- offspeed %
- pitch mix entropy
- pitch mix changes

### Movement / Mechanics Proxies

Examples:

- release height
- release side
- extension
- horizontal break
- vertical break
- spin rate
- release point drift
- movement drift
- spin drift

### Injury History

Examples:

- prior injuries
- prior IL stints
- prior days missed
- prior severe injury flag
- days since injury

## Label Construction

Create:

- injury within 30 days
- injury within 60 days
- injury within 90 days
- injury within 180 days
- days until injury
- expected days missed
- severity class

Prevent future leakage.

Only use information available before the prediction date.

## Validation

Include:

- matrix shape
- feature counts
- missingness summary
- label distributions
- memory summary
- sample rows
- saved-file confirmation

---

# Notebook 6: Baseline Models

## Goal

Build baseline injury prediction models.

## Inputs

```text
data/processed/feature_matrix.parquet
```

## Outputs

```text
models/baseline_logistic.joblib
models/baseline_random_forest.joblib
models/baseline_xgboost.joblib

reports/tables/baseline_model_metrics.csv
reports/figures/*
```

## Models

Train:

- Logistic Regression
- Random Forest
- XGBoost (if available)

## Evaluation

Include:

- ROC AUC
- PR AUC
- Brier Score
- Calibration
- Confusion Matrix
- Injured-class recall

Use:

- class weights
- scale_pos_weight where appropriate

Use time-aware splitting if dates exist.

Avoid leakage.

---

# Notebook 6 Hyperparameter Tuning

Add tuning for:

## Logistic Regression

Tune:

- C
- penalty
- class_weight

## Random Forest

Tune:

- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf
- max_features
- class_weight

## XGBoost

Tune:

- max_depth
- learning_rate
- n_estimators
- subsample
- colsample_bytree
- min_child_weight
- gamma
- reg_alpha
- reg_lambda
- scale_pos_weight

Use:

- RandomizedSearchCV first
- GridSearchCV only for final refinement

Metrics:

- PR AUC
- ROC AUC
- Brier Score
- Calibration

Required outputs:

```text
reports/tables/hyperparameter_tuning_results.csv
reports/tables/tuned_baseline_model_metrics.csv

models/baseline_logistic_tuned.joblib
models/baseline_random_forest_tuned.joblib
models/baseline_xgboost_tuned.joblib
```

---

# Notebook 7: Survival Models

## Goal

Model time-to-injury.

## Outputs

```text
models/survival_cox.pkl
models/survival_rsf.pkl

reports/tables/survival_model_metrics.csv
reports/figures/*
```

## Models

Use:

- Cox Proportional Hazards
- Random Survival Forest if feasible

Handle censoring correctly.

Non-injured observations are censored, not permanently healthy.

## Metrics

- Concordance Index
- Risk stratification
- Survival curves
- Calibration if feasible

---

# Notebook 7 Hyperparameter Tuning

Tune:

## Cox

- penalizer
- l1_ratio

## Random Survival Forest

- n_estimators
- max_depth
- min_samples_split
- min_samples_leaf
- max_features

Metric:

- Concordance Index

Outputs:

```text
reports/tables/survival_hyperparameter_tuning_results.csv
```

---

# Notebook 8: Multitask Models

## Goal

Predict multiple components of injury risk.

## Targets

- injury probability
- days missed
- severity
- time-to-injury

## Outputs

```text
models/multitask_chained.joblib

reports/tables/multitask_model_metrics.csv
reports/figures/*
```

Acceptable implementations:

- chained models
- multi-output models
- shared preprocessing with separate targets

Prioritize stability.

---

# Notebook 8 Hyperparameter Tuning

Tune each submodel.

Metrics:

## Classification

- PR AUC
- ROC AUC
- Recall
- Brier Score

## Regression

- MAE
- RMSE

## Severity

- Macro F1
- Weighted F1

## Survival

- Concordance Index

Outputs:

```text
reports/tables/multitask_hyperparameter_tuning_results.csv
reports/tables/tuned_multitask_model_metrics.csv

models/multitask_chained_tuned.joblib
```

---

# Laptop-Safe Tuning

Create:

```python
RUN_TUNING = True
FAST_TUNING = True
N_ITER_TUNING = 20
```

Do not make tuning impossible to run.

Use:

- checkpoints
- saved tuning results
- limited search spaces

---

# Calibration

For tuned models:

Apply calibration where appropriate:

- Platt Scaling
- Isotonic Regression

Generate calibration plots.

Because Injury Risk+ depends on probabilities, calibration matters.

---

# Notebook 9: Injury Risk+ Construction

## Goal

Construct final Injury Risk+ score.

## Inputs

Outputs from notebooks 6–8.

## Outputs

```text
data/processed/injury_risk_plus_scores.parquet

reports/tables/injury_risk_plus_leaderboard.csv
reports/tables/risk_score_component_summary.csv
reports/tables/risk_score_model_sources.csv

reports/figures/injury_risk_plus_distribution.png
reports/figures/risk_score_components.png
```

---

# Injury Risk+ Definition

Like OPS+.

League average:

```text
100
```

Higher:

```text
More injury risk
```

Lower:

```text
Less injury risk
```

Example:

```text
150 = 50% riskier than average
80 = 20% safer than average
```

Construct:

Raw Risk = weighted combination of:

1. Injury probability
2. Expected days missed
3. Severity risk
4. Time-to-injury risk

Then:

```python
InjuryRiskPlus = (
    raw_risk /
    league_average_raw_risk
) * 100
```

Document weights.

If weights are first-draft estimates, state so explicitly.

---

# Risk Score Validation

Perform:

- Distribution analysis
- Leaderboards
- Component breakdowns
- Sensitivity testing

Sanity checks:

- Prior injuries generally increase risk
- Missing values do not create extreme scores
- Low workload does not create absurd risk
- League average ≈ 100

---

# Model Selection Logic

Use:

1. Tuned multitask model
2. Untuned multitask model
3. Tuned baseline/survival models
4. Untuned baseline/survival models

Document what was used.

Create:

```text
reports/tables/risk_score_model_sources.csv
```

Columns:

- component
- model file
- tuned
- validation metric
- notes

---

# Documentation Requirements

Create or update:

```text
docs/notebook_5_to_9_debug_log.md
docs/data_dictionary.md
docs/project_roadmap.md
README.md
```

Debug log must include:

- errors found
- fixes applied
- assumptions
- limitations
- outputs created
- rerun instructions

---

# Methodology Constraints

Do NOT:

- allow leakage
- use post-injury data in features
- make causal claims
- optimize for accuracy alone
- silently drop data
- fabricate labels

If labels are insufficient:

Document the limitation.

Implement a reasonable fallback.

---

# Final Completion Criteria

Task is complete only if:

1. Notebook 5 runs safely on the full dataset.
2. Notebook 6 trains and evaluates baseline models.
3. Notebook 7 trains survival models or documents fallback.
4. Notebook 8 produces multitask outputs.
5. Notebook 9 creates Injury Risk+.
6. Hyperparameter tuning completes.
7. Calibration analysis completes.
8. All required files exist.
9. Pipeline can be restarted from scratch.
10. Debug log documents all work.

Prioritize:

1. Correctness
2. Stability
3. Reproducibility
4. Calibration quality
5. Predictive performance

Do not prioritize complexity for its own sake.
