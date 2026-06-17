# Claude Instructions: Pitcher Injury Risk+ — Complete, Critique, and Improve

You are working on the `Pitcher-Injury-Risk` project.

**Phase 1 goal (MAY BE COMPLETE):** finish, debug, optimize, and validate notebooks 5–13.

**Phase 2 goal (runs automatically when Phase 1 is done):** critique each modeling notebook against baseball injury research, implement targeted improvements, rerun, and log. See the **Phase 2 Critique Protocol** section below.

---

## How to Determine Which Phase You Are In

Run `python scripts/verify_outputs.py` at session start.

- **Exit code 1 (failures exist):** You are in Phase 1. Follow the Session Protocol below to fix the first failing notebook.
- **Exit code 0 (all pass):** You are in Phase 2. Read `.scratch/critique_progress.json` and follow the Phase 2 Critique Protocol.

---

## Phase 2 Critique Protocol

The goal of Phase 2 is to make the first version of each modeling notebook *better*, not just *working*. You do this by:

1. Reading the notebook cell by cell.
2. Identifying every substantive modeling decision (feature choice, algorithm choice, label construction, evaluation metric, hyperparameter range, weighting scheme, threshold, etc.).
3. Researching each decision against the published baseball injury research literature using web search.
4. Writing a critique in `docs/model_critique_log.md`.
5. Implementing the highest-value improvements in `src/` modules and the notebook itself.
6. Rerunning and verifying.
7. Committing and moving on.

### Orientation

```bash
cat .scratch/critique_progress.json
cat docs/model_critique_log.md
```

`critique_progress.json` tracks which notebooks have been critiqued:

```json
{
  "05": {"status": "pending"},
  "06": {"status": "in_progress", "started": "2026-06-16T14:00Z"},
  "07": {"status": "done", "finished": "2026-06-16T15:00Z"}
}
```

Work on the **first notebook whose status is `pending`**, in order: 05, 06, 07, 08, 09, 10, 11, 12.

### Baseball Research Step (required, not optional)

Before writing the critique, use WebSearch to look up:

- Published peer-reviewed research on the specific injury risk factors being modeled in that notebook (e.g. "pitcher Tommy John surgery risk factors", "ACWR baseball injury risk", "pitcher velocity decline injury predictor", "pitch count fatigue biomechanics").
- Statcast-specific or sabermetric references when relevant (FanGraphs, Baseball Prospectus, Baseball Savant research articles, SABR proceedings).
- Any well-known industry standards or MLB workload policies (e.g. MLB's pitch count limits, rest requirements).

Record 3–5 key findings from your research in the critique log. These findings should directly motivate the improvements you implement.

### Critique Targets by Notebook

#### NB05 — Feature Engineering

Key questions to research and critique:
- Is the ACWR window (7-day acute, 28-day chronic) the correct choice for pitchers? What windows does the literature support?
- Are movement drift features computed correctly (rolling mean subtracted from current)? Do published studies use this formulation?
- Should pitch mix entropy be included? Is it predictive of injury in the literature?
- Is `intragame_velo_drop` correctly computed? What is the clinical threshold for concern?
- Are there published features from baseball injury research that are missing entirely?

#### NB06 — Baseline Models

Key questions:
- Is PR-AUC the right primary metric? What does the injury prediction literature use?
- Is class-weight balancing the right approach, or should oversampling (SMOTE) be used?
- Are the hyperparameter ranges reasonable? Does the literature suggest specific ranges?
- Is time-based splitting (by season) implemented correctly to prevent leakage?
- Are there model types used in published baseball injury prediction work that we have not tried?

#### NB07 — Survival Models

Key questions:
- Is the Cox Proportional Hazards assumption (proportional hazards) likely valid for pitcher injury? Research this.
- Is using 20 top features for Cox the right tradeoff between performance and assumption validity?
- What censoring assumptions does the literature use — is right-censoring at season end correct?
- Does the literature use Accelerated Failure Time models rather than Cox for this domain?

#### NB08 — Multitask Models

Key questions:
- Is "days missed" the right regression target for severity, or should it be log-transformed?
- Is the severity classification (mild/moderate/severe) at the right thresholds? What does the IL data support?
- Is chaining the models in probability → days_missed → severity the right order, or should severity predict days_missed?
- Are the multitask regression metrics (MAE/RMSE) appropriate for a highly right-skewed target?

#### NB09 — Injury Risk+ Construction

Key questions:
- Are the blend weights (injury probability, days missed, severity, time-to-injury) appropriately chosen? Does the literature suggest different weighting?
- Is era normalization being applied correctly? Should it be done per-season or per-era?
- Is the league-average = 100 normalization robust to the imbalanced dataset?
- Are the sanity checks sufficient? What edge cases could produce pathological scores?

#### NB10 — Interpretability

Key questions:
- Do the SHAP global feature importances align with what baseball injury research says are the strongest predictors? If not, that is a red flag for leakage or data issues.
- Are the PDP plots showing clinically plausible dose-response curves (e.g. injury risk increasing with pitch count)?
- Are there specific high-risk / high-surprise pitchers in the local explanations that suggest model errors?

#### NB11 — Baseball-Specific Insights

Key questions:
- Do the velocity × risk and workload × risk relationships match published findings?
- Are there published research findings about pitch-type risk (e.g. slider as highest UCL stress) that the notebook does not show?
- Are the archetype risk differences in the direction the literature predicts (e.g. high-velo power pitchers at higher risk)?

#### NB12 — Simulation

Key questions:
- Are the counterfactual simulations using the right feature perturbations (e.g. reducing `pitches_90d` without also reducing `acwr_7_28`)?
- Does the pitch count reduction → risk reduction curve show a clinically plausible shape?
- Are the slider reduction simulations capturing the right mechanism?

### Critique Log Format

Append one entry to `docs/model_critique_log.md` per notebook, in this format:

```markdown
## [2026-06-16 14:00] NB06 — Critique & Improvements

### Research Findings
1. [Smith et al. 2021] ACWR >1.5 associated with 2× injury risk in MLB pitchers — our ACWR window (7:28) matches the industry standard.
2. [Lyman et al. 2001] Pitch count is a significant predictor, but the relationship is non-linear above 75 pitches/game — XGBoost should capture this.
3. ...

### Decisions Critiqued
- **Class balancing:** Used `class_weight='balanced'` in sklearn models. Literature (Kovalchik & Reid 2019) recommends SMOTE for severe class imbalance in sports injury data. **Verdict: implement SMOTE as an alternative and compare PR-AUC.**
- **Hyperparameter ranges:** max_depth 2–10 for XGBoost is standard but wide. **Verdict: acceptable.**
- ...

### Improvements Implemented
1. Added SMOTE comparison in `src/models/baseline_models.py::train_gradient_boosting` — optional `use_smote=True` flag.
2. Tightened XGBoost depth range to 3–7 based on dataset size (~30K train rows).
3. ...

### Verified
- `run_notebooks.py --only 06` passed in Xs.
- `verify_outputs.py --only 06` PASS.
```

### After Critique

After a successful critique session:
1. Update `.scratch/critique_progress.json` — set the notebook's status to `done`.
2. Commit: `git add notebooks/NN_*.ipynb src/ docs/model_critique_log.md .scratch/critique_progress.json && git commit -m "NB06 critique: class balance + XGBoost depth range improved"`
3. Continue to the next `pending` notebook if you have capacity.

### Quality Bar

An improvement is worth implementing if it satisfies ALL of these:
- Grounded in baseball research (not just generic ML advice)
- Measurably changes a metric (PR-AUC, C-index, etc.) OR closes a known gap in the literature
- Does not break the notebook or introduce leakage
- Can be tested in TEST_MODE in under 2 minutes before the full run

If a decision cannot be improved — if it is already the right choice — write "**Verdict: no change — current approach matches literature**" in the critique log and move on. Do not change things for the sake of changing them.

---

# Phase 1 Session Protocol

*(Follow this only if `verify_outputs.py` returns exit code 1.)*

The goal of notebooks listed below is the original implementation of the content described further below.

The target notebooks are:

- `05_feature_engineering.ipynb`
- `06_baseline_models.ipynb`
- `07_survival_models.ipynb`
- `08_multitask_models.ipynb`
- `09_risk_score_construction.ipynb`
- `10_model_interpretability.ipynb`
- `11_baseball_specific_insights.ipynb`
- `12_usage_strategy_simulation.ipynb`
- `13_dashboard.ipynb`

Notebooks 10, 12, and 13 were previously stubs but are now implemented.
All notebooks are passing as of 2026-06-16. Phase 1 work is complete.

---

## Credit-Safe Execution Rule

Do not spend Claude turns waiting for long notebook execution.

Before launching any notebook, check whether another notebook process is already running:

````bash
ps aux | grep -E "run_notebooks|jupyter|nbconvert" | grep -v grep

If a notebook is already running, update .scratch/progress.json and stop.

If you need to run a full notebook, launch:

python run_notebooks.py --only NN --fail-fast

Then stop immediately unless the command finishes quickly.

Do not repeatedly poll.
Do not say you are waiting.
Do not start another Claude session just to monitor execution.
The user's laptop should do long execution; Claude should only diagnose, edit, launch, verify finished runs, and commit.

--

# Session Protocol (read this first, every session)

You are one iteration of an outer loop (`run_project.sh`). Previous sessions
may have already done part of the work; future sessions will pick up whatever
you leave behind. Therefore:

1. **Orient before doing anything.** Run:

   ```bash
   python scripts/verify_outputs.py
````

Then read `.scratch/progress.json`, `docs/notebook_debug_log.md`, and
`.scratch/nb_execution_summary.json` if they exist. Do not re-derive
project state from scratch — trust the verifier and the logs.

2. **Work on the FIRST failing notebook only.** Notebooks have a strict
   dependency order (05 → 06 → 07 → 08 → 09 → 10 → 11 → 12 → 13). Fixing a
   later notebook before an earlier one passes wastes work.

3. **Debug on a sample, validate on the full data.** When iterating on a fix,
   set `TEST_MODE = True` (or use a row sample) for fast cycles. Before
   declaring the notebook done, run it end-to-end in full mode:

   ```bash
   python run_notebooks.py --only NN --fail-fast
   ```

   **Mandatory test-mode gate before any full run:**
   After every code/notebook edit, run the notebook in TEST_MODE first:
   ```python
   # Temporarily set at the top of the notebook
   TEST_MODE = True
   ```
   Check that the cell outputs look correct and there are no errors.
   Only then switch to `TEST_MODE = False` and launch the full run.
   Never launch a full run on an unverified fix — a broken 10-minute run wastes
   more quota than a passing 30-second test run.

4. **Verify, then commit.** A notebook is done when
   `python scripts/verify_outputs.py --only NN` passes. Then immediately:
   - append a debug-log entry (format in Documentation Requirements below)
   - update `.scratch/progress.json`
   - `git add` the notebook, any `src/` changes, and docs, and commit with a
     message like `NB07 passing: survival models trained and verified`

   Committing per-notebook means a bad fix in a later session can never
   destroy passing work.

5. **The verifier's manifest is the contract.** If a notebook's design
   legitimately changes what files it outputs, update the `REQUIRED` dict in
   `scripts/verify_outputs.py` in the same session and record the reason in
   the debug log. Never delete a requirement just to make verification pass.

6. **Do not declare overall completion yourself.** The outer loop re-runs the
   verifier and stops when it passes. Your job each session is to move the
   first failing notebook to passing, then continue to the next if you have
   capacity.

## `.scratch/progress.json` format

```json
{
  "06": {
    "status": "fail",
    "last_error": "KeyError: 'spin_rate' in cell 14",
    "attempts": 2,
    "notes": "imputer fixed; tuning cells not yet run on full data",
    "updated": "2026-06-11T21:00:00Z"
  }
}
```

Keep entries short. `status` is one of: `pending`, `in_progress`, `fail`,
`pass`. Update it whenever a notebook's state changes.

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

# Python Environment (important — three interpreters exist)

- **Notebook cells execute under the `pitcher311` Jupyter kernel** →
  `/opt/homebrew/opt/python@3.11/bin/python3.11`. This has the full stack
  (pandas, sklearn, xgboost, lifelines, scikit-survival, plotly). If a
  notebook needs a new package (e.g. `shap` for notebook 10), install it
  there: `/opt/homebrew/opt/python@3.11/bin/python3.11 -m pip install shap`
- **`.venv/bin/python` (3.13)** only has nbconvert/nbformat — it drives
  `run_notebooks.py` and `scripts/verify_outputs.py` but cannot import pandas.
  Never install data-science packages here.
- The default `python3` kernel points at a third interpreter. Do not use it;
  `run_notebooks.py` already selects `pitcher311`.

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

1. Run notebook top-to-bottom: `python run_notebooks.py --only NN --fail-fast`
   (each run uses a fresh kernel, so restart safety is checked by construction).
2. Fix all errors. Prefer fixing logic in `src/` modules and keeping notebooks
   as thin orchestration over the modules — that is the established pattern.
3. Run again from scratch with the same command.
4. Confirm completion and outputs: `python scripts/verify_outputs.py --only NN`
5. Log, update progress.json, commit (see Session Protocol).

Do not patch isolated cells without verifying full notebook execution.

The final notebooks must be restart-safe.

Note: `run_notebooks.py` executes via nbconvert and saves partially executed
notebooks on failure — the traceback is inside the .ipynb and in
`.scratch/nb_execution_summary.json`. Read those instead of guessing.

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

# Notebook 10: Model Interpretability

## Goal

Understand what drives the model's predictions. Follow the plan already in
the notebook's first markdown cell and `docs/project_roadmap.md` Phase 7.

## Inputs

Best available classifier from notebook 6 (prefer the tuned XGBoost) and the
multitask model from notebook 8, plus the feature matrix.

## Outputs

```text
reports/figures/shap_global_importance.png
reports/figures/shap_beeswarm.png
reports/figures/partial_dependence_*.png
```

## Required Analyses

- SHAP global importance and beeswarm
- Partial dependence for the highest-impact workload, velocity, and
  injury-history features
- Local explanations / case studies: one high-risk and one low-risk pitcher
- Sanity commentary: do importances align with domain knowledge?

Use a row sample for SHAP if the full matrix is too slow or memory-heavy —
document the sample size.

---

# Notebook 11: Baseball-Specific Insights

Already written but never executed. Run it, fix what breaks, and verify its
input filenames against what actually exists on disk (e.g. it references
`data/processed/pitcher_archetypes.parquet` — confirm whether the real file
is `pitcher_clusters.parquet` and reconcile, fixing either the producer or
the consumer, not by duplicating data).

---

# Notebook 12: Usage Strategy Simulation

## Goal

Simulate alternative usage strategies using the trained models. Follow the
plan in the notebook's first markdown cell and roadmap Phase 9.

## Inputs

Trained models from notebooks 6–8 and `injury_risk_plus_scores.parquet`.

## Outputs

```text
reports/tables/simulation_results.csv
reports/figures/pitch_count_optimization.png
reports/figures/rest_schedule_optimization.png
```

## Required Simulations

Use the existing `src/simulation/` modules (`workload_simulator`,
`pitch_mix_simulator`, `usage_strategy_simulator`). Implement or repair them
as needed:

- pitch count reduction → change in predicted risk
- rest schedule variations
- pitch mix changes (e.g. slider reduction)
- role transition (starter → hybrid/reliever)

These are model-based counterfactuals, not causal estimates. Say so in the
notebook.

---

# Notebook 13: Dashboard

## Goal

Prototype the interactive dashboard (plotly) per the plan in the notebook's
first markdown cell. This is a prototyping notebook — the contract is that it
executes cleanly end-to-end, not that it produces files.

## Required Components

- pitcher lookup: Injury Risk+ with archetype-relative context
- season leaderboard with filters
- Injury Risk+ trend over time for a selected pitcher
- component breakdown view (probability / days missed / severity / survival)

Keep it laptop-safe: load only the scores parquet and slices of the feature
matrix, not the full pitch-level data.

---

# Notebook Style Guide

Match the conventions already established in notebooks 01–09 and the
`scripts/generate_notebook_*.py` generators:

1. **Cell structure.** First cell is markdown: `# NN — Title`, then
   `## Purpose` with a short prose paragraph and a numbered list of what the
   notebook does, then `## Target`/`## Inputs` where relevant. Every major
   step gets a numbered markdown header (`## 1. Load Feature Matrix`) with
   1–3 sentences of _why_, not just _what_.

2. **Markdown carries the reasoning, code stays clean.** Explanations,
   caveats, and methodology notes live in markdown cells (including small
   tables like the "Model | Why it's here" table in notebook 06). Inline `#`
   comments in code are sparse and only for non-obvious constraints.

3. **Logic lives in `src/`, notebooks orchestrate.** Import functions from
   `src/models/`, `src/scoring/`, `src/simulation/` and call them. If a
   notebook needs new logic, add it to the appropriate `src/` module.

4. **Every code cell prints evidence.** Shapes, rates, season lists, file
   paths written — formatted like `print(f'Train: {X.shape[0]:,} rows | positive rate = {y.mean():.1%}')`.
   A cell that runs silently is unverifiable.

5. **Fail loudly on missing inputs:**

   ```python
   if not fm_path.exists():
       raise FileNotFoundError('Run notebook 05 first to build the feature matrix.')
   ```

6. **New notebooks are built via generator scripts.** For notebooks 10, 12,
   and 13, write `scripts/generate_notebook_NN.py` following the existing
   `md()`/`code()` pattern, run it to produce the .ipynb, then execute. This
   keeps the notebook source reviewable and regenerable.

---

# Documentation Requirements

Create or update:

```text
docs/notebook_debug_log.md
docs/data_dictionary.md
docs/project_roadmap.md
README.md
```

## Debug Log Entry Format

Append one entry per fix, newest at the bottom, in exactly this shape so the
log stays parseable across sessions:

```markdown
## [2026-06-11 21:04] NB06 cell 14 — KeyError: 'spin_rate'

- **Cause:** column renamed to `release_spin_rate` in NB05 refactor
- **Fix:** updated feature list in cell 3; same fix applied in src/models/baseline_models.py
- **Assumptions/limitations:** none
- **Verified:** `run_notebooks.py --only 06` passed; `verify_outputs.py --only 06` passed
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

The single source of truth is:

```bash
python scripts/verify_outputs.py
```

Task is complete only if it exits 0, which requires:

1. Notebook 5 runs safely on the full dataset.
2. Notebook 6 trains and evaluates baseline models.
3. Notebook 7 trains survival models or documents fallback.
4. Notebook 8 produces multitask outputs.
5. Notebook 9 creates Injury Risk+.
6. Notebook 10 produces SHAP and partial dependence analyses.
7. Notebook 11 executes end-to-end.
8. Notebook 12 produces simulation results.
9. Notebook 13 executes end-to-end.
10. Hyperparameter tuning completes.
11. Calibration analysis completes.
12. All required files exist.
13. Pipeline can be restarted from scratch.
14. Debug log documents all work and each passing notebook is committed.

Prioritize:

1. Correctness
2. Stability
3. Reproducibility
4. Calibration quality
5. Predictive performance

Do not prioritize complexity for its own sake.
