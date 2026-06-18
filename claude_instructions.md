# Claude Instructions: Pitcher Injury Risk+ — Phase 3

You are working on the `Pitcher-Injury-Risk` project.

## Status

**Phase 1 (Implementation): COMPLETE.**
All notebooks 01–13 implemented, executed on the full 2015–2024 dataset, and verified by `python scripts/verify_outputs.py`.

**Phase 2 (Baseball Research Critique & Improvement): COMPLETE.**
All modeling notebooks 05–12 critiqued against published literature, improvements applied and re-verified. Full record in `docs/model_critique_log.md`.

**Phase 3: IN PROGRESS — two parallel goals:**
- **3A — Model Improvement:** Iteratively improve model discrimination (PR-AUC) and calibration on the 30-day injury prediction task.
- **3B — Analysis Dashboard:** Build a unified Streamlit dashboard combining the NB13 prototype components into a genuinely usable analysis tool.

Phase 3A runs first. Once `_meta.status == "converged"` in `.scratch/improvement_progress.json`, move to Phase 3B.

---

## How to Orient at Session Start

```bash
# Check what phase of 3 you're in
cat .scratch/improvement_progress.json 2>/dev/null || echo "IMPROVEMENT NOT STARTED"
cat .scratch/dashboard_progress.json 2>/dev/null || echo "DASHBOARD NOT STARTED"

# Current best metrics
cat reports/tables/baseline_model_metrics.csv
cat reports/tables/tuned_baseline_model_metrics.csv

# What's already been tried
cat docs/model_improvement_log.md 2>/dev/null | tail -120
```

---

## Phase 3A — Model Improvement Protocol

### Goal

The current models are weak. Key baseline metrics (temporal CV, folds 1–4):
- **XGBoost 30d:** AUC-ROC = 0.578, PR-AUC = 0.127
- **Random Forest 30d:** AUC-ROC = 0.586, PR-AUC = 0.137
- **Survival (RSF) C-index:** 0.514 (near-random)

The primary metric is **PR-AUC on the 30-day injury horizon, averaged across temporal CV folds 1–4** (exclude fold 0 = 2020, COVID year). Target: push PR-AUC above 0.20. Secondary metrics: AUC-ROC, Brier score.

### Improvement Categories to Explore

Work through these roughly in priority order. Do not repeat an approach that has already been logged as tried. Read `docs/model_improvement_log.md` first.

#### Feature Engineering (highest expected value)
- **Year-over-year workload spike:** `season_pitches_current / season_pitches_prior_year`. Pitchers crossing a large YoY increase are high-risk.
- **Irregular rest:** binary flag for `days_rest < 4 OR days_rest > 12`. Both extremes are injury-associated.
- **Age × workload interaction:** `age * pitches_90d`. Older pitchers tolerate less accumulated load.
- **Pitch shape deterioration:** rolling std of `spin_rate`, `pfx_x`, `pfx_z` — increasing variance signals mechanical breakdown.
- **Better ACWR windows:** add 5:14 and 3:21 ACWR ratios alongside current 7:28.
- **First-career-high-workload flag:** first season crossing 150 IP or 2500 pitches — high-risk transition.
- **Consecutive high-effort starts:** count of prior starts with pitch_count > 100 in last 3 starts.
- **Command drift:** rolling trend in zone% or first-pitch-strike% (declining = fatigue signal).

#### Label / Training Strategy
- **Arm-specific injury labels:** re-label `injured_next_30d` using only shoulder/elbow/forearm IL stints from the injury database, not all IL types. This removes noise from non-arm injuries. Check `data/processed/injuries.parquet` for injury type columns.
- **Focal loss for XGBoost:** use `scale_pos_weight` tuned to maximize PR-AUC rather than accuracy. Try values from 3× to 10× the class imbalance ratio.
- **SMOTE only within CV folds:** verify that SMOTE is applied inside each training fold, never to the held-out test set. If leakage exists, fix it.
- **Positive-unlabeled (PU) correction:** pitchers who avoided the IL may have had unreported injuries. Apply Elkan & Noto (2008) PU correction as a post-processing step on predictions.

#### Model Architecture
- **LightGBM:** try LightGBM as an alternative to XGBoost. Often outperforms on imbalanced tabular data. Add to NB06 alongside existing models.
- **Feature selection:** use permutation importance from temporal CV to identify top-25 features and retrain on that reduced set. Many near-zero-SHAP features add noise.
- **Stacking ensemble:** use out-of-fold predictions from RF, XGBoost, LightGBM, and the RSF hazard score as inputs to a logistic regression meta-learner.
- **Calibration post-processing:** apply Platt scaling (sigmoid calibration) or isotonic regression calibration to the best classifier's output and verify Brier score improves.

#### Threshold Optimization
- **F2-optimal threshold:** tune the decision threshold to maximize F2 score (β=2, recall-weighted) on the temporal CV validation folds. Report precision and recall at that threshold.
- **Precision-at-recall curves:** target recall ≥ 0.50 and report what precision is achievable at that operating point.

### Improvement Protocol — One Round Per Session

**Step 1 — Orient:**
```bash
cat .scratch/improvement_progress.json
cat docs/model_improvement_log.md | tail -120
cat reports/tables/baseline_model_metrics.csv
```

**Step 2 — Pick the next idea:**
- Read `improvement_progress.json` to see what's been tried and what the deltas were.
- Pick the highest-expected-value idea NOT yet attempted from the categories above.
- If all obvious ideas are exhausted and 3+ consecutive rounds showed PR-AUC delta < 0.005, set `_meta.status = "converged"` in `improvement_progress.json` and stop Phase 3A.

**Step 3 — Research (1–2 WebSearches):**
Run targeted searches to ground the approach in evidence before implementing. Examples:
- `"LightGBM class imbalance baseball injury prediction"`
- `"focal loss XGBoost injury prediction"`
- `"SMOTE temporal cross-validation leakage"`
- `"pitcher workload spike year-over-year injury risk"`
Note the 1–2 most relevant findings. Do not spend more than 2 searches — implement and measure.

**Step 4 — Implement:**
- Edit the relevant notebook (usually NB05 for features, NB06 for models, NB07 for survival, NB09 for score construction).
- Keep changes minimal and targeted — one idea per round.
- Respect the methodology constraints (no leakage, no causal claims).

**Step 5 — Test:**
```bash
# TEST_MODE first
python run_notebooks.py --only NN --fail-fast

# Full run if TEST_MODE passes
python run_notebooks.py --only NN --fail-fast

# Verify
python scripts/verify_outputs.py --only NN
```

**Step 6 — Measure:**
Read the updated metrics CSVs. Compute the delta in PR-AUC (30d, temporal CV mean folds 1–4).

**Step 7 — Log in `docs/model_improvement_log.md`:**

```markdown
## [DATE] Round N — [Short Title]

### Hypothesis
[What improvement was expected and why, with research reference]

### Implementation
[What was changed and where]

### Results
| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| PR-AUC 30d (CV mean) | X.XXX | X.XXX | +X.XXX |
| AUC-ROC 30d (CV mean) | X.XXX | X.XXX | +X.XXX |
| Brier score | X.XXX | X.XXX | +X.XXX |

### Verdict
[Kept / Reverted — reason]
```

If the change made things worse, revert it (git checkout the notebook) and log the null result. Null results are informative.

**Step 8 — Update `.scratch/improvement_progress.json`:**

Structure:
```json
{
  "_meta": {
    "status": "in_progress",
    "baseline_pr_auc_30d": 0.137,
    "best_pr_auc_30d": 0.137,
    "consecutive_non_improvements": 0,
    "rounds_completed": 0
  },
  "round_001": {
    "status": "done",
    "category": "feature_engineering",
    "description": "Add YoY workload spike feature",
    "pr_auc_before": 0.137,
    "pr_auc_after": 0.152,
    "delta": 0.015,
    "notebooks_changed": ["05", "06"],
    "verdict": "kept"
  }
}
```

Set `_meta.status = "converged"` when:
- 3+ consecutive rounds produced delta < 0.005 and no obvious high-value ideas remain, OR
- 10 rounds have been completed.

**Step 9 — Commit:**
```bash
git add notebooks/ src/ docs/model_improvement_log.md .scratch/improvement_progress.json reports/
git commit -m "Phase 3A round N: [short description] (PR-AUC +X.XXX)"
```

---

## Phase 3B — Analysis Dashboard Protocol

### Goal

NB13 currently generates four separate Plotly HTML files. Build a unified Streamlit app at `dashboard/app.py` that combines all four components into one interactive analysis tool Nate can use for real analysis.

### Requirements

The dashboard must support these analysis workflows:
1. **Season leaderboard** — rank all pitchers by Injury Risk+ for a selected season; filter by archetype; click a pitcher to jump to their profile.
2. **Pitcher profile** — search by name; show IR+ trend across seasons, component breakdown (30d / 60d / 90d blend), and workload history.
3. **Multi-pitcher comparison** — overlay 2–5 pitchers' IR+ trends on one chart for direct comparison.
4. **Archetype analysis** — mean IR+ and component breakdown by pitcher archetype, by season.

### Tech Stack

Check if Streamlit is installed in the pitcher311 environment:
```bash
/opt/homebrew/opt/python@3.11/bin/python3.11 -c "import streamlit; print(streamlit.__version__)" 2>/dev/null
```

If not installed:
```bash
/opt/homebrew/opt/python@3.11/bin/pip3.11 install streamlit
```

Use Plotly for all charts (already installed). Do not introduce new chart libraries.

### File Layout

```
dashboard/
  app.py          ← main Streamlit entry point
  components/
    leaderboard.py
    pitcher_profile.py
    trend_comparison.py
    archetype_panel.py
  data_loader.py  ← cached data loading from data/processed/ and reports/tables/
```

### Implementation Protocol

**Step 1 — Scaffold:**
Create `dashboard/app.py` with sidebar navigation and stub views for all four panels. Run it to confirm it launches:
```bash
cd /path/to/project && /opt/homebrew/opt/python@3.11/bin/streamlit run dashboard/app.py
```

**Step 2 — Implement each component**, porting logic from NB13 cells. Use `@st.cache_data` for all data loads. Each component should be importable from its module.

**Step 3 — Integrate:** Wire the sidebar to switch between panels. Add a pitcher name search box (fuzzy match on `player_name` column) that pre-populates the pitcher profile panel.

**Step 4 — Validate:** Run the app, navigate through all four panels, confirm charts render with real data.

**Step 5 — Log in `.scratch/dashboard_progress.json`:**
```json
{
  "status": "done",
  "launch_command": "/opt/homebrew/opt/python@3.11/bin/streamlit run dashboard/app.py",
  "panels_complete": ["leaderboard", "pitcher_profile", "trend_comparison", "archetype_panel"]
}
```

**Step 6 — Commit:**
```bash
git add dashboard/ .scratch/dashboard_progress.json
git commit -m "Phase 3B: unified Streamlit analysis dashboard"
```

---

## Methodology Constraints (Always Enforce)

- No leakage: post-injury data must never appear in features
- No causal claims: all counterfactuals are model-based, not causal
- SMOTE and any resampling must be applied inside CV folds only, never on the full dataset before splitting
- Do not silently drop pitchers or seasons from the model training set
- If a change breaks `verify_outputs.py`, revert it before the next round

---

## Environment and Infrastructure

### Python Environments

- **Notebook cells:** `pitcher311` kernel → `/opt/homebrew/opt/python@3.11/bin/python3.11`
- **`.venv/bin/python` (3.13):** drives `run_project.sh` and `scripts/verify_outputs.py` only — cannot import data-science packages
- **Do not mix environments**

### Credit-Safe Execution Rule

Check before launching any notebook:
```bash
ps aux | grep -E "run_notebooks|jupyter|nbconvert" | grep -v grep
```

If a notebook must be rerun:
1. `TEST_MODE = True` first → `python run_notebooks.py --only NN --fail-fast`
2. Full run if TEST_MODE passes
3. `python scripts/verify_outputs.py --only NN`
4. Commit

Do not poll. After launching a notebook run, STOP IMMEDIATELY.

### Directory Standards

```text
data/raw/
data/processed/
data/processed/features/
models/
reports/figures/
reports/tables/
docs/
dashboard/
```
