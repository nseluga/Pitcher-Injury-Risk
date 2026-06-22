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

## Phase 3A — Survival Model Improvement Protocol

### Goal

The binary injury classifiers are near their ceiling. The focus now shifts to **survival models in NB07**, which answer a richer question: *when* is a pitcher likely to get injured, not just *whether*. A good survival model produces actionable workload management signals — e.g., "this pitcher's median time-to-next-injury is 14 days given current load" — that go beyond a binary flag.

Current survival model baselines (NB07):
- **Cox PH C-index:** 0.514 (near-random)
- **Weibull AFT C-index:** ~0.51
- **RSF C-index:** ~0.51
- **IBS:** not yet meaningfully better than null model

Primary metric: **C-index on the held-out test set** (higher = better, 0.5 = random, 1.0 = perfect). Secondary: Integrated Brier Score (IBS), meaningful hazard ratios, interpretable survival curves that tell a real story about pitcher risk.

### Step 0 — Brainstorm Before Every Round

**Before picking an idea, generate a fresh brainstorm list.** Read NB07 cell by cell to understand the current model setup, then write a list of 8–10 concrete approaches you haven't tried yet to `.scratch/survival_improvement_ideas.md`. Structure it as:

```markdown
# Survival Model Improvement Ideas — [DATE]

## Already tried (from improvement_progress.json)
- [list]

## New approaches to consider
1. [Approach] — [why it might help, evidence]
2. ...
```

Then pick the highest-expected-value idea from that list and proceed. This brainstorm step is mandatory every round — it prevents narrow fixation on one approach.

### Improvement Categories to Explore

Do not repeat an approach already logged as tried. Read `docs/model_improvement_log.md` first.

#### Survival-Specific Model Architectures
- **Penalized Cox (LASSO/Elastic Net):** replace the current Cox PH with a regularized version (`CoxnetSurvivalAnalysis` from scikit-survival). Better feature selection, more stable coefficients when features >> events.
- **Stratified Cox PH:** stratify the baseline hazard by pitcher role (starter vs reliever) or age group to relax the proportional hazards assumption. Test with Schoenfeld residuals first.
- **Log-normal and log-logistic AFT:** add these distributions alongside Weibull AFT — different assumptions about hazard shape, may fit injury timing better.
- **Gradient boosted survival (GBS):** try `GradientBoostingSurvivalAnalysis` from scikit-survival. Nonlinear, handles interactions automatically, often outperforms Cox on tabular data.
- **Frailty / shared frailty model:** add pitcher-level random effects to capture unobserved heterogeneity (some pitchers are just injury-prone regardless of workload). Requires `lifelines` GammaMixtureFrailtyFitter.
- **DeepSurv / neural survival:** small MLP trained with negative log-likelihood of Cox PH using `pycox` or `auton-survival`. Can capture feature interactions Cox misses.

#### Feature Engineering for Time-to-Event
- **Time since last injury:** days elapsed since the pitcher's most recent IL stint. Strong prior in clinical survival literature.
- **Cumulative injury count:** total prior IL stints — "frailty proxy" for pitchers with recurrent injuries.
- **Season phase:** early (games 1–30), mid, late, postseason — hazard likely varies across the season.
- **Start-count fatigue:** starts_this_season × avg_pitch_count_per_start — accumulated within-season stress.
- **Arm-injury-only survival target:** re-define the event as only arm/shoulder/elbow IL stints (filter `data/processed/injuries_clean.parquet` on injury type). Removes noise from oblique strains, leg injuries, etc.

#### Calibration and Interpretability
- **Survival curve visualization:** plot Kaplan-Meier curves stratified by workload quartile (pitches_90d) and compare to model-predicted curves. If they align, the model is capturing real signal.
- **Hazard ratio table:** extract and report the top-10 most significant Cox PH coefficients with 95% CIs. A table of meaningful HRs (e.g., ACWR HR = 1.8, p < 0.05) is publishable signal even if C-index is modest.
- **Calibration at fixed horizons:** compute Brier score at t=30, t=60, t=90 days to see where the model is most/least calibrated.
- **IBS improvement via recalibration:** apply isotonic regression to survival probability outputs.

### Improvement Protocol — One Round Per Session

**Step 1 — Orient:**
```bash
cat .scratch/improvement_progress.json
cat docs/model_improvement_log.md | tail -120
```

**Step 2 — Brainstorm (required):**
- Read NB07 cell by cell to understand the current survival model implementation.
- Write `.scratch/survival_improvement_ideas.md` with 8–10 ideas (see Step 0 format above).
- Pick the highest-expected-value idea not yet attempted.
- If 3+ consecutive rounds showed C-index delta < 0.005 AND no obvious high-value ideas remain, set `_meta.status = "converged"` in `improvement_progress.json`.

**Step 3 — Research (1–2 WebSearches):**
Ground the chosen idea in evidence before implementing. Example searches:
- `"penalized Cox survival analysis baseball injury prediction"`
- `"gradient boosted survival trees C-index improvement"`
- `"time-varying covariates pitcher workload survival model"`
- `"frailty model recurrent sports injury"`
Note the 1–2 most relevant findings. Do not spend more than 2 searches — implement and measure.

**Step 4 — Implement:**
- Primary target: `notebooks/07_survival_models.ipynb` and `src/models/survival_models.py`.
- May also touch NB05 for new survival-relevant features.
- Keep changes minimal and targeted — one idea per round.
- Respect the methodology constraints (no leakage, no causal claims).

**Step 5 — Test:**
```bash
# TEST_MODE first
python run_notebooks.py --only 07 --fail-fast

# Full run if TEST_MODE passes
python run_notebooks.py --only 07 --fail-fast

# Verify
python scripts/verify_outputs.py --only 07
```

**Step 6 — Measure:**
Report C-index (test set) and IBS. Also note whether survival curves or hazard ratios tell a meaningful story — interpretability is a win even when C-index gain is small.

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
| C-index (test) | X.XXX | X.XXX | +X.XXX |
| IBS | X.XXX | X.XXX | +X.XXX |
| Notable HRs / curves | — | [description] | — |

### Verdict
[Kept / Reverted — reason. Note any interpretable signal even if C-index was flat.]
```

If the change made things worse, revert it (`git checkout notebooks/07_survival_models.ipynb src/models/survival_models.py`) and log the null result. Null results are informative.

**Step 8 — Update `.scratch/improvement_progress.json`:**

Structure:
```json
{
  "_meta": {
    "status": "in_progress",
    "baseline_c_index": 0.514,
    "best_c_index": 0.514,
    "consecutive_non_improvements": 0,
    "rounds_completed": 0
  },
  "round_001": {
    "status": "done",
    "category": "model_architecture",
    "description": "Penalized Cox with LASSO feature selection",
    "c_index_before": 0.514,
    "c_index_after": 0.531,
    "delta": 0.017,
    "notebooks_changed": ["07"],
    "verdict": "kept — C-index +0.017, top HRs interpretable"
  }
}
```

Set `_meta.status = "converged"` when:
- 3+ consecutive rounds produced C-index delta < 0.005 and no obvious high-value ideas remain, OR
- 10 rounds have been completed.

**Step 9 — Commit:**
```bash
git add notebooks/07_survival_models.ipynb src/models/survival_models.py docs/model_improvement_log.md .scratch/improvement_progress.json .scratch/survival_improvement_ideas.md
git commit -m "Phase 3A round N: [short description] (C-index +X.XXX)"
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
