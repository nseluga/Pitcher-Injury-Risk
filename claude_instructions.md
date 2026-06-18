# Claude Instructions: Pitcher Injury Risk+ — Interpret the Findings

You are working on the `Pitcher-Injury-Risk` project.

## Current Phase: Phase 3 — Interpretation

**Phase 1 (Implementation): COMPLETE.**  
All notebooks 01–13 implemented, executed on the full 2015–2024 dataset, and verified by `python scripts/verify_outputs.py`.

**Phase 2 (Baseball Research Critique & Improvement): COMPLETE.**  
All modeling notebooks 05–12 critiqued against published literature, improvements applied and re-verified. Full record in `docs/model_critique_log.md`.

**Phase 3 (Interpretation): IN PROGRESS.**  
The pipeline is stable and research-grounded. The task now is to interpret what the models actually found — synthesizing findings across notebooks into clear, defensible conclusions about pitcher injury risk.

---

## How to Orient at Session Start

```bash
cat docs/model_critique_log.md          # full critique history
cat reports/tables/shap_global_importance.csv
cat reports/tables/baseball_specific_insights_summary.csv
cat reports/tables/injury_risk_plus_leaderboard.csv
cat .scratch/critique_progress.json     # all 05–12 done
```

Do not re-run `verify_outputs.py` unless you have changed a notebook or source module. The pipeline is passing and stable.

---

## Phase 3 Interpretation Protocol

### Goal

Write a rigorous narrative interpretation of what the model learned. This is not a methods section — it is a findings section. The output is a document (or set of documents) that could be read by a baseball analyst or team medical staff and answer the question: *"What does this model tell us about pitcher injury risk?"*

### Interpretation Targets

Work through these in order. Each produces one written section. Commit after each.

#### 1. Top Predictors — What drives the Injury Risk+ score?

- Read `reports/tables/shap_global_importance.csv` and `reports/figures/shap_global_importance.png`.
- Read the SHAP beeswarm and PDP plots in `reports/figures/`.
- For each of the top 10 features by mean |SHAP|:
  - State the direction: does high values increase or decrease risk?
  - State the magnitude: how large is the effect relative to other features?
  - Cross-reference the critique log: does this align with the published baseball research findings, or is it surprising?
  - If surprising, call it out explicitly — surprises may indicate leakage, data artifacts, or genuinely novel findings.
- Summarize in plain language: "The three strongest independent predictors are X, Y, and Z."

#### 2. Injury Risk+ Distribution — What does the score landscape look like?

- Read `reports/tables/risk_score_component_summary.csv` and `data/processed/injury_risk_plus_scores.parquet` (sample).
- Read `reports/figures/injury_risk_plus_distribution.png`.
- Answer:
  - What is the actual score range? (Not just "centered at 100" — what are the extremes, and are they plausible?)
  - How skewed is the distribution? Are there score outliers that suggest data issues or genuinely extreme cases?
  - Do the component weights in `reports/tables/injury_risk_plus_blend_weights.csv` produce a sensible score relative to what the individual models say?

#### 3. Historical Leaderboards — Which pitchers and seasons stand out?

- Read the per-year top-25 tables in `reports/tables/injury_risk_plus_top25_*.csv`.
- Answer:
  - Do the highest-risk pitcher-seasons correspond to real-world injury events? (Use your knowledge of baseball history — cross-reference with the injury database if needed.)
  - Are there any notable false positives (high Injury Risk+ but no subsequent injury) or false negatives (actual TJ surgery cases not flagged)?
  - Are there historical patterns across seasons (e.g., post-COVID 2020 cluster, 2023 high-velocity pitchers)?

#### 4. Archetype Differences — Who is most at risk by pitcher type?

- Read `reports/tables/baseball_specific_insights_summary.csv`.
- Read `reports/figures/fig_19_archetype_injury_rates.png` and `fig_22_survival_curves_by_archetype.png`.
- Answer:
  - Which pitcher archetype has the highest median Injury Risk+ and why?
  - Does the power pitcher > finesse pitcher risk hierarchy hold, as predicted by the biomechanical literature?
  - How large are the between-archetype differences? Are they clinically meaningful?

#### 5. Simulation Insights — What interventions actually move the needle?

- Read `reports/tables/simulation_results.csv`.
- Read `reports/figures/pitch_count_optimization.png` and `rest_schedule_optimization.png`.
- Answer:
  - What reduction in Injury Risk+ does a 10% pitch count reduction produce?
  - What reduction does adding one extra rest day produce?
  - Are the effects approximately additive, or do they show diminishing returns?
  - Does slider reduction produce a meaningful effect? For whom?
  - Important: frame these as *model-based counterfactuals*, not causal recommendations.

#### 6. Model Limitations and Caveats

Synthesize the known limitations from the critique log and notebook markdown into a single honest section:
- Label construction uncertainty (soft labels from IL stints, not confirmed diagnoses)
- Data gaps (pre-2015 history missing, minor league stints missing)
- Calibration status: are the raw probabilities well-calibrated? (See `reports/figures/fig_20_baseline_calibration.png` and `fig_20b_tuned_calibration.png`)
- What the model cannot predict (acute traumatic injuries, fielding injuries, freak incidents)
- Time-lag issues: the model predicts injury *within 30/60/90 days* — it is not a real-time pitch-by-pitch alert system

### Output Format

Write findings as a new document: `docs/findings_summary.md`.

Structure:

```markdown
# Pitcher Injury Risk+ — Findings Summary

## 1. Top Predictors
...

## 2. Score Distribution
...

## 3. Historical Leaderboards
...

## 4. Archetype Risk Differences
...

## 5. Simulation Insights
...

## 6. Limitations and Caveats
...
```

Each section should be 2–5 paragraphs. Cite specific numbers from the tables and figures. Be honest about uncertainty.

### Quality Bar

A finding is worth stating if it:
- Is grounded in a specific artifact (table, figure, or model metric) — not just intuition
- Has a direction and approximate magnitude
- Is framed with appropriate caution (distinguish "the model says X" from "X is causal")

Do not editorialize. Do not overstate model performance. Do not claim causal conclusions.

---

## Commit Protocol

After completing each interpretation section:

```bash
git add docs/findings_summary.md
git commit -m "Phase 3: [section name] interpretation complete"
```

---

# Environment and Infrastructure

*(Keep for reference — relevant only if re-running notebooks or fixing bugs.)*

## Python Environment

- **Notebook cells:** `pitcher311` kernel → `/opt/homebrew/opt/python@3.11/bin/python3.11`
- **`.venv/bin/python` (3.13):** drives `run_project.sh` and `scripts/verify_outputs.py` only — cannot import data-science packages
- Do not mix environments

## Credit-Safe Execution Rule

Do not spend Claude turns waiting for long notebook runs. Check before launching:

```bash
ps aux | grep -E "run_notebooks|jupyter|nbconvert" | grep -v grep
```

If a notebook must be rerun (e.g., to validate a newly discovered bug fix):

1. Run in `TEST_MODE = True` first
2. Then `python run_notebooks.py --only NN --fail-fast`
3. Verify: `python scripts/verify_outputs.py --only NN`
4. Commit

## Directory Standards

```text
data/raw/
data/processed/
data/processed/features/
models/
reports/figures/
reports/tables/
docs/
```

## Methodology Constraints (Always Enforce)

Do NOT:
- allow leakage (post-injury data in features)
- make causal claims (all simulation results are model-based counterfactuals)
- optimize for accuracy alone (PR-AUC and calibration matter more)
- silently drop data
- fabricate labels

If a limitation is discovered: document it in `docs/findings_summary.md` under Section 6, not in a workaround.
