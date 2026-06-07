"""
Generate notebooks/07_survival_models.ipynb programmatically.
Run from project root: python scripts/generate_notebook_07.py
"""

import json
from pathlib import Path


def md(source: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source}


def code(source: str, cell_id: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "id": cell_id,
            "metadata": {}, "outputs": [], "source": source}


# ---------------------------------------------------------------------------
# CELL SOURCES
# ---------------------------------------------------------------------------

C0 = """\
# 07 — Survival Models

## Purpose
Reframe injury prediction as a **time-to-event** problem. Binary classifiers
(notebook 06) answer "will this pitcher be injured in the next 30 days?" —
survival models answer the richer question "*when* is this pitcher likely to
be injured, and how does that risk evolve over time?"

This framing is statistically correct for our data because it naturally
handles **censoring**: most pitcher-game observations never see a
follow-up injury within the data window, but that doesn't mean they're risk-
free — it means we simply ran out of observation time. Binary labels throw
that information away; survival models use it directly.

## Models implemented
1. **Cox Proportional Hazards** — interpretable, widely-used semi-parametric model
2. **Weibull AFT** — parametric accelerated-failure-time model
3. **Random Survival Forest** — non-parametric, captures non-linear interactions

## Evaluation
- **C-index** — concordance between predicted risk ranking and actual event order
- **Integrated Brier Score (IBS)** — calibration of survival probabilities over time
- Survival curves by pitcher archetype (starter vs. reliever)

## Key concepts
| Concept | Definition here |
|---|---|
| Event | IL placement |
| Duration (T) | `days_to_next_injury`, capped at a 90-day horizon |
| Censoring (E=0) | No injury observed within the horizon (right-censored) |\
"""

C1 = """\
import sys, json, warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 60)

PROJECT_ROOT = str(Path('.').resolve())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.survival_models import (
    prepare_survival_dataset,
    train_cox_ph,
    train_aft_model,
    train_random_survival_forest,
    predict_survival_function,
    compute_concordance_index,
    DEFAULT_HORIZON_DAYS,
)
from src.models.evaluation import evaluate_survival_model

MODELS_DIR  = Path('models')
TABLES_DIR  = Path('reports/tables')
FIGURES_DIR = Path('reports/figures')
for d in (MODELS_DIR, TABLES_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

print('Modules loaded. Survival horizon:', DEFAULT_HORIZON_DAYS, 'days')\
"""

C2_MD = """\
## 1. Load Feature Matrix and Build the Survival Dataset

`prepare_survival_dataset` converts each pitcher-game row into a
`(duration, event)` pair:

- **event = 1**: the pitcher was placed on the IL within `DEFAULT_HORIZON_DAYS`
  of this appearance, with `duration = days_to_next_injury`
- **event = 0** (censored): no injury was observed in that window;
  `duration` is capped at the horizon

This mirrors how survival data is structured in clinical trials — most
"patients" (pitcher-appearances) are censored, and that's expected.\
"""

C2 = """\
fm = pd.read_parquet('data/processed/feature_matrix.parquet')
fm['game_date'] = pd.to_datetime(fm['game_date'])
seasons = sorted(fm['season'].unique().tolist())

X_train, X_test, T_train, T_test, E_train, E_test = prepare_survival_dataset(fm)

print(f'Seasons: {seasons}')
print(f'Train: {len(X_train):,} rows | events = {int(E_train.sum())} '
      f'({E_train.mean():.1%}) | censored = {int((1 - E_train).sum())}')
print(f'Test : {len(X_test):,} rows | events = {int(E_test.sum())} '
      f'({E_test.mean():.1%}) | censored = {int((1 - E_test).sum())}')
print()
print('Duration summary (train):')
display(T_train.describe().to_frame('days_to_event_or_censoring'))\
"""

C3_MD = """\
## 2. Train Survival Models

All three models are fit on the same `(X, T, E)` representation. Lifelines'
Cox/AFT models additionally require:
- median imputation (rolling-window features are null early in a pitcher's
  observed history)
- dropping zero-variance columns (a constant column makes the Cox design
  matrix singular)

Both are handled inside `survival_models.py` and the resulting imputer/column
list are stashed on the fitted model so prediction can replay the same steps.\
"""

C3 = """\
%%time
survival_models = {}

print('Training Cox Proportional Hazards...')
survival_models['cox_ph'] = train_cox_ph(X_train, T_train, E_train)

print('Training Weibull AFT...')
survival_models['weibull_aft'] = train_aft_model(X_train, T_train, E_train, distribution='weibull')

print('Training Random Survival Forest...')
survival_models['random_survival_forest'] = train_random_survival_forest(X_train, T_train, E_train)

print('\\nModels trained:', list(survival_models.keys()))\
"""

C4_MD = """\
## 3. Evaluate — C-index and Integrated Brier Score

- **C-index** ≈ 0.5 → no better than random ranking; → 1.0 → perfect ranking
  of who gets injured first.
- **IBS** (mean Brier score across the 30/60/90-day horizons) measures how
  well the predicted survival probabilities match observed outcomes — lower
  is better, 0 is perfect.\
"""

C4 = """\
eval_rows = []
for name, model in survival_models.items():
    try:
        metrics = evaluate_survival_model(model, X_test, T_test, E_test, time_points=[30, 60, 90])
    except Exception as exc:
        print(f'{name}: evaluation failed — {exc}')
        continue
    metrics['model'] = name
    eval_rows.append(metrics)

survival_results_df = pd.DataFrame(eval_rows).set_index('model')[
    ['c_index', 'ibs', 'brier_30', 'brier_60', 'brier_90']
].sort_values('c_index', ascending=False)

display(survival_results_df.style.format('{:.3f}'))\
"""

C5_MD = """\
## 4. Survival Curves by Archetype

How does projected survival (i.e. "probability of remaining injury-free")
diverge between starters and relievers over the next 90 days? We use the
same pitch-count heuristic as elsewhere in the pipeline (`pitch_count >= 50`
⇒ starter) since `encode_pitcher_archetype` is not yet implemented.\
"""

C5 = """\
best_model_name = survival_results_df.index[0]
best_survival_model = survival_models[best_model_name]
print(f'Using {best_model_name} for survival curve visualization (highest C-index)')

time_grid = list(range(0, DEFAULT_HORIZON_DAYS + 1, 5))
test_meta = fm.loc[X_test.index, ['pitcher', 'season', 'pitch_count']].copy()
season_avg_pitches = fm.groupby(['pitcher', 'season'])['pitch_count'].transform('mean')
test_meta['archetype'] = np.where(season_avg_pitches.loc[X_test.index] >= 50, 'starter', 'reliever')

surv_grid = predict_survival_function(best_survival_model, X_test, time_points=time_grid)

fig, ax = plt.subplots(figsize=(8, 6))
for archetype, color in [('starter', '#C44E52'), ('reliever', '#4C72B0')]:
    mask = test_meta['archetype'] == archetype
    if mask.sum() == 0:
        continue
    mean_curve = surv_grid.loc[mask.values].mean(axis=0)
    ax.plot(time_grid, mean_curve.values, marker='o', markersize=3, label=f'{archetype} (n={mask.sum()})', color=color)

ax.set_xlabel('Days since appearance')
ax.set_ylabel('P(remains injury-free)')
ax.set_title(f'Mean predicted survival curve by archetype — {best_model_name}')
ax.set_ylim(0, 1.02)
ax.legend()
fig.tight_layout()
fig_path = FIGURES_DIR / 'fig_22_survival_curves_by_archetype.png'
fig.savefig(fig_path, dpi=120, bbox_inches='tight')
plt.show()
print(f'Saved {fig_path}')\
"""

C6_MD = """\
## 5. Cox Model Coefficients (Interpretability Preview)

The Cox model's hazard ratios are directly interpretable: `exp(coef) > 1`
means the feature *increases* the instantaneous injury hazard, `< 1` means it
*decreases* it. This is a useful sanity check before the deeper SHAP-based
analysis in notebook 10.\
"""

C6 = """\
cox_model = survival_models['cox_ph']
coef_df = (
    cox_model.summary[['coef', 'exp(coef)', 'p']]
    .sort_values('exp(coef)', ascending=False)
    .reset_index()
    .rename(columns={'covariate': 'feature'})
)

print('Top 10 features by hazard ratio (highest risk association):')
display(coef_df.head(10))
print('\\nBottom 10 features by hazard ratio (most protective association):')
display(coef_df.tail(10))\
"""

C7_MD = """\
## 6. Save Models and Results\
"""

C7 = """\
import joblib

for name, model in survival_models.items():
    out_path = MODELS_DIR / f'survival_{name}.pkl'
    joblib.dump(model, out_path)
    print(f'Saved {out_path}')

survival_results_df.reset_index().to_csv(TABLES_DIR / 'survival_model_results.csv', index=False)
print(f'\\nSaved {TABLES_DIR / "survival_model_results.csv"}')

coef_df.to_csv(TABLES_DIR / 'survival_cox_coefficients.csv', index=False)
print(f'Saved {TABLES_DIR / "survival_cox_coefficients.csv"}')\
"""

C8_MD = """\
## 7. Summary

* **Best survival model:** ranked by C-index in section 3 — this is the model
  that will supply the `hazard_rate` component of Injury Risk+ (notebook 09).
* **Censoring matters:** roughly {censor_note} of pitcher-appearances are
  censored (no injury observed within the {horizon}-day horizon) — these are
  *not* "negative" examples, they're "we don't know yet" examples, and the
  survival framing is what lets us use them honestly.
* **Archetype divergence:** the survival-curve comparison in section 4 shows
  whether starters and relievers carry meaningfully different baseline risk
  trajectories — directly informing the archetype-normalization step in
  Injury Risk+ scoring.
* **Next step:** notebook 08 combines binary, regression, and survival-style
  signals into a single multi-task model.\
"""

C8 = """\
provenance = {
    'notebook': '07_survival_models',
    'run_at': datetime.now(timezone.utc).isoformat(),
    'seasons_used': seasons,
    'horizon_days': DEFAULT_HORIZON_DAYS,
    'n_train': int(len(X_train)),
    'n_test': int(len(X_test)),
    'censoring_rate_train': float(1 - E_train.mean()),
    'best_model': best_model_name,
    'test_metrics': survival_results_df.loc[best_model_name].to_dict(),
}
print(json.dumps(provenance, indent=2, default=str))

prov_path = TABLES_DIR / 'survival_model_provenance.json'
prov_path.write_text(json.dumps(provenance, indent=2, default=str))
print(f'\\nSaved {prov_path}')\
"""

# ---------------------------------------------------------------------------
# ASSEMBLE
# ---------------------------------------------------------------------------

cells = [
    md(C0,    "c00-header"),
    code(C1,  "c01-setup"),
    md(C2_MD, "c02-data-header"),
    code(C2,  "c02-data"),
    md(C3_MD, "c03-train-header"),
    code(C3,  "c03-train"),
    md(C4_MD, "c04-eval-header"),
    code(C4,  "c04-eval"),
    md(C5_MD, "c05-curves-header"),
    code(C5,  "c05-curves"),
    md(C6_MD, "c06-coef-header"),
    code(C6,  "c06-coef"),
    md(C7_MD, "c07-save-header"),
    code(C7,  "c07-save"),
    md(C8_MD, "c08-summary-header"),
    code(C8,  "c08-provenance"),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "pitcher-injury-risk",
            "language": "python",
            "name": "pitcher-injury-risk",
        },
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path("notebooks/07_survival_models.ipynb")
out.parent.mkdir(exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Written: {out}  ({len(cells)} cells)")
