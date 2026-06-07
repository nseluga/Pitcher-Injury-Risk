"""
Generate notebooks/08_multitask_models.ipynb programmatically.
Run from project root: python scripts/generate_notebook_08.py
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
# 08 — Multi-Task Models

## Purpose
Notebooks 06 and 07 each predict a *single* outcome in isolation. But injury
risk is really a bundle of related questions:

- **Will** this pitcher be injured (30 / 60 / 90 days)?
- **When** will it happen (`days_to_next_injury`)?
- **How severe** will it be (`next_injury_days_lost`)?
- **What kind** of injury (`next_injury_type`)?

These tasks share an underlying cause — accumulated workload and mechanical
strain — so a model that learns them jointly can borrow statistical strength
from the more common tasks (binary injury within 30d) to improve the rarer,
harder ones (severity, type). This notebook compares two multi-task
architectures:

1. **Chained model** — predicts injury probability first, then feeds that
   prediction as an input feature to the downstream severity/type/timing heads
   (reflects the causal ordering: *whether* precedes *how bad*)
2. **Shared-representation model** — a single fitted preprocessing trunk
   feeds independent per-task heads (the practical analogue of hard parameter
   sharing for heterogeneous task types with tree ensembles)

## Why this matters for Injury Risk+
The composite score (notebook 09) needs `injury_prob_30d`,
`expected_days_lost`, and a hazard signal *simultaneously, for the same
pitcher* — multi-task models are a natural way to produce a coherent bundle
of these predictions from one fitting pass.\
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

from src.models.baseline_models import _infer_feature_cols, save_model
from src.models.multitask_models import (
    prepare_multitask_dataset,
    train_chained_multitask_model,
    train_shared_representation_model,
    predict_all_tasks,
    compute_multitask_metrics,
    ALL_TASKS,
    CLASSIFICATION_TASKS,
    REGRESSION_TASKS,
)

MODELS_DIR  = Path('models')
TABLES_DIR  = Path('reports/tables')
FIGURES_DIR = Path('reports/figures')
for d in (MODELS_DIR, TABLES_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

print('Modules loaded. Tasks:', ALL_TASKS)\
"""

C2_MD = """\
## 1. Load Feature Matrix and Build the Multi-Task Dataset

`prepare_multitask_dataset` builds a single shared `(X_train, X_test)` split
(temporal — most recent seasons held out) and a dictionary of target arrays
per task. Note that the regression and multiclass targets
(`days_to_next_injury`, `next_injury_days_lost`, `next_injury_type`) are only
*observed* for pitchers who were actually later injured — everyone else is
`NaN`. The training routines filter to the observed subset internally
(`_regression_subset`).\
"""

C2 = """\
fm = pd.read_parquet('data/processed/feature_matrix.parquet')
fm['game_date'] = pd.to_datetime(fm['game_date'])
seasons = sorted(fm['season'].unique().tolist())

feature_cols = _infer_feature_cols(fm)
X_train, X_test, y_train_dict, y_test_dict = prepare_multitask_dataset(fm, feature_cols=feature_cols)

print(f'Seasons: {seasons}')
print(f'Train: {len(X_train):,} rows | Test: {len(X_test):,} rows | Features: {len(feature_cols)}')
print()
print('Task observation rates (train):')
for task in ALL_TASKS:
    n_obs = int(y_train_dict[task].notna().sum())
    print(f'  {task:24s}  observed = {n_obs:5,d} / {len(y_train_dict[task]):,}  '
          f'({n_obs / len(y_train_dict[task]):.1%})')\
"""

C3_MD = """\
## 2. Train Both Architectures\
"""

C3 = """\
%%time
print('Training chained multi-task model...')
chained_model = train_chained_multitask_model(X_train, y_train_dict)

print('Training shared-representation model...')
shared_model = train_shared_representation_model(X_train, y_train_dict, shared_model_type='gradient_boosting')

print('\\nBoth architectures trained.')\
"""

C4_MD = """\
## 3. Evaluate Both Architectures on the Held-Out Test Set

For each architecture we run inference for every task in one pass
(`predict_all_tasks`) and score with `compute_multitask_metrics`:
classification tasks get AUC-ROC / PR-AUC / Brier, regression tasks get
MAE / RMSE / R², and the multiclass task gets accuracy / macro-F1.\
"""

C4 = """\
chained_preds = predict_all_tasks(chained_model, X_test)
shared_preds  = predict_all_tasks(shared_model, X_test)

chained_metrics = compute_multitask_metrics(y_test_dict, chained_preds)
shared_metrics  = compute_multitask_metrics(y_test_dict, shared_preds)

print('Chained multi-task model — test metrics:')
display(chained_metrics)
print()
print('Shared-representation model — test metrics:')
display(shared_metrics)\
"""

C5_MD = """\
## 4. Head-to-Head Comparison

Side-by-side AUC-ROC (classification tasks) and MAE (regression tasks) for
both architectures — which one wins, and on which kinds of tasks?\
"""

C5 = """\
compare_rows = []
for task in CLASSIFICATION_TASKS:
    if task in chained_metrics.index and task in shared_metrics.index:
        compare_rows.append({
            'task': task, 'metric': 'auc_roc',
            'chained': chained_metrics.loc[task, 'auc_roc'],
            'shared': shared_metrics.loc[task, 'auc_roc'],
        })
for task in REGRESSION_TASKS:
    if task in chained_metrics.index and task in shared_metrics.index:
        compare_rows.append({
            'task': task, 'metric': 'mae',
            'chained': chained_metrics.loc[task, 'mae'],
            'shared': shared_metrics.loc[task, 'mae'],
        })

compare_df = pd.DataFrame(compare_rows).set_index(['task', 'metric'])
compare_df['winner'] = np.where(
    compare_df.index.get_level_values('metric') == 'mae',
    np.where(compare_df['chained'] < compare_df['shared'], 'chained', 'shared'),
    np.where(compare_df['chained'] > compare_df['shared'], 'chained', 'shared'),
)
display(compare_df.style.format({'chained': '{:.3f}', 'shared': '{:.3f}'}))

n_chained_wins = int((compare_df['winner'] == 'chained').sum())
n_shared_wins = int((compare_df['winner'] == 'shared').sum())
best_architecture = 'chained' if n_chained_wins >= n_shared_wins else 'shared'
print(f'\\nChained wins {n_chained_wins}/{len(compare_df)} comparisons; '
      f'Shared wins {n_shared_wins}/{len(compare_df)}.')
print(f'Selected architecture for downstream use: {best_architecture}')\
"""

C6_MD = """\
## 5. Visualize Task Performance

A compact view of how well each architecture handles each task family —
classification (AUC-ROC, higher is better) vs. regression (MAE, lower is
better, shown as inverted bars for visual consistency).\
"""

C6 = """\
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

clf_compare = compare_df.xs('auc_roc', level='metric')
ax = axes[0]
x = np.arange(len(clf_compare))
width = 0.35
ax.bar(x - width / 2, clf_compare['chained'], width, label='chained', color='#4C72B0')
ax.bar(x + width / 2, clf_compare['shared'], width, label='shared', color='#C44E52')
ax.set_xticks(x)
ax.set_xticklabels(clf_compare.index, rotation=20, ha='right')
ax.set_ylabel('AUC-ROC')
ax.set_title('Classification tasks (higher is better)')
ax.legend()

reg_compare = compare_df.xs('mae', level='metric')
ax = axes[1]
x = np.arange(len(reg_compare))
ax.bar(x - width / 2, reg_compare['chained'], width, label='chained', color='#4C72B0')
ax.bar(x + width / 2, reg_compare['shared'], width, label='shared', color='#C44E52')
ax.set_xticks(x)
ax.set_xticklabels(reg_compare.index, rotation=20, ha='right')
ax.set_ylabel('MAE (days)')
ax.set_title('Regression tasks (lower is better)')
ax.legend()

fig.tight_layout()
fig_path = FIGURES_DIR / 'fig_23_multitask_comparison.png'
fig.savefig(fig_path, dpi=120, bbox_inches='tight')
plt.show()
print(f'Saved {fig_path}')\
"""

C7_MD = """\
## 6. Save Models and Results

We persist both architectures (notebook 09 can choose either), but flag the
winner from the head-to-head comparison as the default for Injury Risk+
inference.\
"""

C7 = """\
import joblib

chained_path = MODELS_DIR / 'multitask_chained.pkl'
shared_path  = MODELS_DIR / 'multitask_shared.pkl'
joblib.dump(chained_model, chained_path)
joblib.dump(shared_model, shared_path)
print(f'Saved {chained_path}')
print(f'Saved {shared_path}')

chained_metrics.assign(architecture='chained').reset_index().to_csv(
    TABLES_DIR / 'multitask_chained_metrics.csv', index=False)
shared_metrics.assign(architecture='shared').reset_index().to_csv(
    TABLES_DIR / 'multitask_shared_metrics.csv', index=False)
compare_df.reset_index().to_csv(TABLES_DIR / 'multitask_comparison.csv', index=False)
print(f'Saved {TABLES_DIR / "multitask_chained_metrics.csv"}')
print(f'Saved {TABLES_DIR / "multitask_shared_metrics.csv"}')
print(f'Saved {TABLES_DIR / "multitask_comparison.csv"}')\
"""

C8_MD = """\
## 7. Summary

* **Architecture comparison:** see section 4 — whichever model wins more
  task-level comparisons becomes the default multi-task model fed into
  Injury Risk+ (notebook 09), though both are saved to `models/` for
  flexibility.
* **Task difficulty gradient:** classification tasks (binary injury within N
  days) are the best-observed and easiest; regression/multiclass tasks
  (severity, timing, type) are trained on a much smaller "injured-only"
  subset and are correspondingly noisier — exactly the rare-outcome problem
  multi-task learning is meant to help with.
* **Next step:** notebook 09 combines `injury_prob_30d`,
  `expected_days_lost`, and a survival-derived `hazard_rate` into the
  composite Injury Risk+ score, calibrates it, and produces the final
  pitcher rankings.\
"""

C8 = """\
provenance = {
    'notebook': '08_multitask_models',
    'run_at': datetime.now(timezone.utc).isoformat(),
    'seasons_used': seasons,
    'n_train': int(len(X_train)),
    'n_test': int(len(X_test)),
    'n_features': len(feature_cols),
    'best_architecture': best_architecture,
    'n_chained_wins': n_chained_wins,
    'n_shared_wins': n_shared_wins,
    'chained_metrics': chained_metrics.to_dict(orient='index'),
    'shared_metrics': shared_metrics.to_dict(orient='index'),
}
print(json.dumps(provenance, indent=2, default=str))

prov_path = TABLES_DIR / 'multitask_model_provenance.json'
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
    md(C5_MD, "c05-compare-header"),
    code(C5,  "c05-compare"),
    md(C6_MD, "c06-viz-header"),
    code(C6,  "c06-viz"),
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

out = Path("notebooks/08_multitask_models.ipynb")
out.parent.mkdir(exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Written: {out}  ({len(cells)} cells)")
