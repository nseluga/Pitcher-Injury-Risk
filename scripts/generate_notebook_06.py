"""
Generate notebooks/06_baseline_models.ipynb programmatically.
Run from project root: python scripts/generate_notebook_06.py
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
# 06 — Baseline Models

## Purpose
Establish performance benchmarks for injury prediction using interpretable,
well-understood models before moving to survival (notebook 07) and multi-task
(notebook 08) approaches.

This notebook:
1. Loads the full-data feature matrix from notebook 05
2. Builds a temporal train/test split (train on earlier seasons, test on the
   most recent ones — avoids look-ahead leakage)
3. Trains four baselines for the 30-day injury classification task:
   - Naive historical-rate-by-archetype/age-band
   - Logistic Regression (L2, balanced class weights)
   - Random Forest (cross-validated depth/leaf size)
   - XGBoost (gradient boosting)
4. Evaluates all models with AUC-ROC, PR-AUC, Brier score, and calibration
5. Runs walk-forward temporal cross-validation for the strongest model
6. Saves fitted models to `models/` and a comparison table to `reports/tables/`

## Target
Primary target: `injured_next_30d` (binary). The same pipeline generalizes to
the 60- and 90-day horizons, which we also report for comparison.\
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

from src.models.baseline_models import (
    prepare_classification_dataset,
    train_logistic_regression,
    train_random_forest,
    train_gradient_boosting,
    train_naive_baseline,
    save_model,
)
from src.models.evaluation import (
    evaluate_classifier,
    plot_calibration_curve,
    compute_feature_importance,
    temporal_cross_validate,
)

MODELS_DIR  = Path('models')
TABLES_DIR  = Path('reports/tables')
FIGURES_DIR = Path('reports/figures')
for d in (MODELS_DIR, TABLES_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

print('Modules loaded. Output dirs ready.')\
"""

C2_MD = """\
## 1. Load Feature Matrix

We use the full feature matrix (`feature_matrix.parquet`) rather than the
single-season train split — with multiple seasons on disk we can build a
genuine temporal holdout instead of relying on a single-season fallback.\
"""

C2 = """\
fm_path = Path('data/processed/feature_matrix.parquet')
if not fm_path.exists():
    raise FileNotFoundError('Run notebook 05 first to build the feature matrix.')

fm = pd.read_parquet(fm_path)
fm['game_date'] = pd.to_datetime(fm['game_date'])

seasons = sorted(fm['season'].unique().tolist())
print(f'Feature matrix: {fm.shape[0]:,} rows x {fm.shape[1]} cols')
print(f'Seasons present: {seasons}')
print(f'Pitchers: {fm["pitcher"].nunique():,}')
print()
print('Label balance:')
for col in ['injured_next_30d', 'injured_next_60d', 'injured_next_90d']:
    rate = fm[col].mean()
    print(f'  {col:20s}  positive rate = {rate:.1%}  (n={int(fm[col].sum())})')\
"""

C3_MD = """\
## 2. Temporal Train / Test Split

Models are evaluated on seasons they have never seen — this is the only
honest way to estimate how Injury Risk+ would perform going forward. With the
full 2015–2024 pull, the two most recent seasons become the held-out test set
and everything earlier is used for training (the `prepare_classification_dataset`
helper falls back gracefully to a single-season chronological split if only one
season is present, e.g. when running in TEST_MODE).\
"""

C3 = """\
TARGET = 'injured_next_30d'

X_train, X_test, y_train, y_test = prepare_classification_dataset(
    fm, target_col=TARGET,
)

print(f'Train: {X_train.shape[0]:,} rows  | positive rate = {y_train.mean():.1%}')
print(f'Test : {X_test.shape[0]:,} rows  | positive rate = {y_test.mean():.1%}')
print(f'Features: {X_train.shape[1]}')
print(f'Train seasons: {sorted(fm.loc[X_train.index, "season"].unique().tolist())}')
print(f'Test  seasons: {sorted(fm.loc[X_test.index, "season"].unique().tolist())}')\
"""

C4_MD = """\
## 3. Train Baseline Models

Four models, increasing in sophistication:

| Model | Why it's here |
|---|---|
| Naive (historical rate × archetype × age band) | Performance floor — anything we ship must beat a lookup table |
| Logistic Regression | Interpretable linear baseline; coefficients are directly inspectable |
| Random Forest | Captures non-linearities and interactions without much tuning |
| XGBoost | Strong tabular baseline; usually the benchmark to beat |

All tree/linear models are wrapped in pipelines that median-impute missing
values (rolling-window features are null early in a pitcher's observed
history) before fitting.\
"""

C4 = """\
%%time
models = {}

print('Training naive baseline (historical rate by archetype x age band)...')
models['naive'] = train_naive_baseline(fm.loc[X_train.index], strategy='historical_rate')

print('Training logistic regression...')
models['logistic_regression'] = train_logistic_regression(X_train, y_train)

print('Training random forest (grid search over depth / leaf size)...')
models['random_forest'] = train_random_forest(X_train, y_train)

print('Training XGBoost...')
models['xgboost'] = train_gradient_boosting(X_train, y_train, framework='xgboost')

print('\\nAll models trained:', list(models.keys()))\
"""

C5_MD = """\
## 4. Evaluate on the Held-Out Test Set\
"""

C5 = """\
results = []
test_probs = {}
for name, model in models.items():
    proba = model.predict_proba(X_test)[:, 1]
    test_probs[name] = proba
    metrics = evaluate_classifier(y_test, proba)
    metrics['model'] = name
    results.append(metrics)

results_df = pd.DataFrame(results).set_index('model')[
    ['auc_roc', 'pr_auc', 'brier_score', 'accuracy', 'precision', 'recall', 'f1', 'mcc']
].sort_values('auc_roc', ascending=False)

print(f'Held-out test set — target = {TARGET}')
display(results_df.style.format('{:.3f}'))\
"""

C6_MD = """\
## 5. Calibration

A well-calibrated model's predicted probabilities should match observed
injury rates — this matters enormously for Injury Risk+, since the score is
built directly on top of `injury_prob_30d`. We plot reliability diagrams for
the two strongest models.\
"""

C6 = """\
best_two = results_df.index[:2].tolist()

fig, axes = plt.subplots(1, len(best_two), figsize=(6 * len(best_two), 5.5))
if len(best_two) == 1:
    axes = [axes]

for ax, name in zip(axes, best_two):
    sub_fig = plot_calibration_curve(y_test, test_probs[name])
    sub_ax = sub_fig.axes[0]
    for line in sub_ax.get_lines():
        ax.plot(line.get_xdata(), line.get_ydata(), marker=line.get_marker(), label=line.get_label())
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax.set_xlabel('Mean predicted probability')
    ax.set_ylabel('Observed injury rate')
    ax.set_title(f'Calibration — {name}')
    ax.legend()
    plt.close(sub_fig)

fig.tight_layout()
fig_path = FIGURES_DIR / 'fig_20_baseline_calibration.png'
fig.savefig(fig_path, dpi=120, bbox_inches='tight')
plt.show()
print(f'Saved {fig_path}')\
"""

C7_MD = """\
## 6. Feature Importance

What is the strongest tree-based model actually using? This is a first look —
notebook 10 covers SHAP-based interpretability in depth.\
"""

C7 = """\
top_tree_model_name = results_df.index[results_df.index.isin(['random_forest', 'xgboost'])][0]
top_tree_model = models[top_tree_model_name]

importance_df = compute_feature_importance(top_tree_model, X_train.columns.tolist())
print(f'Top 15 features — {top_tree_model_name}:')
display(importance_df.head(15))

fig, ax = plt.subplots(figsize=(8, 6))
top15 = importance_df.head(15).iloc[::-1]
ax.barh(top15['feature'], top15['importance'], color='#4C72B0')
ax.set_xlabel('Importance')
ax.set_title(f'Top 15 features — {top_tree_model_name}')
fig.tight_layout()
fig_path = FIGURES_DIR / 'fig_21_baseline_feature_importance.png'
fig.savefig(fig_path, dpi=120, bbox_inches='tight')
plt.show()
print(f'Saved {fig_path}')\
"""

C8_MD = """\
## 7. Walk-Forward Temporal Cross-Validation

A single train/test split can be lucky or unlucky. Walk-forward CV trains on
seasons `[..., k]` and tests on season `k+1`, repeating across the available
history — this gives a much more honest read on stability than one holdout.
We run it for the strongest model from the comparison above.\
"""

C8 = """\
best_model_name = results_df.index[0]
print(f'Running temporal CV for: {best_model_name}')

if best_model_name == 'logistic_regression':
    model_fn = train_logistic_regression
elif best_model_name == 'random_forest':
    model_fn = train_random_forest
elif best_model_name == 'xgboost':
    model_fn = lambda X, y: train_gradient_boosting(X, y, framework='xgboost')
else:
    model_fn = train_logistic_regression

cv_df = temporal_cross_validate(model_fn, fm, n_splits=5, target_col=TARGET)
display(cv_df[['fold', 'test_season', 'n_test', 'auc_roc', 'pr_auc', 'brier_score']])

if len(cv_df):
    print()
    print(f'Mean AUC-ROC across folds: {cv_df["auc_roc"].mean():.3f} '
          f'(std {cv_df["auc_roc"].std():.3f})')\
"""

C9_MD = """\
## 8. Multi-Horizon Comparison (30 / 60 / 90 days)

How does the strongest model's discrimination change as the prediction
horizon widens? Longer horizons are easier to predict (more positive cases,
less precise timing required) but less actionable.\
"""

C9 = """\
horizon_rows = []
for horizon_col in ['injured_next_30d', 'injured_next_60d', 'injured_next_90d']:
    Xh_train, Xh_test, yh_train, yh_test = prepare_classification_dataset(fm, target_col=horizon_col)
    if best_model_name == 'logistic_regression':
        m = train_logistic_regression(Xh_train, yh_train)
    elif best_model_name == 'random_forest':
        m = train_random_forest(Xh_train, yh_train)
    elif best_model_name == 'xgboost':
        m = train_gradient_boosting(Xh_train, yh_train, framework='xgboost')
    else:
        m = train_logistic_regression(Xh_train, yh_train)
    proba = m.predict_proba(Xh_test)[:, 1]
    metrics = evaluate_classifier(yh_test, proba)
    metrics['horizon'] = horizon_col
    metrics['positive_rate_test'] = float(yh_test.mean())
    horizon_rows.append(metrics)

horizon_df = pd.DataFrame(horizon_rows).set_index('horizon')[
    ['positive_rate_test', 'auc_roc', 'pr_auc', 'brier_score', 'f1']
]
display(horizon_df.style.format('{:.3f}'))\
"""

C10_MD = """\
## 9. Save Models and Results\
"""

C10 = """\
for name, model in models.items():
    out_path = MODELS_DIR / f'baseline_{name}.joblib'
    save_model(model, str(out_path))
    print(f'Saved {out_path}')

results_out = results_df.reset_index()
results_out.to_csv(TABLES_DIR / 'baseline_model_results.csv', index=False)
print(f'\\nSaved {TABLES_DIR / "baseline_model_results.csv"}')

cv_df.to_csv(TABLES_DIR / 'baseline_temporal_cv_results.csv', index=False)
print(f'Saved {TABLES_DIR / "baseline_temporal_cv_results.csv"}')

horizon_df.reset_index().to_csv(TABLES_DIR / 'baseline_horizon_comparison.csv', index=False)
print(f'Saved {TABLES_DIR / "baseline_horizon_comparison.csv"}')\
"""

C11_MD = """\
## 10. Summary

* **Best model:** see the AUC-ROC ranking in section 4 — this becomes the
  benchmark that survival (07) and multi-task (08) models must beat.
* **Calibration:** reliability diagrams in section 5 show whether predicted
  probabilities can be trusted at face value, or whether `score_calibration`
  (notebook 09) needs to apply isotonic/Platt correction before they feed
  Injury Risk+.
* **Stability:** walk-forward CV (section 7) is the more trustworthy estimate
  of real-world performance than any single train/test split.
* **Next step:** notebook 07 reframes this as a time-to-event problem —
  survival models can use *all* pitchers (including censored ones) rather than
  just those with a binary label at a fixed horizon.\
"""

C11 = """\
provenance = {
    'notebook': '06_baseline_models',
    'run_at': datetime.now(timezone.utc).isoformat(),
    'seasons_used': seasons,
    'n_train': int(len(X_train)),
    'n_test': int(len(X_test)),
    'target': TARGET,
    'best_model': best_model_name,
    'test_metrics': results_df.loc[best_model_name].to_dict(),
}
print(json.dumps(provenance, indent=2, default=str))

prov_path = TABLES_DIR / 'baseline_model_provenance.json'
prov_path.write_text(json.dumps(provenance, indent=2, default=str))
print(f'\\nSaved {prov_path}')\
"""

# ---------------------------------------------------------------------------
# ASSEMBLE
# ---------------------------------------------------------------------------

cells = [
    md(C0,    "c00-header"),
    code(C1,  "c01-setup"),
    md(C2_MD, "c02-load-header"),
    code(C2,  "c02-load"),
    md(C3_MD, "c03-split-header"),
    code(C3,  "c03-split"),
    md(C4_MD, "c04-train-header"),
    code(C4,  "c04-train"),
    md(C5_MD, "c05-eval-header"),
    code(C5,  "c05-eval"),
    md(C6_MD, "c06-calib-header"),
    code(C6,  "c06-calib"),
    md(C7_MD, "c07-importance-header"),
    code(C7,  "c07-importance"),
    md(C8_MD, "c08-cv-header"),
    code(C8,  "c08-cv"),
    md(C9_MD, "c09-horizon-header"),
    code(C9,  "c09-horizon"),
    md(C10_MD,"c10-save-header"),
    code(C10, "c10-save"),
    md(C11_MD,"c11-summary-header"),
    code(C11, "c11-provenance"),
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

out = Path("notebooks/06_baseline_models.ipynb")
out.parent.mkdir(exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Written: {out}  ({len(cells)} cells)")
