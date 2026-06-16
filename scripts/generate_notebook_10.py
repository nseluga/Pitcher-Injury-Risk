"""
Generate notebooks/10_model_interpretability.ipynb programmatically.
Run from project root: python scripts/generate_notebook_10.py
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
# 10 — Model Interpretability

## Purpose
Understand what drives the trained injury prediction models using SHAP (SHapley
Additive exPlanations) and partial dependence plots. The goal is threefold:

1. Validate that learned feature importances align with baseball medicine knowledge
   (prior injury history, workload, velocity) — building trust in the model.
2. Surface novel or surprising predictors that could inform future research.
3. Produce publication-ready global and local explanations.

## Inputs
- `data/processed/feature_matrix.parquet`
- `models/baseline_xgboost_tuned.joblib` (preferred classifier)
- `models/multitask_chained.joblib` (fallback / secondary reference)

## Outputs
- `reports/figures/shap_global_importance.png`
- `reports/figures/shap_beeswarm.png`
- `reports/figures/partial_dependence_workload.png`
- `reports/figures/partial_dependence_velocity.png`
- `reports/figures/partial_dependence_injury_history.png`

## Analyses
1. SHAP global feature importance (mean |SHAP|)
2. SHAP beeswarm — direction and magnitude per feature
3. Partial dependence plots for top workload, velocity, and injury-history features
4. Local explanations: SHAP waterfall for one high-risk and one low-risk pitcher
5. Domain validation commentary\
"""

C1 = """\
import sys, warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import joblib
import shap

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 60)
shap.initjs()

PROJECT_ROOT = str(Path('.').resolve())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.baseline_models import _infer_feature_cols

MODELS_DIR  = Path('models')
FIGURES_DIR = Path('reports/figures')
TABLES_DIR  = Path('reports/tables')
for d in (FIGURES_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Configuration ---
TEST_MODE = False
SHAP_SAMPLE_SIZE = 500 if TEST_MODE else 5000

print(f'TEST_MODE={TEST_MODE}, SHAP_SAMPLE_SIZE={SHAP_SAMPLE_SIZE}')
print(f'shap version: {shap.__version__}')\
"""

C2_MD = """\
## 1. Load Feature Matrix and Best Classifier

We use the tuned XGBoost pipeline from notebook 06 as the primary subject for
SHAP analysis: it has the best PR-AUC among baseline classifiers and, as a tree
ensemble, admits exact TreeSHAP in O(T·D) time. The multitask chained model is
referenced for comparison where relevant.\
"""

C2 = """\
fm_path = Path('data/processed/feature_matrix.parquet')
if not fm_path.exists():
    raise FileNotFoundError('Run notebook 05 first to build the feature matrix.')

fm = pd.read_parquet(fm_path)
fm['game_date'] = pd.to_datetime(fm['game_date'])
feature_cols = _infer_feature_cols(fm)
X_all = fm[feature_cols].copy()

xgb_path = MODELS_DIR / 'baseline_xgboost_tuned.joblib'
if not xgb_path.exists():
    xgb_path = MODELS_DIR / 'baseline_xgboost.joblib'
    print(f'Tuned XGBoost not found; falling back to {xgb_path}')

xgb_pipeline = joblib.load(xgb_path)
imputer   = xgb_pipeline.named_steps['imputer']
xgb_model = xgb_pipeline.named_steps['clf']

# Imputer may drop entirely-NaN features; use its actual output names.
imp_feature_cols = list(imputer.get_feature_names_out())
dropped = set(feature_cols) - set(imp_feature_cols)
if dropped:
    print(f'Note: imputer dropped {len(dropped)} all-NaN feature(s): {dropped}')

print(f'Feature matrix: {fm.shape[0]:,} rows x {len(feature_cols)} input features, '
      f'{len(imp_feature_cols)} after imputation')
print(f'Loaded XGBoost from: {xgb_path}')
print(f'Pipeline steps: {[s[0] for s in xgb_pipeline.steps]}')\
"""

C3_MD = """\
## 2. Prepare SHAP Sample

SHAP TreeExplainer scales linearly with sample size for tree models — 205k
rows takes ~60 s but is unnecessary for global explanations. We draw a
stratified sample (by `injured_next_30d`) so the positive class is
representated, and impute it through the pipeline's imputer step so the SHAP
values are computed on the actual imputed space the model sees.\
"""

C3 = """\
y_all = fm['injured_next_30d']
rng = np.random.default_rng(42)

pos_idx = np.where(y_all == 1)[0]
neg_idx = np.where(y_all == 0)[0]
n_pos = min(len(pos_idx), SHAP_SAMPLE_SIZE // 5)
n_neg = SHAP_SAMPLE_SIZE - n_pos

sampled_pos = rng.choice(pos_idx, size=n_pos, replace=False)
sampled_neg = rng.choice(neg_idx, size=n_neg, replace=False)
sampled_idx = np.sort(np.concatenate([sampled_pos, sampled_neg]))

X_sample_raw = X_all.iloc[sampled_idx].copy()
y_sample     = y_all.iloc[sampled_idx].values

X_sample_imp = pd.DataFrame(
    imputer.transform(X_sample_raw),
    columns=imp_feature_cols,
    index=X_sample_raw.index,
)

print(f'SHAP sample: {len(X_sample_imp):,} rows '
      f'({n_pos} injured, {n_neg} not-injured)')
print(f'Positive rate in sample: {y_sample.mean():.1%}')\
"""

C4_MD = """\
## 3. SHAP Global Feature Importance

We use `TreeExplainer` (exact Shapley values via the path-dependent
algorithm) to compute SHAP values for the injury-within-30-days class.
The global importance bar chart ranks features by mean absolute SHAP value —
a model-agnostic importance that accounts for interaction effects and direction.\
"""

C4 = """\
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_sample_imp)

# For binary XGBoost the output is a 2D array (n_samples x n_features).
if isinstance(shap_values, list):
    sv = shap_values[1]   # positive class
else:
    sv = shap_values

mean_abs_shap = np.abs(sv).mean(axis=0)
importance_df = pd.DataFrame({
    'feature': imp_feature_cols,
    'mean_abs_shap': mean_abs_shap,
}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)

print('Top 20 features by mean |SHAP|:')
display(importance_df.head(20))

fig, ax = plt.subplots(figsize=(9, 7))
top20 = importance_df.head(20)
ax.barh(top20['feature'][::-1], top20['mean_abs_shap'][::-1], color='steelblue')
ax.set_xlabel('Mean |SHAP value|')
ax.set_title('SHAP Global Feature Importance — Injury within 30 days')
fig.tight_layout()
out_path = FIGURES_DIR / 'shap_global_importance.png'
fig.savefig(out_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out_path}')\
"""

C5_MD = """\
## 4. SHAP Beeswarm Plot

The beeswarm adds _direction_: red = high feature value, blue = low; position
on the x-axis = SHAP contribution (positive = increases injury risk). This
lets us see whether high workload monotonically increases risk, whether velocity
decline has a nonlinear pattern, etc.\
"""

C5 = """\
shap_explanation = shap.Explanation(
    values=sv,
    base_values=explainer.expected_value if not isinstance(explainer.expected_value, list)
                else explainer.expected_value[1],
    data=X_sample_imp.values,
    feature_names=imp_feature_cols,
)

fig, ax = plt.subplots(figsize=(10, 8))
shap.plots.beeswarm(shap_explanation, max_display=20, show=False)
fig = plt.gcf()
fig.suptitle('SHAP Beeswarm — Injury within 30 days', y=1.01, fontsize=12)
out_path = FIGURES_DIR / 'shap_beeswarm.png'
fig.savefig(out_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out_path}')\
"""

C6_MD = """\
## 5. Partial Dependence Plots

Partial dependence isolates the marginal effect of one feature while averaging
over the joint distribution of all others. We group features into three
clinically meaningful categories — **workload**, **velocity**, and **injury
history** — and show the two highest-SHAP representatives of each. PDP uses
a random subsample (1 000 rows) for speed; the effect shapes are stable at
that size.\
"""

C6 = """\
from sklearn.inspection import partial_dependence

# Keyword-based grouping against the actual post-imputation feature names.
FEATURE_KEYWORDS = {
    'workload':       ['pitch', 'acwr', 'rest', 'appear'],
    'velocity':       ['velo', 'fb_v', 'velo_change', 'velo_delta'],
    'injury_history': ['prior_il', 'days_since', 'injury'],
}

# PDP is called on the full pipeline, so pass raw (pre-imputation) feature cols.
# Use a subsample to keep it fast.
pdp_sample_size = min(1000, len(fm))
pdp_raw_idx = rng.choice(len(fm), size=pdp_sample_size, replace=False)
X_pdp_raw = X_all.iloc[pdp_raw_idx].reset_index(drop=True)

for group_name, keywords in FEATURE_KEYWORDS.items():
    # Find features from imp_feature_cols whose name contains any keyword.
    def _matches(f):
        return any(k in f for k in keywords)
    group_feats_in_imp = [f for f in imp_feature_cols if _matches(f)]
    if not group_feats_in_imp:
        print(f'[{group_name}] no matching features in imp_feature_cols, skipping')
        continue

    group_imp = importance_df[importance_df['feature'].isin(group_feats_in_imp)]
    top_feats = group_imp.head(2)['feature'].tolist()
    if not top_feats:
        print(f'[{group_name}] no features in importance_df, skipping')
        continue

    fig, axes = plt.subplots(1, len(top_feats), figsize=(6 * len(top_feats), 4))
    if len(top_feats) == 1:
        axes = [axes]

    for ax, feat in zip(axes, top_feats):
        feat_idx = feature_cols.index(feat)
        pdp_result = partial_dependence(
            xgb_pipeline, X_pdp_raw[feature_cols],
            features=[feat_idx], kind='average', grid_resolution=40,
        )
        grid_vals = pdp_result['grid_values'][0]
        avg_pred  = pdp_result['average'][0]
        ax.plot(grid_vals, avg_pred, color='steelblue', linewidth=2)
        ax.set_xlabel(feat)
        ax.set_ylabel('Predicted injury prob')
        ax.set_title(f'PDP: {feat}')
        ax.tick_params(axis='x', rotation=20)

    fig.suptitle(f'Partial Dependence — {group_name.replace("_", " ").title()} features',
                 fontsize=12)
    fig.tight_layout()
    out_path = FIGURES_DIR / f'partial_dependence_{group_name}.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path} (features: {top_feats})')\
"""

C7_MD = """\
## 6. Local Explanations: Case Studies

We pick one **high-risk** pitcher-appearance (highest predicted injury probability)
and one **low-risk** appearance (lowest) and show their SHAP waterfall plots.
This answers: *which specific features drove this individual prediction?*
These are the explanations a team analyst would share with a coaching staff.\
"""

C7 = """\
X_all_imp = pd.DataFrame(
    imputer.transform(X_all),
    columns=imp_feature_cols,
    index=X_all.index,
)
all_probs = xgb_model.predict_proba(X_all_imp)[:, 1]

high_risk_iloc = int(np.argmax(all_probs))
low_risk_iloc  = int(np.argmin(all_probs))

high_risk_row = fm.iloc[high_risk_iloc]
low_risk_row  = fm.iloc[low_risk_iloc]

print(f'Highest-risk: pitcher={high_risk_row["pitcher"]}, '
      f'date={high_risk_row["game_date"].date()}, '
      f'season={high_risk_row["season"]}, '
      f'predicted_prob={all_probs[high_risk_iloc]:.3f}')
print(f'Lowest-risk:  pitcher={low_risk_row["pitcher"]}, '
      f'date={low_risk_row["game_date"].date()}, '
      f'season={low_risk_row["season"]}, '
      f'predicted_prob={all_probs[low_risk_iloc]:.3f}')

base_val = explainer.expected_value if not isinstance(explainer.expected_value, list) \
           else explainer.expected_value[1]

for label, iloc in [('high_risk', high_risk_iloc), ('low_risk', low_risk_iloc)]:
    x_row = X_all_imp.iloc[[iloc]]
    sv_row = explainer.shap_values(x_row)
    if isinstance(sv_row, list):
        sv_row = sv_row[1]
    sv_row = sv_row[0]

    exp = shap.Explanation(
        values=sv_row,
        base_values=base_val,
        data=x_row.values[0],
        feature_names=imp_feature_cols,
    )
    shap.plots.waterfall(exp, max_display=15, show=False)
    fig = plt.gcf()
    prob = all_probs[iloc]
    fig.suptitle(f'SHAP Waterfall — {label.replace("_", " ").title()} '
                 f'(predicted prob={prob:.3f})', y=1.01, fontsize=11)
    out_path = FIGURES_DIR / f'shap_waterfall_{label}.png'
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')\
"""

C8_MD = """\
## 7. Domain Validation Commentary

We compare the top-ranked SHAP features against what baseball medicine literature
and practitioner knowledge tell us to expect. Misalignment would be a signal
to re-examine the feature engineering or labeling pipeline.\
"""

C8 = """\
EXPECTED_IMPORTANT = {
    'prior_il_total':       'prior IL stints (injury history) should rank in top 10',
    'pitches_7d':           'acute workload (7-day pitch count) should rank highly',
    'acwr_7_28':            'acute-chronic workload ratio is a standard injury predictor',
    'days_rest':            'short rest increases arm injury risk',
    'fb_velo_mean':         'velocity is a canonical injury precursor',
    'velo_delta_vs_season': 'velocity spike/decline signals biomechanical change',
    'days_since_last_injury': 'recency of last injury affects re-injury risk',
}

top10 = set(importance_df.head(10)['feature'].tolist())
top20 = set(importance_df.head(20)['feature'].tolist())

print('Domain alignment check — expected important features:')
for feat, expectation in EXPECTED_IMPORTANT.items():
    if feat not in imp_feature_cols:
        tier = 'NOT IN MODEL'
    elif feat in top10:
        tier = 'TOP 10'
    elif feat in top20:
        tier = 'TOP 20'
    else:
        tier = 'NOT IN TOP 20'
    print(f'  [{tier:>14}] {feat}: {expectation}')

print()
print('Top 10 features by SHAP (any domain surprises?):')
for i, row in importance_df.head(10).iterrows():
    print(f'  {i+1:2d}. {row["feature"]:40s}  mean|SHAP|={row["mean_abs_shap"]:.4f}')\
"""

C9_MD = """\
## 8. Save Summary and Provenance\
"""

C9 = """\
importance_df.to_csv(TABLES_DIR / 'shap_global_importance.csv', index=False)
print(f'Saved {TABLES_DIR}/shap_global_importance.csv')

provenance = {
    'notebook': '10_model_interpretability',
    'run_at': datetime.now(timezone.utc).isoformat(),
    'model_used': str(xgb_path),
    'feature_matrix_shape': list(fm.shape),
    'shap_sample_size': SHAP_SAMPLE_SIZE,
    'top_10_features': importance_df.head(10)['feature'].tolist(),
}
import json
prov_path = TABLES_DIR / 'interpretability_provenance.json'
prov_path.write_text(json.dumps(provenance, indent=2))
print(f'Saved {prov_path}')
print()
print('Notebook 10 complete. Outputs written to reports/figures/')\
"""

# ---------------------------------------------------------------------------
# ASSEMBLE
# ---------------------------------------------------------------------------

cells = [
    md(C0,    "c00-header"),
    code(C1,  "c01-setup"),
    md(C2_MD, "c02-load-header"),
    code(C2,  "c02-load"),
    md(C3_MD, "c03-sample-header"),
    code(C3,  "c03-sample"),
    md(C4_MD, "c04-global-header"),
    code(C4,  "c04-global"),
    md(C5_MD, "c05-beeswarm-header"),
    code(C5,  "c05-beeswarm"),
    md(C6_MD, "c06-pdp-header"),
    code(C6,  "c06-pdp"),
    md(C7_MD, "c07-local-header"),
    code(C7,  "c07-local"),
    md(C8_MD, "c08-domain-header"),
    code(C8,  "c08-domain"),
    md(C9_MD, "c09-save-header"),
    code(C9,  "c09-save"),
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

out = Path("notebooks/10_model_interpretability.ipynb")
out.parent.mkdir(exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Written: {out}  ({len(cells)} cells)")
