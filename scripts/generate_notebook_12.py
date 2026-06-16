"""
Generate notebooks/12_usage_strategy_simulation.ipynb programmatically.
Run from project root: python scripts/generate_notebook_12.py
"""

import json
from pathlib import Path


def md(source: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source}


def code(source: str, cell_id: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


# ---------------------------------------------------------------------------
# CELL SOURCES
# ---------------------------------------------------------------------------

C0 = """\
# 12 — Usage Strategy Simulation

## Purpose
Use the trained injury prediction models to simulate alternative pitcher usage
strategies and estimate their effect on injury risk. All simulations are
**counterfactual** (model-based): we modify specific features in a pitcher's
feature vector while holding all others constant, then re-run model inference.

**Important limitation:** These are ceteris paribus simulations. In reality,
changing pitch count or pitch mix affects performance and sequencing decisions.
These results should be interpreted as model-conditional predictions, not causal
estimates.

## Inputs
- `data/processed/feature_matrix.parquet`
- `models/baseline_xgboost_tuned.joblib`
- `data/processed/injury_risk_plus_scores.parquet`

## Outputs
- `reports/tables/simulation_results.csv`
- `reports/figures/pitch_count_optimization.png`
- `reports/figures/rest_schedule_optimization.png`
- `reports/figures/slider_reduction_analysis.png`
- `reports/figures/role_transition_impact.png`

## Analyses
1. Pitch count optimization: how does predicted injury probability change as
   pitch count varies from 50 to 120?
2. Rest schedule optimization: what is the model-predicted effect of varying
   days of rest between appearances?
3. Slider reduction: how much does reducing slider usage change predicted risk?
4. Role comparison: starters vs. relievers — predicted and observed injury rates\
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
import joblib

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 40)

PROJECT_ROOT = str(Path('.').resolve())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.baseline_models import _infer_feature_cols
from src.simulation.workload_simulator import (
    simulate_pitch_count_reduction,
    simulate_rest_schedule,
    find_optimal_pitch_count,
)
from src.simulation.pitch_mix_simulator import (
    simulate_slider_reduction,
    find_risk_minimizing_pitch_mix,
    compute_pitch_type_risk_sensitivity,
)
from src.simulation.usage_strategy_simulator import (
    simulate_role_transition,
    compare_starter_vs_hybrid,
    simulate_staff_workload_distribution,
)

MODELS_DIR  = Path('models')
FIGURES_DIR = Path('reports/figures')
TABLES_DIR  = Path('reports/tables')
for d in (FIGURES_DIR, TABLES_DIR):
    d.mkdir(parents=True, exist_ok=True)

# --- Configuration ---
TEST_MODE = False
N_SAMPLE_PITCHERS = 30 if TEST_MODE else 300  # pitcher profiles to simulate over

print(f'TEST_MODE={TEST_MODE}, N_SAMPLE_PITCHERS={N_SAMPLE_PITCHERS}')\
"""

C2_MD = """\
## 1. Load Models and Feature Matrix

We use the tuned XGBoost pipeline from notebook 06 as the primary model for
all counterfactual simulations. It has the best PR-AUC among single classifiers
and accepts the same feature vector as the feature matrix, making single-row
inference straightforward.

We merge the archetype labels from the Injury Risk+ scores parquet so role
comparisons can be made without re-clustering.\
"""

C2 = """\
fm_path = Path('data/processed/feature_matrix.parquet')
if not fm_path.exists():
    raise FileNotFoundError('Run notebook 05 first to build the feature matrix.')

fm = pd.read_parquet(fm_path)
fm['game_date'] = pd.to_datetime(fm['game_date'])
feature_cols = _infer_feature_cols(fm)

xgb_path = MODELS_DIR / 'baseline_xgboost_tuned.joblib'
if not xgb_path.exists():
    xgb_path = MODELS_DIR / 'baseline_xgboost.joblib'
    print(f'Tuned XGBoost not found; using {xgb_path}')

xgb_pipeline = joblib.load(xgb_path)

# Merge archetype from scores parquet.
scores_path = Path('data/processed/injury_risk_plus_scores.parquet')
if scores_path.exists():
    scores = pd.read_parquet(scores_path, columns=['pitcher_id', 'season', 'archetype'])
    scores = scores.drop_duplicates(subset=['pitcher_id', 'season'])
    fm = fm.merge(
        scores.rename(columns={'pitcher_id': 'pitcher'}),
        on=['pitcher', 'season'], how='left',
    )
    print(f'Archetype distribution: {fm["archetype"].value_counts().to_dict()}')
else:
    fm['archetype'] = 'unknown'
    print('scores parquet not found; archetype set to unknown')

league_avg_prob = float(fm['injured_next_30d'].mean())
model_dict = {
    'xgb_pipeline': xgb_pipeline,
    'feature_cols': feature_cols,
    'league_avg_prob': league_avg_prob,
}

print(f'Feature matrix: {fm.shape[0]:,} rows x {len(feature_cols)} features')
print(f'League avg injury prob (30d): {league_avg_prob:.3%}')
print(f'Loaded model: {xgb_path}')\
"""

C3_MD = """\
## 2. Sample Pitcher Profiles

To keep simulations computationally tractable, we select one representative
game per pitcher-season (the median appearance by pitch count) rather than
running inference on all 200k+ rows. This gives a cross-sectional profile
of each pitcher's typical workload in a given season.\
"""

C3 = """\
# One profile per pitcher-season: pick the game closest to that pitcher-season's median pitch count.
def median_profile(grp):
    if 'pitch_count' in grp.columns and grp['pitch_count'].notna().any():
        med = grp['pitch_count'].median()
        idx = (grp['pitch_count'] - med).abs().idxmin()
    else:
        idx = grp.index[len(grp) // 2]
    return grp.loc[idx]

profiles_all = (
    fm.groupby(['pitcher', 'season'], group_keys=False)
    .apply(median_profile)
    .reset_index(drop=True)
)

# Optionally restrict to starters for workload sims (they have meaningful pitch counts).
starters_mask = profiles_all['archetype'] == 'starter'
profiles_starters = profiles_all[starters_mask].copy()

# Use a random sample for the simulation loops.
rng = np.random.default_rng(42)
n_starters = min(N_SAMPLE_PITCHERS, len(profiles_starters))
n_all      = min(N_SAMPLE_PITCHERS, len(profiles_all))
sample_starters = profiles_starters.sample(n=n_starters, random_state=42).reset_index(drop=True)
sample_all      = profiles_all.sample(n=n_all,      random_state=42).reset_index(drop=True)

print(f'Pitcher-season profiles: {len(profiles_all):,} total, {len(profiles_starters):,} starters')
print(f'Simulation sample: {n_starters} starter profiles, {n_all} all-archetype profiles')\
"""

C4_MD = """\
## 3. Pitch Count Optimization

We vary `pitch_count` from 50 to 120 in steps of 5, proportionally scaling
rolling pitch totals (7-day, 28-day), and measure the mean predicted injury
probability across the sample of pitcher profiles.

**Interpretation:** This curve shows the model's estimated risk as a function
of pitch count, averaged across real pitcher profiles. It is not a causal
estimate — it reflects learned correlations, which include selection effects
(e.g. pitchers who throw more pitches may be healthier on average).\
"""

C4 = """\
pitch_count_range = list(range(50, 125, 5))
pc_results = []

for pc in pitch_count_range:
    probs = []
    for _, row in sample_starters.iterrows():
        orig_count = float(row.get('pitch_count', 95.0) or 95.0)
        reduction = orig_count - pc
        result = simulate_pitch_count_reduction(row, reduction, model_dict)
        probs.append(result['counterfactual_prob'])
    pc_results.append({'pitch_count': pc, 'mean_prob': float(np.mean(probs))})

pc_df = pd.DataFrame(pc_results)
print('Pitch count optimization results:')
print(pc_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(pc_df['pitch_count'], pc_df['mean_prob'] * 100, color='steelblue',
        linewidth=2, marker='o', markersize=5)
ax.set_xlabel('Simulated Pitch Count per Game')
ax.set_ylabel('Mean Predicted 30-Day Injury Probability (%)')
ax.set_title('Pitch Count vs. Predicted Injury Risk\\n(counterfactual simulation, starter sample)')
ax.axhline(league_avg_prob * 100, linestyle='--', color='gray', alpha=0.7, label='League avg')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
out_path = FIGURES_DIR / 'pitch_count_optimization.png'
fig.savefig(out_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out_path}')

opt_row = pc_df.loc[pc_df['mean_prob'].idxmin()]
print(f'Lowest predicted risk at pitch count = {int(opt_row["pitch_count"])} '
      f'({opt_row["mean_prob"]:.3%})')\
"""

C5_MD = """\
## 4. Rest Schedule Optimization

We vary `days_rest` from 1 to 10 and measure the mean predicted injury
probability across the full sample. This captures the model's learned
relationship between rest and injury risk for both starters and relievers.\
"""

C5 = """\
rest_range = list(range(1, 11))
rest_results = []

for rest in rest_range:
    probs = []
    for _, row in sample_all.iterrows():
        result = simulate_rest_schedule(row, rest, model_dict)
        probs.append(result['counterfactual_prob'])
    rest_results.append({'days_rest': rest, 'mean_prob': float(np.mean(probs))})

rest_df = pd.DataFrame(rest_results)
print('Rest schedule optimization results:')
print(rest_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(rest_df['days_rest'], rest_df['mean_prob'] * 100, color='darkorange',
        linewidth=2, marker='o', markersize=6)
ax.set_xlabel('Simulated Days of Rest')
ax.set_ylabel('Mean Predicted 30-Day Injury Probability (%)')
ax.set_title('Days of Rest vs. Predicted Injury Risk\\n(counterfactual simulation)')
ax.axhline(league_avg_prob * 100, linestyle='--', color='gray', alpha=0.7, label='League avg')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
out_path = FIGURES_DIR / 'rest_schedule_optimization.png'
fig.savefig(out_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out_path}')

opt_rest = rest_df.loc[rest_df['mean_prob'].idxmin(), 'days_rest']
print(f'Lowest predicted risk at days_rest = {int(opt_rest)} '
      f'({rest_df.loc[rest_df["mean_prob"].idxmin(), "mean_prob"]:.3%})')\
"""

C6_MD = """\
## 5. Slider Reduction Analysis

Sliders are biomechanically demanding and often associated with elbow injury.
We simulate reducing slider usage from each pitcher's actual rate to a range
of target rates (0% to 40%), redistributing removed usage to the fastball.

This analysis answers: *at the population level, does the model predict lower
injury risk for pitchers with lower slider usage?*\
"""

C6 = """\
slider_targets = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]
slider_results = []

for target in slider_targets:
    probs = []
    for _, row in sample_all.iterrows():
        orig_sl = float(row.get('sl_pct', 0.15) or 0.0)
        if orig_sl >= target:
            # Pitcher throws more sliders than target — simulate reduction
            result = simulate_slider_reduction(row, target, model_dict)
            probs.append(result['counterfactual_prob'])
        else:
            # Already below target — no change; use original probability
            result = simulate_slider_reduction(row, orig_sl, model_dict)
            probs.append(result['original_prob'])
    slider_results.append({'target_sl_pct': target, 'mean_prob': float(np.mean(probs)),
                           'n_pitchers': len(probs)})

slider_df = pd.DataFrame(slider_results)
print('Slider reduction results:')
print(slider_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(slider_df['target_sl_pct'] * 100, slider_df['mean_prob'] * 100,
        color='firebrick', linewidth=2, marker='o', markersize=6)
ax.set_xlabel('Target Slider Usage Rate (%)')
ax.set_ylabel('Mean Predicted 30-Day Injury Probability (%)')
ax.set_title('Slider Usage vs. Predicted Injury Risk\\n(counterfactual simulation)')
ax.axhline(league_avg_prob * 100, linestyle='--', color='gray', alpha=0.7, label='League avg')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
out_path = FIGURES_DIR / 'slider_reduction_analysis.png'
fig.savefig(out_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out_path}')\
"""

C7_MD = """\
## 6. Role Comparison: Starters vs. Relievers

We compare observed injury rates and mean predicted injury probabilities
between starters and relievers. This answers: does the model predict
systematically different risk levels for different roles, and does this
match observed injury frequency?\
"""

C7 = """\
comparison_df = compare_starter_vs_hybrid(fm, model_dict)
print('Role comparison:')
print(comparison_df.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
archetypes = comparison_df['archetype'].tolist()
colors = ['steelblue', 'darkorange', 'forestgreen', 'firebrick'][:len(archetypes)]

axes[0].bar(archetypes, comparison_df['observed_injury_rate'] * 100, color=colors)
axes[0].set_ylabel('Observed 30-Day Injury Rate (%)')
axes[0].set_title('Observed Injury Rate by Role')
axes[0].tick_params(axis='x', rotation=15)

axes[1].bar(archetypes, comparison_df['mean_predicted_prob'] * 100, color=colors)
axes[1].set_ylabel('Mean Predicted Injury Probability (%)')
axes[1].set_title('Predicted Injury Risk by Role')
axes[1].tick_params(axis='x', rotation=15)

for ax in axes:
    ax.axhline(league_avg_prob * 100, linestyle='--', color='gray', alpha=0.6, label='League avg')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

fig.suptitle('Injury Risk Comparison: Starters vs. Relievers', fontsize=13)
fig.tight_layout()
out_path = FIGURES_DIR / 'role_transition_impact.png'
fig.savefig(out_path, dpi=120, bbox_inches='tight')
plt.close(fig)
print(f'Saved {out_path}')\
"""

C8_MD = """\
## 7. Pitch Type Risk Sensitivity

For a representative starter profile, we compute the sensitivity of predicted
injury probability to a +5pp change in each pitch type's usage rate. This
identifies which pitch types the model considers highest-risk.\
"""

C8 = """\
# Use the median starter profile for sensitivity analysis.
if len(profiles_starters) > 0:
    representative = profiles_starters.iloc[len(profiles_starters) // 2]
    sensitivity_df = compute_pitch_type_risk_sensitivity(representative, model_dict, delta=0.05)
    print('Pitch type risk sensitivity (change in injury prob per +5pp usage):')
    print(sensitivity_df.to_string(index=False))
else:
    sensitivity_df = pd.DataFrame()
    print('No starter profiles available for sensitivity analysis')\
"""

C9_MD = """\
## 8. Save Simulation Results

We consolidate all simulation results into a single tidy CSV for downstream
analysis and reporting.\
"""

C9 = """\
all_rows = []

# Pitch count results
for _, r in pc_df.iterrows():
    all_rows.append({'simulation': 'pitch_count', 'parameter': 'pitch_count',
                     'value': r['pitch_count'], 'mean_predicted_prob': r['mean_prob']})

# Rest schedule results
for _, r in rest_df.iterrows():
    all_rows.append({'simulation': 'rest_schedule', 'parameter': 'days_rest',
                     'value': r['days_rest'], 'mean_predicted_prob': r['mean_prob']})

# Slider reduction results
for _, r in slider_df.iterrows():
    all_rows.append({'simulation': 'slider_reduction', 'parameter': 'target_sl_pct',
                     'value': r['target_sl_pct'], 'mean_predicted_prob': r['mean_prob']})

# Role comparison results
for _, r in comparison_df.iterrows():
    all_rows.append({'simulation': 'role_comparison', 'parameter': 'archetype',
                     'value': r['archetype'],
                     'mean_predicted_prob': r['mean_predicted_prob'],
                     'observed_injury_rate': r['observed_injury_rate'],
                     'n_observations': r['n_observations']})

sim_results = pd.DataFrame(all_rows)
out_path = TABLES_DIR / 'simulation_results.csv'
sim_results.to_csv(out_path, index=False)
print(f'Saved {out_path} ({len(sim_results):,} rows)')

provenance = {
    'notebook': '12_usage_strategy_simulation',
    'run_at': datetime.now(timezone.utc).isoformat(),
    'model_used': str(xgb_path),
    'test_mode': TEST_MODE,
    'n_sample_pitchers': N_SAMPLE_PITCHERS,
    'simulations': ['pitch_count_optimization', 'rest_schedule_optimization',
                    'slider_reduction_analysis', 'role_comparison'],
}
import json
(TABLES_DIR / 'simulation_provenance.json').write_text(json.dumps(provenance, indent=2))

print()
print('Notebook 12 complete.')
print(f'  pitch_count_optimization.png — {len(pc_df)} data points')
print(f'  rest_schedule_optimization.png — {len(rest_df)} data points')
print(f'  slider_reduction_analysis.png — {len(slider_df)} data points')
print(f'  role_transition_impact.png — {len(comparison_df)} archetypes')
print(f'  simulation_results.csv — {len(sim_results)} rows')\
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
    md(C4_MD, "c04-pitchcount-header"),
    code(C4,  "c04-pitchcount"),
    md(C5_MD, "c05-rest-header"),
    code(C5,  "c05-rest"),
    md(C6_MD, "c06-slider-header"),
    code(C6,  "c06-slider"),
    md(C7_MD, "c07-role-header"),
    code(C7,  "c07-role"),
    md(C8_MD, "c08-sensitivity-header"),
    code(C8,  "c08-sensitivity"),
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

out = Path("notebooks/12_usage_strategy_simulation.ipynb")
out.parent.mkdir(exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Written: {out}  ({len(cells)} cells)")
