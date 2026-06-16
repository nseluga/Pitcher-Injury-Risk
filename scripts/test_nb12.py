"""Quick TEST_MODE validation for NB12 simulation functions."""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import joblib

sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

from src.models.baseline_models import _infer_feature_cols
from src.simulation.workload_simulator import (
    simulate_pitch_count_reduction, simulate_rest_schedule, find_optimal_pitch_count
)
from src.simulation.pitch_mix_simulator import (
    simulate_slider_reduction, compute_pitch_type_risk_sensitivity
)
from src.simulation.usage_strategy_simulator import compare_starter_vs_hybrid

MODELS_DIR = Path('models')

fm = pd.read_parquet('data/processed/feature_matrix.parquet')
fm['game_date'] = pd.to_datetime(fm['game_date'])
feature_cols = _infer_feature_cols(fm)

xgb_path = MODELS_DIR / 'baseline_xgboost_tuned.joblib'
xgb_pipeline = joblib.load(xgb_path)

scores = pd.read_parquet(
    'data/processed/injury_risk_plus_scores.parquet',
    columns=['pitcher_id', 'season', 'archetype']
)
scores = scores.drop_duplicates(subset=['pitcher_id', 'season'])
fm = fm.merge(scores.rename(columns={'pitcher_id': 'pitcher'}), on=['pitcher', 'season'], how='left')
print('Archetype distribution:', fm['archetype'].value_counts().to_dict())

league_avg_prob = float(fm['injured_next_30d'].mean())
model_dict = {
    'xgb_pipeline': xgb_pipeline,
    'feature_cols': feature_cols,
}

N = 10

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
profiles_starters = profiles_all[profiles_all['archetype'] == 'starter'].copy()
sample_starters = profiles_starters.sample(n=min(N, len(profiles_starters)), random_state=42).reset_index(drop=True)
sample_all = profiles_all.sample(n=min(N, len(profiles_all)), random_state=42).reset_index(drop=True)
print(f'Sample: {len(sample_starters)} starters, {len(sample_all)} all')

print('Testing pitch count...')
for pc in [60, 90, 110]:
    probs = []
    for _, row in sample_starters.iterrows():
        orig_count = float(row.get('pitch_count', 95.0) or 95.0)
        r = simulate_pitch_count_reduction(row, orig_count - pc, model_dict)
        probs.append(r['counterfactual_prob'])
    print(f'  pc={pc}: mean_prob={np.mean(probs):.4f}')

print('Testing rest schedule...')
for rest in [2, 5, 9]:
    probs = []
    for _, row in sample_all.iterrows():
        r = simulate_rest_schedule(row, rest, model_dict)
        probs.append(r['counterfactual_prob'])
    print(f'  rest={rest}: mean_prob={np.mean(probs):.4f}')

print('Testing slider reduction...')
r = simulate_slider_reduction(sample_all.iloc[0], 0.10, model_dict)
print(f'  slider: orig={r["original_prob"]:.4f} cf={r["counterfactual_prob"]:.4f}')

print('Testing pitch sensitivity...')
if len(profiles_starters) > 0:
    rep = profiles_starters.iloc[0]
    sens = compute_pitch_type_risk_sensitivity(rep, model_dict)
    print(sens.to_string(index=False))

print('Testing role comparison...')
comp = compare_starter_vs_hybrid(fm.sample(n=3000, random_state=42), model_dict)
print(comp.to_string(index=False))

print()
print('All TEST_MODE checks passed!')
