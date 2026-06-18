"""
Generate notebooks/09_risk_score_construction.ipynb programmatically.
Run from project root: python scripts/generate_notebook_09.py
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
# 09 — Injury Risk+ Score Construction

## Purpose
Bring everything from notebooks 06–08 together into a single, interpretable
metric: **Injury Risk+**, a composite, era- and archetype-adjusted score
where 100 = league-average risk for that pitcher's role and season.

> 100 = riskier than average for their archetype/season
> < 100 = safer than average

Full design specification: `docs/injury_risk_plus_design.md`

## Pipeline
1. Calibrate the raw `injury_prob_30d` predictions (isotonic / Platt)
2. Derive the three blend components:
   - `injury_prob_30d` — calibrated probability of injury within 30 days
   - `expected_days_lost` — predicted severity if injured
   - `hazard_rate` — survival-derived instantaneous risk (`1 - S(30)`)
3. Optimize blend weights against observed outcomes (compare to the
   design-doc defaults: 0.50 / 0.30 / 0.20)
4. Compute the raw composite score and normalize it to mean 100 within each
   season × archetype group
5. Produce the final ranked tables: top-risk pitchers, percentile ranks,
   season-over-season stability\
"""

C1 = """\
import sys, json, warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 60)

PROJECT_ROOT = str(Path('.').resolve())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.models.baseline_models import _infer_feature_cols
from src.models.multitask_models import predict_all_tasks
from src.models.survival_models import predict_survival_function
from src.scoring.score_calibration import (
    calibrate_probabilities,
    apply_calibration,
    evaluate_calibration,
    optimize_blend_weights,
    build_normalization_reference,
)
from src.scoring.injury_risk_plus import (
    DEFAULT_WEIGHTS,
    _archetype_for,
    compute_raw_risk_score,
    normalize_to_injury_risk_plus,
    compute_seasonal_risk_plus,
    get_top_risk_pitchers,
    compute_risk_percentile,
)

MODELS_DIR  = Path('models')
TABLES_DIR  = Path('reports/tables')
FIGURES_DIR = Path('reports/figures')
for d in (MODELS_DIR, TABLES_DIR, FIGURES_DIR):
    d.mkdir(parents=True, exist_ok=True)

print('Modules loaded. Default blend weights:', DEFAULT_WEIGHTS)\
"""

C2_MD = """\
## 1. Load Feature Matrix and Upstream Models

We reuse the multi-task model and survival model fitted in notebooks 08 and
07 — Injury Risk+ is explicitly designed as a *downstream consumer* of those
predictions rather than a model in its own right. We load whichever
multi-task architecture won the head-to-head comparison in notebook 08.\
"""

C2 = """\
fm = pd.read_parquet('data/processed/feature_matrix.parquet')
fm['game_date'] = pd.to_datetime(fm['game_date'])
seasons = sorted(fm['season'].unique().tolist())
feature_cols = _infer_feature_cols(fm)

mt_provenance_path = TABLES_DIR / 'multitask_model_provenance.json'
if mt_provenance_path.exists():
    mt_provenance = json.loads(mt_provenance_path.read_text())
    best_architecture = mt_provenance.get('best_architecture', 'chained')
else:
    best_architecture = 'chained'

mt_model_path = MODELS_DIR / f'multitask_{best_architecture}.pkl'
multitask_model = joblib.load(mt_model_path)
print(f'Loaded multi-task model: {mt_model_path} (architecture = {best_architecture})')

surv_provenance_path = TABLES_DIR / 'survival_model_provenance.json'
if surv_provenance_path.exists():
    surv_provenance = json.loads(surv_provenance_path.read_text())
    best_survival_name = surv_provenance.get('best_model', 'cox_ph')
else:
    best_survival_name = 'cox_ph'

survival_model_path = MODELS_DIR / f'survival_{best_survival_name}.pkl'
survival_model = joblib.load(survival_model_path)
print(f'Loaded survival model: {survival_model_path} (model = {best_survival_name})')

print(f'\\nFeature matrix: {fm.shape[0]:,} rows across seasons {seasons}')\
"""

C3_MD = """\
## 2. Build Component Predictions and Calibrate `injury_prob_30d`

Injury Risk+ leans heavily on `injury_prob_30d` (50% of the blend by
default), so its calibration quality directly determines whether the final
score is trustworthy. We hold out the most recent season as a calibration
test set, fit an isotonic calibrator on the remaining data, and verify it
tightens the ECE/MCE/Brier metrics relative to the raw probabilities.\
"""

C3 = """\
X_all = fm[feature_cols]
task_preds = predict_all_tasks(multitask_model, X_all)

surv = predict_survival_function(survival_model, X_all, time_points=[30])
hazard_rate = 1.0 - surv['S(30)'].values

components_raw = pd.DataFrame({
    'injury_prob_30d_raw': task_preds['injured_next_30d_pred'].values,
    'expected_days_lost': task_preds['next_injury_days_lost_pred'].values,
    'hazard_rate': hazard_rate,
}, index=fm.index)

# Calibrate injury_prob_30d on the most recent season; apply to everyone.
calib_season = seasons[-1]
calib_mask = fm['season'] == calib_season
fit_mask = ~calib_mask

calibrator = calibrate_probabilities(
    fm.loc[fit_mask, 'injured_next_30d'],
    components_raw.loc[fit_mask, 'injury_prob_30d_raw'],
    method='isotonic',
)
calibrated_full = apply_calibration(components_raw['injury_prob_30d_raw'], calibrator)
components_raw['injury_prob_30d'] = calibrated_full.values

raw_calib_metrics = evaluate_calibration(
    fm.loc[calib_mask, 'injured_next_30d'], components_raw.loc[calib_mask, 'injury_prob_30d_raw'])
new_calib_metrics = evaluate_calibration(
    fm.loc[calib_mask, 'injured_next_30d'], components_raw.loc[calib_mask, 'injury_prob_30d'])

calib_compare = pd.DataFrame({'raw': raw_calib_metrics, 'calibrated': new_calib_metrics}).T
print(f'Calibration check on held-out season {calib_season}:')
display(calib_compare.style.format('{:.4f}'))\
"""

C4_MD = """\
## 3. Optimize Blend Weights

The design doc specifies default weights (`injury_prob_30d`=0.50,
`expected_days_lost`=0.30, `hazard_rate`=0.20), reflecting the intuition that
*whether* an injury happens matters most, *how bad* it is matters somewhat
less, and the *survival-hazard* signal is a useful but noisier third input.
We search for empirically optimal weights against observed 30-day injury
outcomes and compare them to those defaults — large divergences would be a
signal to revisit the design assumptions.\
"""

C4 = """\
optimized_weights = optimize_blend_weights(
    components_raw.loc[fit_mask, ['injury_prob_30d', 'expected_days_lost', 'hazard_rate']],
    fm.loc[fit_mask, 'injured_next_30d'],
    metric='brier_score',
)

weight_compare = pd.DataFrame({'design_default': DEFAULT_WEIGHTS, 'optimized': optimized_weights}).T
display(weight_compare.style.format('{:.3f}'))

# Use the design-doc defaults for the production score (documented, stable,
# and reviewed) — but keep the optimized weights alongside for comparison.
weights_used = DEFAULT_WEIGHTS
print(f'\\nUsing design-doc default weights for the production score: {weights_used}')\
"""

C5_MD = """\
## 4. Compute Raw and Normalized Injury Risk+ Scores

`compute_seasonal_risk_plus` runs the full pipeline (archetype assignment →
raw blend → season×archetype normalization) in one call. We reproduce the
key intermediate steps explicitly here using our already-computed,
calibrated components so the calibration from step 2 is reflected in the
final scores.\
"""

C5 = """\
app_df = fm[['pitcher', 'season']].copy().rename(columns={'pitcher': 'pitcher_id'})
app_df['archetype'] = _archetype_for(fm).values
for col in ['injury_prob_30d', 'expected_days_lost', 'hazard_rate']:
    app_df[col] = components_raw[col].values

# Aggregate per-appearance model outputs to pitcher-season level before
# blending and normalizing — mean=100 must hold at pitcher-season grain.
risk_df = (
    app_df.groupby(['pitcher_id', 'season', 'archetype'], observed=True)
    [['injury_prob_30d', 'expected_days_lost', 'hazard_rate']]
    .mean()
    .reset_index()
)

risk_df = compute_raw_risk_score(risk_df, weights=weights_used)

reference_df = build_normalization_reference(risk_df)

irp = pd.Series(index=risk_df.index, dtype=float)
for (season, archetype), group in risk_df.groupby(['season', 'archetype'], observed=True):
    irp.loc[group.index] = normalize_to_injury_risk_plus(
        group['raw_risk_score'], season=season, archetype=archetype, reference_df=reference_df)
risk_df['injury_risk_plus'] = irp
risk_df = compute_risk_percentile(risk_df)

print(f'Computed Injury Risk+ for {len(risk_df):,} pitcher-seasons across {len(seasons)} seasons.')
print()
print('Sanity check — mean Injury Risk+ should be exactly 100 within every season x archetype group:')
display(risk_df.groupby(['season', 'archetype'], observed=True)['injury_risk_plus'].agg(['mean', 'std', 'count']).round(1))\
"""

C6_MD = """\
## 5. Score Distribution and Top-Risk Pitchers

A look at the overall score distribution (should center on 100 by
construction) and the highest-risk pitchers in the most recent season —
these are the names a team's medical/performance staff would want to review
first.\
"""

C6 = """\
fig, ax = plt.subplots(figsize=(8, 5))
for archetype, color in [('starter', '#C44E52'), ('reliever', '#4C72B0')]:
    sub = risk_df.loc[risk_df['archetype'] == archetype, 'injury_risk_plus']
    ax.hist(sub, bins=30, alpha=0.6, label=f'{archetype} (n={len(sub)})', color=color)
ax.axvline(100, color='black', linestyle='--', linewidth=1, label='League average (100)')
ax.set_xlabel('Injury Risk+')
ax.set_ylabel('Count')
ax.set_title('Injury Risk+ distribution by archetype')
ax.legend()
fig.tight_layout()
fig_path = FIGURES_DIR / 'fig_24_injury_risk_plus_distribution.png'
fig.savefig(fig_path, dpi=120, bbox_inches='tight')
plt.show()
print(f'Saved {fig_path}')

print(f'\\nTop 15 highest Injury Risk+ pitchers — season {calib_season}:')
top_risk = get_top_risk_pitchers(risk_df, season=calib_season, n=15)
display(top_risk[['pitcher_id', 'archetype', 'injury_prob_30d', 'expected_days_lost',
                  'hazard_rate', 'raw_risk_score', 'injury_risk_plus', 'risk_percentile']]
        .round(2))\
"""

C7_MD = """\
## 6. Season-Over-Season Stability

A useful score should be reasonably stable for the same pitcher across
adjacent seasons (injury risk is driven by durable factors like mechanics,
workload history, and age — not random noise). We check the
year-over-year correlation of Injury Risk+ for pitchers present in
consecutive seasons as a face-validity check.\
"""

C7 = """\
stability_rows = []
for s_prev, s_next in zip(seasons[:-1], seasons[1:]):
    prev = risk_df.loc[risk_df['season'] == s_prev, ['pitcher_id', 'injury_risk_plus']]
    nxt  = risk_df.loc[risk_df['season'] == s_next, ['pitcher_id', 'injury_risk_plus']]
    merged = prev.merge(nxt, on='pitcher_id', suffixes=('_prev', '_next'))
    if len(merged) < 5:
        continue
    corr = merged['injury_risk_plus_prev'].corr(merged['injury_risk_plus_next'])
    stability_rows.append({'season_pair': f'{s_prev} -> {s_next}', 'n_pitchers': len(merged), 'correlation': corr})

stability_df = pd.DataFrame(stability_rows)
if len(stability_df):
    display(stability_df.style.format({'correlation': '{:.3f}'}))
    print(f'\\nMean year-over-year correlation: {stability_df["correlation"].mean():.3f}')
else:
    print('Not enough overlapping pitchers across seasons to assess stability '
          '(expected when only one season of full data is available).')\
"""

C8_MD = """\
## 7. Save Final Outputs

Persist the full scored table, the top-risk leaderboard, the normalization
reference (needed to score future seasons consistently), and the blend-weight
comparison — these are the artifacts a downstream dashboard or report would
consume directly.\
"""

C8 = """\
scores_path = Path('data/processed/injury_risk_plus_scores.parquet')
scores_path.parent.mkdir(parents=True, exist_ok=True)
risk_df.to_parquet(scores_path, index=False)
print(f'Saved {scores_path}')

reference_df.to_csv(TABLES_DIR / 'injury_risk_plus_normalization_reference.csv', index=False)
print(f'Saved {TABLES_DIR / "injury_risk_plus_normalization_reference.csv"}')

weight_compare.reset_index().rename(columns={'index': 'component'}).to_csv(
    TABLES_DIR / 'injury_risk_plus_blend_weights.csv', index=False)
print(f'Saved {TABLES_DIR / "injury_risk_plus_blend_weights.csv"}')

for season in seasons:
    top_n = get_top_risk_pitchers(risk_df, season=season, n=25)
    out_path = TABLES_DIR / f'injury_risk_plus_top25_{season}.csv'
    top_n.to_csv(out_path, index=False)
print(f'Saved per-season top-25 leaderboards to {TABLES_DIR}/injury_risk_plus_top25_<season>.csv')\
"""

C9_MD = """\
## 8. Summary

* **Calibration:** section 2 shows whether isotonic calibration meaningfully
  tightens `injury_prob_30d` — if so, the production score uses the
  calibrated probabilities rather than raw model outputs.
* **Blend weights:** section 3 compares the design-doc defaults
  (0.50 / 0.30 / 0.20) to empirically optimized weights. We retain the
  documented defaults for the production score (stability and
  interpretability over marginal fit), but report the optimized values for
  future design review.
* **Score construction:** section 4 confirms the normalization step achieves
  its core design goal — mean Injury Risk+ = 100 (exactly) within every
  season × archetype group, making scores comparable across eras and roles.
* **Face validity:** sections 5–6 surface the highest-risk pitchers and check
  that scores are reasonably stable year-over-year, consistent with injury
  risk being driven by durable factors rather than noise.
* **Deliverables:** the full scored table
  (`data/processed/injury_risk_plus_scores.parquet`, one row per
  pitcher-season), normalization reference, blend-weight comparison, and
  per-season leaderboards are saved to `reports/tables/` for downstream
  consumption.

This completes the modeling pipeline through Notebook 9, per
`claude_instructions.md`.\
"""

C9 = """\
provenance = {
    'notebook': '09_risk_score_construction',
    'run_at': datetime.now(timezone.utc).isoformat(),
    'seasons_used': seasons,
    'n_pitcher_seasons_scored': int(len(risk_df)),
    'multitask_architecture': best_architecture,
    'survival_model': best_survival_name,
    'calibration_season': int(calib_season),
    'calibration_metrics_raw': raw_calib_metrics,
    'calibration_metrics_calibrated': new_calib_metrics,
    'weights_used': weights_used,
    'optimized_weights': optimized_weights,
    'mean_injury_risk_plus_by_group': (
        risk_df.groupby(['season', 'archetype'], observed=True)['injury_risk_plus']
        .mean().round(2).reset_index().to_dict(orient='records')
    ),
}
print(json.dumps(provenance, indent=2, default=str))

prov_path = TABLES_DIR / 'injury_risk_plus_provenance.json'
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
    md(C3_MD, "c03-calib-header"),
    code(C3,  "c03-calib"),
    md(C4_MD, "c04-weights-header"),
    code(C4,  "c04-weights"),
    md(C5_MD, "c05-score-header"),
    code(C5,  "c05-score"),
    md(C6_MD, "c06-toprisk-header"),
    code(C6,  "c06-toprisk"),
    md(C7_MD, "c07-stability-header"),
    code(C7,  "c07-stability"),
    md(C8_MD, "c08-save-header"),
    code(C8,  "c08-save"),
    md(C9_MD, "c09-summary-header"),
    code(C9,  "c09-provenance"),
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

out = Path("notebooks/09_risk_score_construction.ipynb")
out.parent.mkdir(exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Written: {out}  ({len(cells)} cells)")
