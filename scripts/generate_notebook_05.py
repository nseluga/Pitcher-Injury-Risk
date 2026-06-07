"""
Generate notebooks/05_feature_engineering.ipynb programmatically.
Run from project root: python scripts/generate_notebook_05.py
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
# 05 — Feature Engineering

## Purpose
Transform raw Statcast pitch data + injury database into a modeling-ready
feature matrix at the **pitcher × game_date** granularity.

This notebook implements the full feature pipeline:
1. Aggregate pitch-level data → pitcher-game stats
2. Compute rolling workload features (ACWR, pitch counts)
3. Compute rolling velocity features (trends, decline metrics)
4. Compute pitch mix features (usage rates, deltas)
5. Compute mechanics features (release drift, spin trends)
6. Build injury history features (prior IL counts, recency)
7. Attach forward-looking injury labels (30 / 60 / 90 days)
8. Assemble and validate the master feature matrix
9. Create temporal train/test split
10. Save to `data/processed/feature_matrix.parquet`

## Output schema
Each row = one pitcher × one game appearance.
Features = computed from data *before* that game (no future leakage).
Labels = computed from injury data *after* that game.\
"""

C1 = """\
import importlib, json, sys, warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 60)
pd.set_option('display.float_format', '{:.4f}'.format)

PROJECT_ROOT = str(Path('.').resolve())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Reload feature modules on every run so edits take effect without restart.
import src.features.workload_features     as _wl
import src.features.velocity_features     as _vf
import src.features.pitch_mix_features    as _pm
import src.features.movement_features     as _mv
import src.features.injury_history_features as _ih
for mod in [_wl, _vf, _pm, _mv, _ih]:
    importlib.reload(mod)

from src.features.workload_features       import build_workload_features
from src.features.velocity_features       import build_velocity_features
from src.features.pitch_mix_features      import build_pitch_mix_features
from src.features.movement_features       import build_movement_features
from src.features.injury_history_features import build_injury_history_features

PROCESSED_DIR = Path('data/processed')
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print('Feature modules loaded.')
print('Output →', PROCESSED_DIR.resolve())\
"""

C2 = """\
# ── Load cleaned source data ──────────────────────────────────────────────────
# Load all season Statcast files (full pull or test slice, whichever is present).
sc_files = sorted(Path('data/raw/statcast').glob('*.parquet'))
if not sc_files:
    raise FileNotFoundError(
        'No Statcast parquet files found under data/raw/statcast/. '
        'Run notebook 01 first.'
    )

sc_chunks = []
for f in sc_files:
    chunk = pd.read_parquet(f)
    chunk['game_date'] = pd.to_datetime(chunk['game_date'])
    sc_chunks.append(chunk)
sc = pd.concat(sc_chunks, ignore_index=True)

inj  = pd.read_parquet('data/processed/injuries_clean.parquet')
meta = pd.read_parquet('data/processed/player_metadata_clean.parquet')

inj['transaction_date'] = pd.to_datetime(inj['transaction_date'])
meta['birth_date']      = pd.to_datetime(meta['birth_date'])

n_seasons = sc['game_date'].dt.year.nunique()
print(f'Statcast   : {len(sc):>8,} pitches  '
      f'| {n_seasons} season(s) '
      f'| {sc["pitcher"].nunique()} pitchers')
print(f'Injuries   : {len(inj):>8,} stints   | {inj["player_id"].nunique()} pitchers')
print(f'Metadata   : {len(meta):>8,} pitchers')
print()
if n_seasons < 3:
    print('NOTE: Fewer than 3 seasons of data detected.')
    print('Rolling features require multi-month history for signal.')
    print('Re-run after completing the full 2015-2024 data pull.')\
"""

C3_MD = """\
---
## 1 — Pitch-Level → Game-Level Aggregation

Aggregate each pitch to the pitcher × game_date grain, computing:
- **Pitch count** — total pitches thrown per outing
- **Game-level velocity stats** — passed to `build_velocity_features`
- **Pitch mix rates** — passed to `build_pitch_mix_features`
- **Movement stats** — passed to `build_movement_features`\
"""

C3 = """\
# ── Build pitch mix features (game-level usage rates) ────────────────────────
print('Building pitch mix features...')
game_mix = build_pitch_mix_features(sc)
print(f'  Rows: {len(game_mix):,}  |  Cols: {game_mix.shape[1]}')
print(f'  Columns: {game_mix.columns.tolist()}')
print()\
"""

C4 = """\
# ── Build velocity features ───────────────────────────────────────────────────
print('Building velocity features...')
game_velo = build_velocity_features(sc)
print(f'  Rows: {len(game_velo):,}  |  Cols: {game_velo.shape[1]}')
print(f'  Columns: {game_velo.columns.tolist()}')
print()\
"""

C5 = """\
# ── Build movement / mechanics features ──────────────────────────────────────
print('Building movement features...')
game_mov = build_movement_features(sc)
print(f'  Rows: {len(game_mov):,}  |  Cols: {game_mov.shape[1]}')
print(f'  Columns: {game_mov.columns.tolist()}')
print()\
"""

C6_MD = """\
---
## 2 — Workload Features

Rolling workload is computed from the pitch-count base produced above.
Key outputs:
- `pitches_7d`, `pitches_28d`, `pitches_90d` — rolling pitch totals
- `acwr_7_28` — acute:chronic workload ratio
- `days_rest` — days since last appearance
- `pitches_season_to_date` — cumulative season workload\
"""

C6 = """\
# ── Build workload features (needs pitch_count from game_mix) ─────────────────
print('Building workload features...')
game_workload = build_workload_features(
    game_mix[['pitcher', 'game_date', 'pitch_count']].drop_duplicates()
)
print(f'  Rows: {len(game_workload):,}  |  Cols: {game_workload.shape[1]}')
print()
print('ACWR distribution:')
print(game_workload['acwr_7_28'].describe().round(3).to_string())
print()
print('Days rest distribution:')
print(game_workload['days_rest'].describe().round(1).to_string())\
"""

C7_MD = """\
---
## 3 — Injury History Features

For each pitcher-game observation, we look back at all prior IL stints
and compute:
- `prior_il_total` — total IL stints in career to date
- `prior_il_{type}` — stints by injury category (elbow, shoulder, etc.)
- `days_since_last_injury` — recency of most recent IL stint
- `prior_il_days_lost` — severity of most recent stint

These features use the injury database with a strict historical cutoff:
only stints with `transaction_date < game_date` are counted.\
"""

C7 = """\
# ── Build injury history features ─────────────────────────────────────────────
# Use pitcher-game dates from the workload frame as the base.
base_games = game_workload[['pitcher', 'game_date']].drop_duplicates()

# Rename player_id -> pitcher for the join
inj_renamed = inj.rename(columns={'player_id': 'pitcher'})

print('Building injury history features...')
game_history = build_injury_history_features(base_games, inj_renamed)
print(f'  Rows: {len(game_history):,}  |  Cols: {game_history.shape[1]}')
print()
print('Prior IL stints distribution:')
print(game_history['prior_il_total'].value_counts().sort_index().head(10).to_string())\
"""

C8_MD = """\
---
## 4 — Injury Labels (Forward-Looking)

For each pitcher-game observation, we look *forward* in the injury database
and compute binary labels:
- `injured_next_30d` — IL placement within 30 days
- `injured_next_60d` — IL placement within 60 days
- `injured_next_90d` — IL placement within 90 days
- `days_to_next_injury` — exact days until next IL placement (NaN = no injury)

**Critical design rule:** labels use only IL placements that occur
*strictly after* the game date. No current-game injury data is included.

These binary labels are used for classification models. The raw
`days_to_next_injury` column supports survival analysis and regression.\
"""

C8 = """\
# ── Build forward-looking injury labels ───────────────────────────────────────
print('Building injury labels...')

# All pitcher-game observations
base = game_workload[['pitcher', 'game_date']].drop_duplicates().copy()
base['game_date'] = pd.to_datetime(base['game_date'])

# Cross-join each pitcher-game with all injuries for that pitcher
inj_r = inj.rename(columns={'player_id': 'pitcher'})
merged = base.merge(
    inj_r[['pitcher', 'transaction_date', 'injury_type', 'days_lost']],
    on='pitcher', how='left'
)
# Keep only future injuries (strictly after game_date)
merged['days_ahead'] = (merged['transaction_date'] - merged['game_date']).dt.days
merged = merged[merged['days_ahead'] > 0]

# Minimum days ahead per pitcher-game (next injury)
next_inj = (
    merged.sort_values('days_ahead')
    .groupby(['pitcher', 'game_date'])
    .first()
    .reset_index()[['pitcher', 'game_date', 'days_ahead', 'injury_type', 'days_lost']]
)
next_inj.columns = ['pitcher', 'game_date', 'days_to_next_injury',
                     'next_injury_type', 'next_injury_days_lost']

labels = base.merge(next_inj, on=['pitcher', 'game_date'], how='left')
labels['days_to_next_injury'] = labels['days_to_next_injury'].fillna(9999).astype(int)

for window in [30, 60, 90]:
    labels[f'injured_next_{window}d'] = (
        labels['days_to_next_injury'] <= window
    ).astype(int)

print(f'  Rows: {len(labels):,}')
print()
print('Label prevalence (% of pitcher-game rows):')
for w in [30, 60, 90]:
    col  = f'injured_next_{w}d'
    rate = labels[col].mean()
    n    = labels[col].sum()
    print(f'  injured_next_{w}d : {n:>5,} positive  ({rate:.1%})')\
"""

C9_MD = """\
---
## 5 — Assemble Master Feature Matrix

Join all feature groups on `(pitcher, game_date)` and attach player
metadata (age at game date). This is the master table consumed by all
downstream modeling notebooks.\
"""

C9 = """\
# ── Join all feature groups ───────────────────────────────────────────────────
print('Assembling feature matrix...')

# Start with workload (has pitch_count + rolling workload cols)
fm = game_workload.copy()

# Velocity features
velo_cols = [c for c in game_velo.columns
             if c not in ['pitcher', 'game_date']]
fm = fm.merge(game_velo[['pitcher', 'game_date'] + velo_cols],
              on=['pitcher', 'game_date'], how='left')

# Pitch mix features (drop pitch_count — already in workload)
mix_cols = [c for c in game_mix.columns
            if c not in ['pitcher', 'game_date', 'pitch_count']]
fm = fm.merge(game_mix[['pitcher', 'game_date'] + mix_cols],
              on=['pitcher', 'game_date'], how='left')

# Movement features
mov_cols = [c for c in game_mov.columns
            if c not in ['pitcher', 'game_date']]
fm = fm.merge(game_mov[['pitcher', 'game_date'] + mov_cols],
              on=['pitcher', 'game_date'], how='left')

# Injury history features
hist_cols = [c for c in game_history.columns
             if c not in ['pitcher', 'game_date']]
fm = fm.merge(game_history[['pitcher', 'game_date'] + hist_cols],
              on=['pitcher', 'game_date'], how='left')

# Injury labels
label_cols = [c for c in labels.columns if c not in ['pitcher', 'game_date']]
fm = fm.merge(labels[['pitcher', 'game_date'] + label_cols],
              on=['pitcher', 'game_date'], how='left')

# Player metadata: age, name, handedness
meta_join = meta[['player_id', 'birth_date', 'player_name']].rename(
    columns={'player_id': 'pitcher'}
)
fm = fm.merge(meta_join, on='pitcher', how='left')
fm['age'] = ((fm['game_date'] - fm['birth_date']).dt.days / 365.25).round(2)

# Season
fm['season'] = fm['game_date'].dt.year

# Sort
fm = fm.sort_values(['pitcher', 'game_date']).reset_index(drop=True)

print(f'Feature matrix: {fm.shape[0]:,} rows × {fm.shape[1]} columns')
print(f'Unique pitchers: {fm["pitcher"].nunique():,}')
print(f'Date range: {fm["game_date"].min().date()} – {fm["game_date"].max().date()}')
print(f'Seasons: {sorted(fm["season"].unique().tolist())}')\
"""

C10 = """\
# ── Feature matrix quality checks ────────────────────────────────────────────
print('=== QUALITY CHECKS ===')
print()

# 1. Duplicate rows
n_dups = fm.duplicated(subset=['pitcher', 'game_date']).sum()
print(f'Duplicate (pitcher, game_date) rows: {n_dups}')

# 2. Null rates for key feature columns
key_features = [
    'pitch_count', 'pitches_7d', 'pitches_28d', 'acwr_7_28', 'days_rest',
    'fb_velo_mean', 'fb_velo_30d_avg', 'velo_change_7_30d',
    'fb_pct', 'sl_pct', 'spin_rate_mean', 'release_drift_30d',
    'prior_il_total', 'days_since_last_injury', 'age',
    'injured_next_30d', 'injured_next_60d', 'injured_next_90d',
]
present_feats = [c for c in key_features if c in fm.columns]
null_rates = fm[present_feats].isna().mean().sort_values(ascending=False)
print('Null rates (key features):')
for col, rate in null_rates.items():
    flag = '  ← review' if rate > 0.30 else ''
    print(f'  {col:<35} {rate:.1%}{flag}')

print()

# 3. Label sanity check
for w in [30, 60, 90]:
    col = f'injured_next_{w}d'
    if col in fm.columns:
        rate = fm[col].mean()
        print(f'{col}: {fm[col].sum()} positives ({rate:.1%})')\
"""

C11_MD = """\
---
## 6 — Temporal Train / Test Split

The train/test split **must be temporal** — never random — to prevent
future-data leakage into training. We reserve the most recent seasons
for evaluation.

Split strategy:
- **Train**: 2015–2021 (7 seasons)
- **Validation**: 2022 (1 season, for hyperparameter tuning)
- **Test**: 2023–2024 (2 seasons, held out until final model evaluation)

With TEST_MODE (1 season), all data goes to a single split and a note
is displayed.\
"""

C11 = """\
# ── Temporal train / test split ───────────────────────────────────────────────
available_seasons = sorted(fm['season'].unique())
n_seasons = len(available_seasons)

if n_seasons >= 5:
    # Full split: train 2015-2021, val 2022, test 2023-2024
    TRAIN_CUTOFF = 2022
    VAL_CUTOFF   = 2023

    train = fm[fm['season'] < TRAIN_CUTOFF]
    val   = fm[(fm['season'] >= TRAIN_CUTOFF) & (fm['season'] < VAL_CUTOFF)]
    test  = fm[fm['season'] >= VAL_CUTOFF]

    print(f'Train: seasons {available_seasons[0]}–{TRAIN_CUTOFF-1}'
          f'  →  {len(train):,} rows')
    print(f'Val  : season  {TRAIN_CUTOFF}  '
          f'→  {len(val):,} rows')
    print(f'Test : seasons {VAL_CUTOFF}+  '
          f'→  {len(test):,} rows')

elif n_seasons >= 2:
    # Small split: last season = test
    cutoff = available_seasons[-1]
    train  = fm[fm['season'] < cutoff]
    val    = pd.DataFrame()
    test   = fm[fm['season'] == cutoff]
    print(f'Train: {available_seasons[:-1]}  →  {len(train):,} rows')
    print(f'Test : {cutoff}              →  {len(test):,} rows')
    print('NOTE: No validation set with < 5 seasons.')

else:
    # TEST_MODE — single season, no meaningful split possible
    train = fm
    val   = pd.DataFrame()
    test  = pd.DataFrame()
    print('NOTE: Single season detected (TEST_MODE).')
    print('All rows placed in train set. Re-run with full data for proper split.')

print()
if len(train) > 0 and 'injured_next_30d' in train.columns:
    for split_name, split_df in [('Train', train), ('Val', val), ('Test', test)]:
        if len(split_df) > 0:
            rate = split_df['injured_next_30d'].mean()
            print(f'{split_name} 30d injury rate: {rate:.1%}')\
"""

C12 = """\
# ── Save feature matrix and splits ────────────────────────────────────────────
fm.to_parquet(PROCESSED_DIR / 'feature_matrix.parquet', index=False)
print(f'Saved feature_matrix.parquet  ({len(fm):,} rows, {fm.shape[1]} cols)')

if len(train) > 0:
    train.to_parquet(PROCESSED_DIR / 'feature_matrix_train.parquet', index=False)
    print(f'Saved feature_matrix_train.parquet  ({len(train):,} rows)')

if len(val) > 0:
    val.to_parquet(PROCESSED_DIR / 'feature_matrix_val.parquet', index=False)
    print(f'Saved feature_matrix_val.parquet  ({len(val):,} rows)')

if len(test) > 0:
    test.to_parquet(PROCESSED_DIR / 'feature_matrix_test.parquet', index=False)
    print(f'Saved feature_matrix_test.parquet  ({len(test):,} rows)')

# Feature column manifest (useful for modeling notebooks)
label_cols_set = {
    'injured_next_30d', 'injured_next_60d', 'injured_next_90d',
    'days_to_next_injury', 'next_injury_type', 'next_injury_days_lost',
}
meta_cols_set  = {'pitcher', 'game_date', 'season', 'player_name',
                   'birth_date', 'age'}
feature_cols   = [c for c in fm.columns
                  if c not in label_cols_set | meta_cols_set]

manifest = {
    'generated_at': datetime.now(timezone.utc).isoformat(),
    'total_rows':   len(fm),
    'total_cols':   fm.shape[1],
    'n_features':   len(feature_cols),
    'feature_cols': feature_cols,
    'label_cols':   list(label_cols_set & set(fm.columns)),
    'seasons':      sorted(fm['season'].unique().tolist()),
    'n_pitchers':   int(fm['pitcher'].nunique()),
}
manifest_path = PROCESSED_DIR / 'feature_manifest.json'
manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
print(f'Saved feature_manifest.json  ({len(feature_cols)} features)')
print()
print('NEXT STEP: Notebook 06 — Baseline Models')
print('  Input : data/processed/feature_matrix_train.parquet')
print('  Target: injured_next_30d (binary classification)')
print('  Models: Logistic Regression, Random Forest, XGBoost (baselines)')\
"""

# ---------------------------------------------------------------------------
# ASSEMBLE
# ---------------------------------------------------------------------------

cells = [
    md(C0,     "c00-header"),
    code(C1,   "c01-setup"),
    code(C2,   "c02-load"),
    md(C3_MD,  "c03-agg-header"),
    code(C3,   "c03-pitch-mix"),
    code(C4,   "c04-velocity"),
    code(C5,   "c05-movement"),
    md(C6_MD,  "c06-workload-header"),
    code(C6,   "c06-workload"),
    md(C7_MD,  "c07-history-header"),
    code(C7,   "c07-history"),
    md(C8_MD,  "c08-labels-header"),
    code(C8,   "c08-labels"),
    md(C9_MD,  "c09-assemble-header"),
    code(C9,   "c09-assemble"),
    code(C10,  "c10-quality"),
    md(C11_MD, "c11-split-header"),
    code(C11,  "c11-split"),
    code(C12,  "c12-save"),
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

out = Path("notebooks/05_feature_engineering.ipynb")
out.parent.mkdir(exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Written: {out}  ({len(cells)} cells)")
