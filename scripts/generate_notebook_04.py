"""
Generate notebooks/04_eda.ipynb programmatically.
Run from project root: python scripts/generate_notebook_04.py
"""

import json
from pathlib import Path

def md(source: str, cell_id: str) -> dict:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {},
            "source": source}

def code(source: str, cell_id: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "id": cell_id,
            "metadata": {}, "outputs": [], "source": source}


# ---------------------------------------------------------------------------
# CELL SOURCES
# ---------------------------------------------------------------------------

C0_HEADER = """\
# 04 — Exploratory Data Analysis

## Purpose
Determine whether predictive injury signal exists in the dataset and identify
the strongest candidate features for future Injury Risk+ modeling.

**Scope:** This notebook does **not** train any models. Every analysis is
purely descriptive or feature-engineering oriented.

## Data (TEST_MODE — June 1–7 2023)
| Dataset | Rows | Notes |
|---|---|---|
| `statcast_clean.parquet` | 25,613 pitches | 400 pitchers, 1-week window |
| `injuries_clean.parquet` | 244 stints | Full 2023 season |
| `player_metadata_clean.parquet` | 400 pitchers | With birth dates |

> **TEST_MODE note:** Rolling workload and velocity trend features require\
 months of history.
> Cells that need longitudinal data are marked ⚠️ and show the analytical\
 framework
> for when full season data is loaded.

## Sections
1. Dataset Overview
2. Injury Database Exploration
3. Class Balance Analysis
4. Demographic Risk Factors
5. Workload Analysis
6. Velocity Analysis
7. Pitch Mix Analysis
8. Pitch Movement and Mechanics Proxies
9. Correlation and Feature Relationships
10. Injury Countdown Analysis
11. Pitcher Archetypes
12. EDA Conclusions\
"""

C1_SETUP = """\
import warnings, json, sys, importlib
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_classif
from sklearn.impute import SimpleImputer

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 50)
pd.set_option('display.float_format', '{:.3f}'.format)

# ── Plotting style ────────────────────────────────────────────────────────────
CLR_HEALTHY  = '#2166ac'   # blue
CLR_INJURED  = '#d73027'   # red
CLR_NEUTRAL  = '#636363'
PALETTE_INJ  = [CLR_HEALTHY, CLR_INJURED]
PALETTE_INJTYPES = sns.color_palette('tab10', 13)
PALETTE_CLUSTER  = sns.color_palette('Set2', 8)

sns.set_theme(style='whitegrid', font_scale=1.1)
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

FIGURES_DIR   = Path('reports/figures')
PROCESSED_DIR = Path('data/processed')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print('Libraries loaded.')
print('Figures →', FIGURES_DIR.resolve())
print('Processed →', PROCESSED_DIR.resolve())\
"""

C2_LOAD = """\
# ── Load cleaned datasets ─────────────────────────────────────────────────────
sc   = pd.read_parquet('data/processed/statcast_clean.parquet')
inj  = pd.read_parquet('data/processed/injuries_clean.parquet')
meta = pd.read_parquet('data/processed/player_metadata_clean.parquet')

sc['game_date']            = pd.to_datetime(sc['game_date'])
inj['transaction_date']    = pd.to_datetime(inj['transaction_date'])
inj['activation_date']     = pd.to_datetime(inj['activation_date'])
meta['birth_date']         = pd.to_datetime(meta['birth_date'])

WINDOW_START = sc['game_date'].min()
WINDOW_END   = sc['game_date'].max()

print(f'Statcast  : {len(sc):>7,} pitches  '
      f'| {WINDOW_START.date()} – {WINDOW_END.date()} '
      f'| {sc["pitcher"].nunique()} pitchers')
print(f'Injuries  : {len(inj):>7,} stints   '
      f'| {inj["player_id"].nunique()} pitchers '
      f'| {inj["transaction_date"].dt.year.unique().tolist()}')
print(f'Metadata  : {len(meta):>7,} pitchers '
      f'| birth_date nulls: {meta["birth_date"].isna().sum()}')\
"""

C3_AGGREGATE = """\
# ── Pitcher × game aggregation ────────────────────────────────────────────────
FASTBALL  = {'FF', 'SI', 'FC'}
BREAKING  = {'SL', 'CU', 'KC', 'SV', 'ST'}
OFFSPEED  = {'CH', 'FS'}

pitcher_game = (
    sc.groupby(['pitcher', 'game_date'])
    .agg(
        pitch_count   = ('pitch_type',        'count'),
        avg_velo      = ('release_speed',      'mean'),
        max_velo      = ('release_speed',      'max'),
        std_velo      = ('release_speed',      'std'),
        avg_spin      = ('release_spin_rate',  'mean'),
        avg_extension = ('release_extension',  'mean'),
        avg_pfx_x     = ('pfx_x',              'mean'),
        avg_pfx_z     = ('pfx_z',              'mean'),
        avg_rel_x     = ('release_pos_x',      'mean'),
        avg_rel_z     = ('release_pos_z',      'mean'),
        std_rel_x     = ('release_pos_x',      'std'),
        std_rel_z     = ('release_pos_z',      'std'),
        fb_pct        = ('pitch_type', lambda x: x.isin(FASTBALL).mean()),
        breaking_pct  = ('pitch_type', lambda x: x.isin(BREAKING).mean()),
        offspeed_pct  = ('pitch_type', lambda x: x.isin(OFFSPEED).mean()),
        sl_pct        = ('pitch_type', lambda x: (x == 'SL').mean()),
        cu_pct        = ('pitch_type', lambda x: x.isin({'CU','KC','SV'}).mean()),
        ch_pct        = ('pitch_type', lambda x: (x == 'CH').mean()),
        sweep_pct     = ('pitch_type', lambda x: (x == 'ST').mean()),
    )
    .reset_index()
)
pitcher_game['pitcher'] = pitcher_game['pitcher'].astype(int)

# ── Pitcher-season summary ────────────────────────────────────────────────────
pitcher_season = (
    pitcher_game.groupby('pitcher')
    .agg(
        n_games        = ('game_date',     'nunique'),
        total_pitches  = ('pitch_count',   'sum'),
        avg_pitch_game = ('pitch_count',   'mean'),
        avg_velo       = ('avg_velo',       'mean'),
        max_velo       = ('max_velo',       'max'),
        std_velo       = ('std_velo',       'mean'),
        avg_spin       = ('avg_spin',       'mean'),
        avg_extension  = ('avg_extension', 'mean'),
        avg_pfx_x      = ('avg_pfx_x',     'mean'),
        avg_pfx_z      = ('avg_pfx_z',     'mean'),
        rel_x_std      = ('std_rel_x',     'mean'),
        rel_z_std      = ('std_rel_z',     'mean'),
        fb_pct         = ('fb_pct',        'mean'),
        breaking_pct   = ('breaking_pct',  'mean'),
        offspeed_pct   = ('offspeed_pct',  'mean'),
        sl_pct         = ('sl_pct',        'mean'),
        cu_pct         = ('cu_pct',        'mean'),
        ch_pct         = ('ch_pct',        'mean'),
        sweep_pct      = ('sweep_pct',     'mean'),
    )
    .reset_index()
)

print(f'Pitcher-game records : {len(pitcher_game):,}')
print(f'Unique pitchers      : {len(pitcher_season):,}')
print(f'Avg games per pitcher: {pitcher_season["n_games"].mean():.1f}')\
"""

C4_LABELS = """\
# ── Injury labels ─────────────────────────────────────────────────────────────
# A pitcher is labeled 1 if they had an IL placement AFTER our Statcast window.
# This frames the question: do June characteristics predict future 2023 injury?
# Pitchers injured before June 1 who returned to pitch in this window are
# treated as healthy (they have recovered and are active).

future_inj = inj[inj['transaction_date'] > WINDOW_END]
injured_ids = set(future_inj['player_id'].astype(int).unique())

pitcher_season = pitcher_season.copy()
pitcher_season['injured_future'] = (
    pitcher_season['pitcher'].isin(injured_ids).astype(int)
)

# ── Join metadata ─────────────────────────────────────────────────────────────
pitcher_season = pitcher_season.merge(
    meta[['player_id', 'birth_date', 'player_name', 'name_first', 'name_last']]
        .rename(columns={'player_id': 'pitcher'}),
    on='pitcher', how='left'
)

# Age at the observation window end
pitcher_season['age'] = (
    (WINDOW_END - pitcher_season['birth_date']).dt.days / 365.25
)

# Velocity delta: pitcher avg vs league avg (how hard they throw relative to peers)
league_avg_velo = pitcher_season['avg_velo'].mean()
pitcher_season['velo_delta'] = pitcher_season['avg_velo'] - league_avg_velo

n_inj = pitcher_season['injured_future'].sum()
n_healthy = len(pitcher_season) - n_inj
print(f'Pitchers in window  : {len(pitcher_season)}')
print(f'  Future injured (1): {n_inj}  ({n_inj/len(pitcher_season):.1%})')
print(f'  No future injury (0): {n_healthy}  ({n_healthy/len(pitcher_season):.1%})')
print()
print(f'Injury label source: IL placements from '
      f'{WINDOW_END.date()} → {inj["transaction_date"].max().date()}')\
"""

C5_S1_HEADER = """\
---
## Section 1: Dataset Overview

We begin by characterizing the structure and quality of all three input datasets.
Understanding which variables are complete and which require imputation is essential
before building any features or models.

### Key questions
- What is the shape and coverage of each dataset?
- Which columns have high missing-value rates?
- Which variables are candidates for imputation vs. removal?\
"""

C6_OVERVIEW = """\
# ── Dataset dimensions and summary statistics ─────────────────────────────────

datasets = {
    'Statcast (pitch-level)':  sc,
    'Injury database':         inj,
    'Pitcher metadata':        meta,
}

for name, df in datasets.items():
    print(f'{"="*60}')
    print(f'{name}')
    print(f'  Shape      : {df.shape[0]:,} rows × {df.shape[1]} columns')
    num_cols = df.select_dtypes(include=np.number).columns
    print(f'  Numeric    : {len(num_cols)}')
    cat_cols = df.select_dtypes(exclude=np.number).columns
    print(f'  Non-numeric: {len(cat_cols)}')
    null_total = df.isna().sum().sum()
    null_pct   = null_total / df.size
    print(f'  Missing    : {null_total:,} cells ({null_pct:.1%})')

print()
print('Statcast numeric summary (key columns):')
key_cols = ['release_speed','release_spin_rate','release_extension',
            'pfx_x','pfx_z','plate_x','plate_z','spin_axis']
present  = [c for c in key_cols if c in sc.columns]
print(sc[present].describe().round(2).to_string())\
"""

C7_MISSING = """\
# ── Missing value analysis ────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Left: Statcast column null rates (top 30 most-null)
sc_nulls = (sc.isna().mean() * 100).sort_values(ascending=False).head(30)
colors_sc = [CLR_INJURED if v > 50 else CLR_NEUTRAL for v in sc_nulls.values]
axes[0].barh(sc_nulls.index[::-1], sc_nulls.values[::-1], color=colors_sc[::-1])
axes[0].axvline(50, color='gray', ls='--', lw=1, label='50% threshold')
axes[0].set_xlabel('Missing Rate (%)')
axes[0].set_title('Statcast: Top 30 Columns by Missing Rate', fontweight='bold')
axes[0].legend(fontsize=9)

# Right: Key modeling columns completeness
model_cols = [
    'release_speed', 'release_spin_rate', 'release_extension',
    'pfx_x', 'pfx_z', 'plate_x', 'plate_z', 'spin_axis',
    'effective_speed', 'release_pos_x', 'release_pos_z',
]
present_model = [c for c in model_cols if c in sc.columns]
completeness  = (1 - sc[present_model].isna().mean()) * 100
colors_model  = [CLR_HEALTHY if v >= 95 else (CLR_NEUTRAL if v >= 50 else CLR_INJURED)
                 for v in completeness.values]
axes[1].barh(completeness.index[::-1], completeness.values[::-1],
             color=colors_model[::-1])
axes[1].axvline(95, color=CLR_HEALTHY, ls='--', lw=1, label='>95% (acceptable)')
axes[1].set_xlabel('Completeness (%)')
axes[1].set_title('Key Modeling Columns: Completeness', fontweight='bold')
axes[1].legend(fontsize=9)
axes[1].set_xlim(0, 105)

plt.suptitle('Figure 1: Missing Value Analysis — Statcast', y=1.01,
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_01_missing_values.png')
plt.show()
print('Saved fig_01_missing_values.png')\
"""

C8_COMPLETENESS = """\
# ── Feature completeness table ────────────────────────────────────────────────
# Categorize columns by completeness tier for downstream imputation planning.

sc_null_rates = sc.isna().mean()

tiers = {
    'Complete (>99%)':         sc_null_rates[sc_null_rates < 0.01].index.tolist(),
    'Minor gaps (1–10%)':      sc_null_rates[(sc_null_rates >= 0.01) &
                                              (sc_null_rates < 0.10)].index.tolist(),
    'Moderate gaps (10–50%)':  sc_null_rates[(sc_null_rates >= 0.10) &
                                              (sc_null_rates < 0.50)].index.tolist(),
    'High missingness (>50%)': sc_null_rates[sc_null_rates >= 0.50].index.tolist(),
}

print('Feature Completeness Tiers (Statcast):')
print('─' * 65)
for tier, cols in tiers.items():
    print(f'{tier}: {len(cols)} columns')
    if cols:
        for c in cols[:8]:
            rate = sc_null_rates[c]
            print(f'    {c:<35} {(1-rate)*100:5.1f}% complete')
        if len(cols) > 8:
            print(f'    ... and {len(cols)-8} more')
    print()

print('─' * 65)
print('IMPUTATION PLAN:')
print('  • release_speed / pfx_x / pfx_z / spin_axis → mean imputation by pitch type')
print('  • hit-based columns (launch_speed, hc_x, etc.) → not applicable to pitch models')
print('  • High-missingness columns → exclude from feature matrix')\
"""

C9_S2_HEADER = """\
---
## Section 2: Injury Database Exploration

We analyze the structure of the 2023 pitcher injury database: frequency by type,
severity (days lost), and the shape of the days-lost distribution.

These distributions directly inform model design decisions:
- **Class weighting** depends on relative injury type frequencies
- **Regression target distribution** for days-lost prediction
- **Season-ending rate** affects censoring in survival models\
"""

C10_INJ_TYPES = """\
# ── Injury type distribution and severity ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 1) Injury type frequency
type_counts = inj['injury_type'].value_counts()
colors_bar  = PALETTE_INJTYPES[:len(type_counts)]
axes[0].barh(type_counts.index[::-1], type_counts.values[::-1],
             color=colors_bar[::-1])
axes[0].set_xlabel('Number of IL Stints')
axes[0].set_title('Injury Type Frequency (2023)', fontweight='bold')
for i, (v, _) in enumerate(zip(type_counts.values[::-1], type_counts.index[::-1])):
    axes[0].text(v + 0.3, i, str(v), va='center', fontsize=9)

# 2) Median days lost by injury type
type_severity = (
    inj[inj['days_lost'].notna()]
    .groupby('injury_type')['days_lost']
    .agg(['median', 'count'])
    .sort_values('median', ascending=True)
)
bar_colors = [CLR_INJURED if m > 30 else CLR_NEUTRAL for m in type_severity['median']]
axes[1].barh(type_severity.index, type_severity['median'], color=bar_colors)
axes[1].axvline(type_severity['median'].mean(), color='black',
                ls='--', lw=1.5, label=f'Mean = {type_severity["median"].mean():.0f}d')
axes[1].set_xlabel('Median Days Lost')
axes[1].set_title('Severity: Median Days Lost by Type', fontweight='bold')
axes[1].legend(fontsize=9)

# 3) Season-ending proportion by type
season_ending_rate = (
    inj.groupby('injury_type')['season_ending']
    .mean()
    .sort_values(ascending=True)
)
colors_se = [CLR_INJURED if v > 0.15 else CLR_NEUTRAL for v in season_ending_rate]
axes[2].barh(season_ending_rate.index, season_ending_rate.values * 100,
             color=colors_se)
axes[2].set_xlabel('Season-Ending Rate (%)')
axes[2].set_title('Season-Ending Rate by Injury Type', fontweight='bold')

plt.suptitle('Figure 2: Injury Type Distribution and Severity', y=1.01,
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_02_injury_type_distribution.png')
plt.show()
print('Saved fig_02_injury_type_distribution.png')\
"""

C11_DAYSLOST = """\
# ── Days-lost distribution ────────────────────────────────────────────────────
paired_inj = inj[inj['days_lost'].notna()].copy()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1) Histogram of days lost
axes[0].hist(paired_inj['days_lost'], bins=30, color=CLR_INJURED, alpha=0.7,
             edgecolor='white', linewidth=0.5)
axes[0].axvline(paired_inj['days_lost'].median(), color='black',
                ls='--', lw=2, label=f'Median = {paired_inj["days_lost"].median():.0f}d')
axes[0].axvline(10, color=CLR_NEUTRAL, ls=':', lw=1.5, label='10-day IL minimum')
axes[0].set_xlabel('Days Lost')
axes[0].set_ylabel('Count')
axes[0].set_title('Days Lost Distribution', fontweight='bold')
axes[0].legend(fontsize=9)

# 2) Boxplot by injury type (top 8 types only for readability)
top_types = inj['injury_type'].value_counts().head(8).index
data_box  = [paired_inj[paired_inj['injury_type'] == t]['days_lost'].values
             for t in top_types]
bp = axes[1].boxplot(data_box, vert=True, patch_artist=True,
                     labels=top_types, showfliers=True)
for patch, color in zip(bp['boxes'], PALETTE_INJTYPES[:8]):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
axes[1].set_xlabel('Injury Type')
axes[1].set_ylabel('Days Lost')
axes[1].set_title('Days Lost by Injury Type', fontweight='bold')
axes[1].tick_params(axis='x', rotation=45)

# 3) CDF of days lost
sorted_days = np.sort(paired_inj['days_lost'].values)
cdf = np.arange(1, len(sorted_days) + 1) / len(sorted_days)
axes[2].plot(sorted_days, cdf, color=CLR_INJURED, lw=2)
for threshold in [10, 15, 30, 60]:
    pct = (sorted_days <= threshold).mean()
    axes[2].axvline(threshold, color='gray', ls=':', lw=1)
    axes[2].text(threshold + 1, pct, f'{pct:.0%}\\n≤{threshold}d',
                 fontsize=8, color='gray')
axes[2].set_xlabel('Days Lost')
axes[2].set_ylabel('Cumulative Proportion')
axes[2].set_title('CDF: Days Lost', fontweight='bold')

plt.suptitle('Figure 3: Injury Severity — Days Lost Distribution', y=1.01,
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_03_injury_severity.png')
plt.show()

print(f'Days lost summary (n={len(paired_inj)}):')
print(paired_inj['days_lost'].describe().round(1).to_string())
print(f'Season-ending (no activation): {inj["season_ending"].sum()} stints '
      f'({inj["season_ending"].mean():.1%})')\
"""

C12_S3_HEADER = """\
---
## Section 3: Class Balance Analysis

Class imbalance is one of the most important practical challenges in injury
prediction. If only 10% of pitcher-seasons result in injury, a model that
predicts "healthy" every time achieves 90% accuracy while providing zero
value.

Understanding the degree of imbalance guides decisions about:
- **SMOTE / oversampling** during training
- **Class weights** in loss functions
- **Evaluation metrics**: precision-recall AUC is more informative than
  ROC-AUC under severe imbalance
- **Threshold tuning** to balance false positives vs. false negatives\
"""

C13_CLASSBALANCE = """\
# ── Class balance ─────────────────────────────────────────────────────────────
n_inj     = pitcher_season['injured_future'].sum()
n_healthy = len(pitcher_season) - n_inj
labels    = ['Healthy (0)', 'Future Injured (1)']
counts    = [n_healthy, n_inj]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 1) Bar chart
bars = axes[0].bar(labels, counts, color=PALETTE_INJ, width=0.5, edgecolor='white')
for bar, count in zip(bars, counts):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                 str(count), ha='center', va='bottom', fontweight='bold', fontsize=12)
axes[0].set_ylabel('Number of Pitchers')
axes[0].set_title('Pitcher Label Distribution\\n(June 1-7 window vs. rest of 2023)',
                  fontweight='bold')
axes[0].set_ylim(0, max(counts) * 1.15)

# Annotate imbalance ratio
ratio = n_healthy / n_inj
axes[0].text(0.5, 0.9, f'Imbalance ratio: {ratio:.1f}:1',
             transform=axes[0].transAxes, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# 2) Pie chart with annotation
wedges, texts, autotexts = axes[1].pie(
    counts, labels=labels, colors=PALETTE_INJ,
    autopct='%1.1f%%', startangle=90,
    wedgeprops=dict(edgecolor='white', linewidth=2)
)
for at in autotexts:
    at.set_fontsize(11)
    at.set_fontweight('bold')
axes[1].set_title(f'Class Balance\\n({len(pitcher_season)} pitchers total)',
                  fontweight='bold')

plt.suptitle('Figure 4: Class Balance Analysis', y=1.01,
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_04_class_balance.png')
plt.show()

print(f'Class distribution:')
print(f'  Healthy         : {n_healthy} ({n_healthy/len(pitcher_season):.1%})')
print(f'  Future injured  : {n_inj} ({n_inj/len(pitcher_season):.1%})')
print(f'  Imbalance ratio : {ratio:.1f}:1')
print()
print('MODELING IMPLICATIONS:')
print('  • At 37% injury rate, imbalance is moderate (not severe).')
print('  • Use class_weight="balanced" in classifiers as a baseline.')
print('  • Monitor both precision and recall; use PR-AUC as primary metric.')
print('  • With full multi-season data (2015–2024), injury rate drops to ~15-20%')
print('    per pitcher-season — imbalance will be more pronounced.')\
"""

C14_S4_HEADER = """\
---
## Section 4: Demographic Risk Factors

Age and accumulated workload are among the most well-studied injury risk
factors in sports medicine. In baseball, the literature consistently finds:

- **Age**: Pitchers 28–32 have relatively stable injury rates; rates increase
  sharply after 32 as tissue resilience declines
- **Experience**: High career innings pitched (TJ surgery predictor research
  by Langer et al.) correlates with elevated elbow injury risk
- **Role**: Starters pitch in longer outings at lower intensity; relievers
  pitch more frequently at maximum effort — distinct risk profiles

In TEST_MODE we can assess age from birth dates and use our 1-week pitch
count as a proxy for role (starters throw 70–110 pitches in a single outing;
relievers throw 10–35).\
"""

C15_AGE = """\
# ── Age analysis ─────────────────────────────────────────────────────────────
ps_age = pitcher_season[pitcher_season['age'].notna()].copy()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1) Age distribution overall
axes[0].hist(ps_age['age'], bins=20, color=CLR_NEUTRAL, alpha=0.75,
             edgecolor='white', linewidth=0.5)
axes[0].axvline(ps_age['age'].median(), color='black', ls='--', lw=2,
                label=f'Median = {ps_age["age"].median():.1f}')
axes[0].set_xlabel('Age (years)')
axes[0].set_ylabel('Count')
axes[0].set_title('Age Distribution (All Pitchers)', fontweight='bold')
axes[0].legend()

# 2) Age distribution by injury status
for label, group in ps_age.groupby('injured_future'):
    color = CLR_INJURED if label == 1 else CLR_HEALTHY
    name  = 'Future Injured' if label == 1 else 'Healthy'
    axes[1].hist(group['age'], bins=18, color=color, alpha=0.55,
                 edgecolor='white', linewidth=0.5, label=f'{name} (n={len(group)})')
axes[1].set_xlabel('Age (years)')
axes[1].set_ylabel('Count')
axes[1].set_title('Age Distribution by Injury Label', fontweight='bold')
axes[1].legend()

# 3) Injury rate by 2-year age bin
ps_age['age_bin'] = pd.cut(ps_age['age'], bins=range(20, 46, 2), right=False)
age_inj_rate = (
    ps_age.groupby('age_bin', observed=True)['injured_future']
    .agg(['mean', 'count'])
    .reset_index()
)
age_inj_rate['ci'] = 1.96 * np.sqrt(
    age_inj_rate['mean'] * (1 - age_inj_rate['mean']) / age_inj_rate['count'].clip(1)
)
age_labels = [str(b) for b in age_inj_rate['age_bin']]
axes[2].bar(range(len(age_inj_rate)), age_inj_rate['mean'] * 100,
            color=[CLR_INJURED if m > 0.40 else CLR_NEUTRAL
                   for m in age_inj_rate['mean']])
axes[2].errorbar(range(len(age_inj_rate)),
                 age_inj_rate['mean'] * 100,
                 yerr=age_inj_rate['ci'] * 100,
                 fmt='none', color='black', capsize=4)
axes[2].set_xticks(range(len(age_inj_rate)))
axes[2].set_xticklabels(age_labels, rotation=45, ha='right', fontsize=9)
axes[2].set_ylabel('Injury Rate (%)')
axes[2].set_title('Injury Rate by Age Group', fontweight='bold')
axes[2].axhline(ps_age['injured_future'].mean() * 100,
                color='gray', ls='--', lw=1, label='Overall rate')
axes[2].legend(fontsize=9)

plt.suptitle('Figure 5: Age and Injury Risk', y=1.01, fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_05_age_analysis.png')
plt.show()

# Statistical test
inj_ages     = ps_age[ps_age['injured_future'] == 1]['age']
healthy_ages = ps_age[ps_age['injured_future'] == 0]['age']
t_stat, p_val = stats.ttest_ind(inj_ages.dropna(), healthy_ages.dropna())
print(f'Age: Injured mean = {inj_ages.mean():.1f}  |  '
      f'Healthy mean = {healthy_ages.mean():.1f}')
print(f'Independent t-test: t = {t_stat:.2f}, p = {p_val:.3f}')\
"""

C16_ROLE = """\
# ── Role proxy: starters vs. relievers ───────────────────────────────────────
# Classify pitchers by their avg pitches per game appearance.
# Starter threshold: avg >= 60 pitches/game (not precise but directionally correct).

pitcher_season['role'] = np.where(
    pitcher_season['avg_pitch_game'] >= 60, 'Starter', 'Reliever'
)
role_counts = pitcher_season['role'].value_counts()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 1) Role distribution
role_counts.plot(kind='bar', color=[CLR_NEUTRAL, CLR_HEALTHY], ax=axes[0],
                 width=0.5, edgecolor='white')
axes[0].set_xlabel('Role (proxy)')
axes[0].set_ylabel('Count')
axes[0].set_title('Pitcher Role Distribution\\n(based on avg pitches/game)',
                  fontweight='bold')
axes[0].tick_params(axis='x', rotation=0)
for bar in axes[0].patches:
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 int(bar.get_height()), ha='center', va='bottom', fontweight='bold')

# 2) Injury rate by role
role_inj = pitcher_season.groupby('role')['injured_future'].mean() * 100
colors_role = [CLR_INJURED if v > pitcher_season['injured_future'].mean()*100
               else CLR_HEALTHY for v in role_inj.values]
role_inj.plot(kind='bar', color=colors_role, ax=axes[1], width=0.5, edgecolor='white')
axes[1].axhline(pitcher_season['injured_future'].mean() * 100,
                color='gray', ls='--', lw=1.5, label='Overall rate')
axes[1].set_ylabel('Future Injury Rate (%)')
axes[1].set_title('Injury Rate by Role', fontweight='bold')
axes[1].tick_params(axis='x', rotation=0)
axes[1].legend(fontsize=9)
for bar in axes[1].patches:
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f'{bar.get_height():.1f}%', ha='center', va='bottom', fontweight='bold')

plt.suptitle('Figure 6: Role and Injury Risk', y=1.01, fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_06_role_analysis.png')
plt.show()
print(role_inj.to_string())\
"""

C17_S5_HEADER = """\
---
## Section 5: Workload Analysis

Workload is the single most-studied modifiable injury risk factor in elite
pitching. The key constructs are:

| Metric | Definition | Significance |
|---|---|---|
| **Pitch count** | Raw count per outing | Direct tissue stress |
| **Innings pitched** | Pitch count proxy | Common front-office metric |
| **ACWR** | Acute:chronic workload ratio | Workload spike detector |
| **Days rest** | Days between appearances | Recovery quality |

The **acute:chronic workload ratio (ACWR)** is derived from athlete workload
research (Gabbett, 2016). An ACWR > 1.3 is associated with meaningfully
higher injury risk across multiple sports.

`ACWR = (pitches in last 7 days) / (avg weekly pitches over last 28 days)`

> ⚠️ **TEST_MODE:** With 1 week of data, chronic baseline cannot be computed.
> The framework and acute metrics are shown below. Full ACWR requires a
> rolling 28-day window, available with full season data.\
"""

C18_WORKLOAD = """\
# ── Workload distributions ────────────────────────────────────────────────────
ps = pitcher_season.copy()

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1) Total pitches in observation window (proxy: acute workload)
axes[0,0].hist(ps['total_pitches'], bins=25, color=CLR_NEUTRAL, alpha=0.75,
               edgecolor='white')
axes[0,0].axvline(ps['total_pitches'].median(), color='black', ls='--', lw=2,
                  label=f'Median = {ps["total_pitches"].median():.0f}')
axes[0,0].set_xlabel('Total Pitches (June 1-7)')
axes[0,0].set_ylabel('Count')
axes[0,0].set_title('Total Pitches in Observation Window', fontweight='bold')
axes[0,0].legend()

# 2) Avg pitches per game
axes[0,1].hist(ps['avg_pitch_game'], bins=25, color=CLR_NEUTRAL, alpha=0.75,
               edgecolor='white')
axes[0,1].axvline(ps['avg_pitch_game'].median(), color='black', ls='--', lw=2,
                  label=f'Median = {ps["avg_pitch_game"].median():.0f}')
axes[0,1].set_xlabel('Avg Pitches per Game')
axes[0,1].set_ylabel('Count')
axes[0,1].set_title('Avg Pitches per Outing', fontweight='bold')
axes[0,1].legend()

# 3) Workload by injury label (boxplot)
for i, (label, group) in enumerate(ps.groupby('injured_future')):
    color = CLR_INJURED if label == 1 else CLR_HEALTHY
    name  = 'Injured' if label == 1 else 'Healthy'
    bp = axes[1,0].boxplot(group['total_pitches'].values,
                           positions=[i], widths=0.5,
                           patch_artist=True, showfliers=True)
    bp['boxes'][0].set_facecolor(color)
    bp['boxes'][0].set_alpha(0.7)
    axes[1,0].text(i, group['total_pitches'].max() + 2,
                   f'n={len(group)}\\nm={group["total_pitches"].median():.0f}',
                   ha='center', fontsize=9)
axes[1,0].set_xticks([0, 1])
axes[1,0].set_xticklabels(['Healthy', 'Future Injured'])
axes[1,0].set_ylabel('Total Pitches (June 1-7)')
axes[1,0].set_title('Workload by Injury Status', fontweight='bold')

# 4) Number of appearances
for i, (label, group) in enumerate(ps.groupby('injured_future')):
    color = CLR_INJURED if label == 1 else CLR_HEALTHY
    bp = axes[1,1].boxplot(group['n_games'].values,
                           positions=[i], widths=0.5,
                           patch_artist=True, showfliers=True)
    bp['boxes'][0].set_facecolor(color)
    bp['boxes'][0].set_alpha(0.7)
axes[1,1].set_xticks([0, 1])
axes[1,1].set_xticklabels(['Healthy', 'Future Injured'])
axes[1,1].set_ylabel('Appearances in Window')
axes[1,1].set_title('Number of Appearances by Injury Status', fontweight='bold')

plt.suptitle('Figure 7: Workload Analysis', y=1.01, fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_07_workload_analysis.png')
plt.show()

# Statistical test: do injured pitchers throw more?
t, p = stats.ttest_ind(
    ps[ps['injured_future']==1]['total_pitches'].dropna(),
    ps[ps['injured_future']==0]['total_pitches'].dropna()
)
print(f'Total pitches:  Injured mean = '
      f'{ps[ps["injured_future"]==1]["total_pitches"].mean():.1f}  |  '
      f'Healthy mean = {ps[ps["injured_future"]==0]["total_pitches"].mean():.1f}')
print(f't-test: t = {t:.2f}, p = {p:.3f}')\
"""

C19_ACWR = """\
# ── ACWR framework (limited by TEST_MODE window) ─────────────────────────────
# With only 1 week of data, we cannot compute the chronic (28-day avg) baseline.
# We define ACWR conceptually and compute the acute component.
# Full ACWR will be available in notebook 05 (feature engineering) once
# full season Statcast data is loaded.

# Acute load proxy: pitches in the 7-day window (= our entire dataset)
pitcher_season['acute_load'] = pitcher_season['total_pitches']

# ACWR formula: ACWR = acute_load / chronic_load
# chronic_load = rolling 28-day avg weekly pitch count (requires history)
# With 1 week of data: ACWR = 1.0 for all pitchers (degenerate case)
# We flag this clearly and show what the distribution will look like.

print('=' * 60)
print('ACWR FRAMEWORK')
print('=' * 60)
print('Acute load (7-day pitches):')
print(pitcher_season['acute_load'].describe().round(1).to_string())
print()
print('Chronic baseline: UNAVAILABLE in TEST_MODE.')
print('  Requires 28+ days of pitch history per pitcher.')
print('  Will be computed in notebook 05 (Feature Engineering)')
print('  using full 2023 season Statcast data.')
print()
print('Expected ACWR distribution with full data:')
print('  • ACWR < 0.8 : underloaded / returning from rest')
print('  • ACWR 0.8–1.3 : typical in-season range (lower risk)')
print('  • ACWR > 1.3 : workload spike zone (elevated injury risk)')
print('  • ACWR > 1.5 : danger zone (multiple studies)')
print()

# Visualize the acute load distribution as a standalone signal
fig, ax = plt.subplots(figsize=(9, 5))
for label, group in pitcher_season.groupby('injured_future'):
    color = CLR_INJURED if label == 1 else CLR_HEALTHY
    name  = f'Future Injured (n={len(group)})' if label == 1 else f'Healthy (n={len(group)})'
    ax.hist(group['acute_load'], bins=20, color=color, alpha=0.5,
            edgecolor='white', label=name)
ax.set_xlabel('Acute Pitch Load (7-day window)')
ax.set_ylabel('Count')
ax.set_title('Acute Workload by Future Injury Status\\n'
             '(Full ACWR requires chronic baseline — see Notebook 05)',
             fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_08_acute_workload.png')
plt.show()\
"""

C20_S6_HEADER = """\
---
## Section 6: Velocity Analysis

Fastball velocity decline is one of the most clinically actionable pre-injury
signals in baseball analytics. The mechanism is well-documented:

1. Fatigue or compensatory mechanics → reduced neuromuscular recruitment
2. Subtle tissue breakdown → velocity decline days before overt injury
3. Pitchers may consciously or unconsciously protect the arm

Key metrics to track:
- `avg_velo`: mean pitch velocity in the window
- `velo_delta`: pitcher's velocity vs. league average (controls for natural variation)
- `velocity_change_30d / 60d / 90d`: rolling velocity decline (not computable in TEST_MODE)
- `velo_variability`: within-game velocity std (fatigue proxy)

> ⚠️ **TEST_MODE:** Longitudinal velocity decline metrics (`velocity_change_Nd`)
> require multi-week Statcast history. The cross-sectional comparison below
> tests whether pitchers who are *about to be injured* had different velocity
> characteristics in June 1-7.\
"""

C21_VELO = """\
# ── Velocity distributions ────────────────────────────────────────────────────
# Filter to pitchers with at least 10 fastball pitches for reliable velocity estimates
FASTBALL_TYPES = {'FF', 'SI', 'FC'}
fb_only = sc[sc['pitch_type'].isin(FASTBALL_TYPES)]
fb_by_pitcher = fb_only.groupby('pitcher')['release_speed'].agg(['mean', 'std', 'count'])
fb_by_pitcher = fb_by_pitcher[fb_by_pitcher['count'] >= 10].copy()
fb_by_pitcher.columns = ['fb_avg', 'fb_std', 'fb_count']
fb_by_pitcher = fb_by_pitcher.reset_index()

# Merge into pitcher_season
ps_velo = pitcher_season.merge(fb_by_pitcher, on='pitcher', how='left')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1) Fastball velocity distribution
axes[0].hist(ps_velo['fb_avg'].dropna(), bins=25, color=CLR_NEUTRAL, alpha=0.75,
             edgecolor='white')
axes[0].axvline(ps_velo['fb_avg'].mean(), color='black', ls='--', lw=2,
                label=f'Mean = {ps_velo["fb_avg"].mean():.1f} mph')
axes[0].set_xlabel('Avg Fastball Velocity (mph)')
axes[0].set_ylabel('Count')
axes[0].set_title('Fastball Velocity Distribution', fontweight='bold')
axes[0].legend()

# 2) Velocity by injury status (KDE)
for label, group in ps_velo.groupby('injured_future'):
    color = CLR_INJURED if label == 1 else CLR_HEALTHY
    name  = 'Future Injured' if label == 1 else 'Healthy'
    data  = group['fb_avg'].dropna()
    if len(data) > 5:
        kde_x = np.linspace(data.min() - 2, data.max() + 2, 200)
        kde   = stats.gaussian_kde(data)
        axes[1].plot(kde_x, kde(kde_x), color=color, lw=2.5, label=name)
        axes[1].fill_between(kde_x, kde(kde_x), alpha=0.15, color=color)
        axes[1].axvline(data.mean(), color=color, ls='--', lw=1.5,
                        label=f'{name} mean: {data.mean():.1f}')
axes[1].set_xlabel('Avg Fastball Velocity (mph)')
axes[1].set_ylabel('Density')
axes[1].set_title('Fastball Velocity by Injury Status', fontweight='bold')
axes[1].legend(fontsize=9)

# 3) Velocity variability (std) by injury status
for label, group in ps_velo.groupby('injured_future'):
    color = CLR_INJURED if label == 1 else CLR_HEALTHY
    name  = 'Future Injured' if label == 1 else 'Healthy'
    data  = group['fb_std'].dropna()
    if len(data) > 5:
        kde_x = np.linspace(data.min() - 0.5, data.max() + 0.5, 200)
        kde   = stats.gaussian_kde(data)
        axes[2].plot(kde_x, kde(kde_x), color=color, lw=2.5, label=name)
        axes[2].fill_between(kde_x, kde(kde_x), alpha=0.15, color=color)
axes[2].set_xlabel('Fastball Velocity Std Dev (mph)')
axes[2].set_ylabel('Density')
axes[2].set_title('Velocity Variability by Injury Status', fontweight='bold')
axes[2].legend(fontsize=9)

plt.suptitle('Figure 9: Fastball Velocity Analysis', y=1.01,
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_09_velocity_analysis.png')
plt.show()

# Statistical comparison
t, p = stats.ttest_ind(
    ps_velo[ps_velo['injured_future']==1]['fb_avg'].dropna(),
    ps_velo[ps_velo['injured_future']==0]['fb_avg'].dropna()
)
print(f'Fastball velocity: Injured = '
      f'{ps_velo[ps_velo["injured_future"]==1]["fb_avg"].mean():.2f} mph  |  '
      f'Healthy = {ps_velo[ps_velo["injured_future"]==0]["fb_avg"].mean():.2f} mph')
print(f't-test: t = {t:.2f}, p = {p:.3f}')
pitcher_season['fb_avg'] = ps_velo.set_index('pitcher')['fb_avg'].reindex(
    pitcher_season['pitcher'].values).values\
"""

C22_VELO_FEATURES = """\
# ── Velocity delta features ───────────────────────────────────────────────────
# velocity_delta: pitcher's mean FB velo vs. league mean (normalizes for era)
# This cross-sectional feature captures whether higher-velocity pitchers
# carry higher injury risk — an important research question.

ps_velo_full = pitcher_season[pitcher_season['fb_avg'].notna()].copy()
ps_velo_full['velo_delta'] = ps_velo_full['fb_avg'] - ps_velo_full['fb_avg'].mean()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 1) Velocity delta distribution
axes[0].hist(ps_velo_full['velo_delta'], bins=25, color=CLR_NEUTRAL, alpha=0.75,
             edgecolor='white')
axes[0].axvline(0, color='black', ls='-', lw=2, label='League average')
axes[0].set_xlabel('Velocity Delta (mph vs. league mean)')
axes[0].set_ylabel('Count')
axes[0].set_title('Pitcher Velocity vs. League Average', fontweight='bold')
axes[0].legend()

# 2) Injury rate by velocity decile
ps_velo_full['velo_decile'] = pd.qcut(ps_velo_full['fb_avg'], q=5,
                                       labels=['Q1\\n(slowest)', 'Q2', 'Q3',
                                               'Q4', 'Q5\\n(fastest)'])
velo_inj = (ps_velo_full.groupby('velo_decile', observed=True)['injured_future']
            .agg(['mean', 'count']))
axes[1].bar(range(len(velo_inj)), velo_inj['mean'] * 100,
            color=[CLR_INJURED if m > ps_velo_full['injured_future'].mean()
                   else CLR_HEALTHY for m in velo_inj['mean']])
axes[1].set_xticks(range(len(velo_inj)))
axes[1].set_xticklabels(velo_inj.index, fontsize=9)
axes[1].axhline(ps_velo_full['injured_future'].mean() * 100,
                color='gray', ls='--', lw=1.5, label='Overall rate')
axes[1].set_ylabel('Future Injury Rate (%)')
axes[1].set_title('Injury Rate by Velocity Quintile\\n'
                  '(higher velocity = higher arm stress?)', fontweight='bold')
axes[1].legend(fontsize=9)
for i, (m, n) in enumerate(zip(velo_inj['mean'], velo_inj['count'])):
    axes[1].text(i, m*100 + 0.5, f'{m*100:.1f}%\\nn={n}',
                 ha='center', fontsize=8)

plt.suptitle('Figure 10: Velocity Delta and Injury Risk', y=1.01,
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_10_velocity_delta.png')
plt.show()

# Compute velocity change features (framework — requires longitudinal data)
print('Velocity change features (to be computed in Notebook 05):')
print('  velocity_change_30d  = velo_this_week − avg_velo_prior_30_days')
print('  velocity_change_60d  = velo_this_week − avg_velo_prior_60_days')
print('  velocity_change_90d  = velo_this_week − avg_velo_prior_90_days')
print()
print('These require full Statcast history (2015–2024) loaded in a rolling fashion.')\
"""

C23_S7_HEADER = """\
---
## Section 7: Pitch Mix Analysis

A pitcher's pitch mix encodes both their physical capabilities and their
strategic tendencies. From an injury-risk perspective:

- **High slider usage** is frequently implicated in elbow stress (UCL loading)
- **High fastball usage at high velocity** accumulates arm stress quickly
- **Changeup / curveball** pitchers may distribute arm stress differently
- **Pitch mix changes** — a pitcher throwing fewer fastballs than their norm —
  can signal mechanical dysfunction or arm fatigue

Key features:
- `fb_pct`: fastball usage rate (FF + SI + FC)
- `sl_pct`: slider usage rate
- `breaking_pct`: all breaking balls combined
- `offspeed_pct`: changeup + splitter rate\
"""

C24_PITCHMIX = """\
# ── Pitch usage distributions ─────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
axes_flat = axes.flatten()

pitch_features = [
    ('fb_pct',       'Fastball %',   CLR_NEUTRAL),
    ('sl_pct',       'Slider %',     '#e6550d'),
    ('cu_pct',       'Curveball %',  '#6a51a3'),
    ('ch_pct',       'Changeup %',   '#74c476'),
    ('breaking_pct', 'Breaking %',   '#969696'),
    ('sweep_pct',    'Sweeper %',    '#3182bd'),
]

for ax, (feat, label, color) in zip(axes_flat, pitch_features):
    data = pitcher_season[feat].dropna()
    ax.hist(data * 100, bins=20, color=color, alpha=0.7, edgecolor='white')
    ax.axvline(data.mean() * 100, color='black', ls='--', lw=1.5,
               label=f'Mean = {data.mean()*100:.1f}%')
    ax.set_xlabel(f'{label} (%)')
    ax.set_ylabel('Count')
    ax.set_title(label, fontweight='bold')
    ax.legend(fontsize=9)

plt.suptitle('Figure 11: Pitch Mix Distributions', y=1.01,
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_11_pitch_mix_distributions.png')
plt.show()\
"""

C25_PITCHMIX_INJ = """\
# ── Pitch mix by injury status ────────────────────────────────────────────────
pitch_feats = ['fb_pct', 'sl_pct', 'cu_pct', 'ch_pct', 'breaking_pct']
pitch_labels = ['Fastball %', 'Slider %', 'Curveball %', 'Changeup %', 'Breaking %']

fig, axes = plt.subplots(1, len(pitch_feats), figsize=(18, 5))

pval_results = {}
for ax, feat, label in zip(axes, pitch_feats, pitch_labels):
    groups = [
        pitcher_season[pitcher_season['injured_future']==0][feat].dropna() * 100,
        pitcher_season[pitcher_season['injured_future']==1][feat].dropna() * 100,
    ]
    bp = ax.boxplot(groups, patch_artist=True, widths=0.5,
                    labels=['Healthy', 'Injured'])
    bp['boxes'][0].set_facecolor(CLR_HEALTHY); bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor(CLR_INJURED); bp['boxes'][1].set_alpha(0.6)
    ax.set_ylabel('%')
    ax.set_title(label, fontweight='bold')
    # Mann-Whitney U test (non-parametric, robust to skewed distributions)
    u_stat, p = stats.mannwhitneyu(groups[0], groups[1], alternative='two-sided')
    pval_results[feat] = p
    sig = '**' if p < 0.01 else ('*' if p < 0.05 else 'ns')
    ax.set_xlabel(f'p={p:.3f} {sig}')

plt.suptitle('Figure 12: Pitch Mix by Future Injury Status', y=1.01,
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_12_pitch_mix_injury.png')
plt.show()

print('Mann-Whitney p-values (pitch mix vs. injury label):')
for feat, p in sorted(pval_results.items(), key=lambda x: x[1]):
    sig = '← significant' if p < 0.05 else ''
    print(f'  {feat:<20} p = {p:.4f}  {sig}')\
"""

C26_S8_HEADER = """\
---
## Section 8: Pitch Movement and Mechanics Proxies

Statcast release point and movement metrics are imprecise proxies for
pitching mechanics. However, research (Boddy et al., Driveline Baseball)
has identified that:

1. **Release point drift** — changes in where a ball is released — can
   indicate compensatory mechanics from discomfort or fatigue
2. **Spin rate changes** — sudden drops in spin rate may signal grip
   weakness or neural fatigue
3. **Extension decrease** — reduced extension can indicate guarding
4. **Induced vertical break (pfx_z)** and horizontal break (pfx_x) changes
   reflect both mechanics and pitch effort level

**Drift metrics** are defined as the within-pitcher standard deviation of
release point coordinates across appearances — higher variability = more
inconsistent mechanics.\
"""

C27_MECHANICS = """\
# ── Release point and movement distributions ──────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

mech_features = [
    ('avg_rel_x',   'Release Position X (ft)',    CLR_NEUTRAL),
    ('avg_rel_z',   'Release Height Z (ft)',       CLR_NEUTRAL),
    ('avg_extension','Extension (ft)',              CLR_NEUTRAL),
    ('avg_pfx_x',   'Horizontal Break (ft)',        CLR_NEUTRAL),
    ('avg_pfx_z',   'Induced Vert. Break (ft)',     CLR_NEUTRAL),
    ('avg_spin',    'Spin Rate (rpm)',               CLR_NEUTRAL),
]

for ax, (feat, label, color) in zip(axes.flatten(), mech_features):
    data = pitcher_season[feat].dropna()
    if len(data) < 5:
        ax.text(0.5, 0.5, 'Insufficient data', transform=ax.transAxes,
                ha='center', va='center')
        ax.set_title(label, fontweight='bold')
        continue
    ax.hist(data, bins=25, color=color, alpha=0.7, edgecolor='white')
    ax.axvline(data.mean(), color='black', ls='--', lw=1.5,
               label=f'Mean = {data.mean():.2f}')
    ax.set_xlabel(label)
    ax.set_ylabel('Count')
    ax.set_title(label, fontweight='bold')
    ax.legend(fontsize=9)

plt.suptitle('Figure 13: Mechanics Proxy Distributions', y=1.01,
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_13_mechanics_distributions.png')
plt.show()\
"""

C28_DRIFT = """\
# ── Release point drift metrics ───────────────────────────────────────────────
# rel_x_std and rel_z_std are the avg within-game std of release position.
# Higher values = more variable release point = potential mechanical inconsistency.

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# 1) Release point scatter (healthy vs injured)
ax = axes[0]
for label, group in pitcher_season.groupby('injured_future'):
    color = CLR_INJURED if label == 1 else CLR_HEALTHY
    name  = 'Future Injured' if label == 1 else 'Healthy'
    alpha = 0.4
    ax.scatter(group['avg_rel_x'].dropna(),
               group['avg_rel_z'].dropna(),
               c=color, alpha=alpha, s=20, label=name)
ax.set_xlabel('Release Position X (ft)')
ax.set_ylabel('Release Height Z (ft)')
ax.set_title('Release Point: Healthy vs Injured', fontweight='bold')
ax.legend(fontsize=9)

# 2) Release drift (std) by injury status
for i, (label, group) in enumerate(pitcher_season.groupby('injured_future')):
    color = CLR_INJURED if label == 1 else CLR_HEALTHY
    bp = axes[1].boxplot(group['rel_x_std'].dropna().values,
                         positions=[i], widths=0.5,
                         patch_artist=True)
    bp['boxes'][0].set_facecolor(color)
    bp['boxes'][0].set_alpha(0.7)
axes[1].set_xticks([0, 1])
axes[1].set_xticklabels(['Healthy', 'Future Injured'])
axes[1].set_ylabel('Release X Std Dev (ft)')
axes[1].set_title('Horizontal Release Drift by Injury Status', fontweight='bold')

# 3) Spin rate by injury status
for i, (label, group) in enumerate(pitcher_season.groupby('injured_future')):
    color = CLR_INJURED if label == 1 else CLR_HEALTHY
    data  = group['avg_spin'].dropna()
    if len(data) > 5:
        bp = axes[2].boxplot(data.values, positions=[i], widths=0.5,
                             patch_artist=True)
        bp['boxes'][0].set_facecolor(color)
        bp['boxes'][0].set_alpha(0.7)
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(['Healthy', 'Future Injured'])
axes[2].set_ylabel('Avg Spin Rate (rpm)')
axes[2].set_title('Spin Rate by Injury Status', fontweight='bold')

# Mann-Whitney tests
for feat, label in [('rel_x_std', 'Release X std'),
                     ('rel_z_std', 'Release Z std'),
                     ('avg_spin',  'Spin rate')]:
    g0 = pitcher_season[pitcher_season['injured_future']==0][feat].dropna()
    g1 = pitcher_season[pitcher_season['injured_future']==1][feat].dropna()
    if len(g0) > 5 and len(g1) > 5:
        _, p = stats.mannwhitneyu(g0, g1, alternative='two-sided')
        print(f'{label:<25} p = {p:.4f}  '
              f'{"← significant" if p < 0.05 else ""}')

plt.suptitle('Figure 14: Release Point Drift and Mechanics', y=1.01,
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_14_mechanics_drift.png')
plt.show()\
"""

C29_S9_HEADER = """\
---
## Section 9: Correlation and Feature Relationships

To identify the most predictive features, we use two complementary approaches:

1. **Pearson/Spearman correlation** — linear relationships between features
   and the binary injury label; fast and interpretable but misses non-linear
   associations
2. **Mutual information (MI)** — measures information shared between each
   feature and the injury label; captures non-linear relationships; preferred
   for feature selection

A high-correlation feature is a strong *linear* predictor. A high-MI feature
is a strong predictor of any kind. For injury risk — a complex, multi-causal
outcome — MI is generally more informative than linear correlation.\
"""

C30_CORR = """\
# ── Feature matrix and correlation matrix ─────────────────────────────────────
feature_cols = [
    'age', 'n_games', 'total_pitches', 'avg_pitch_game',
    'avg_velo', 'max_velo', 'std_velo',
    'avg_spin', 'avg_extension',
    'avg_pfx_x', 'avg_pfx_z',
    'avg_rel_x', 'avg_rel_z',
    'rel_x_std', 'rel_z_std',
    'fb_pct', 'breaking_pct', 'offspeed_pct',
    'sl_pct', 'cu_pct', 'ch_pct', 'sweep_pct',
    'velo_delta',
]
feature_cols = [c for c in feature_cols if c in pitcher_season.columns]

feat_df = pitcher_season[feature_cols + ['injured_future']].dropna(
    subset=['injured_future']
)
# Fill remaining NaNs with column median for correlation computation
feat_df_filled = feat_df.fillna(feat_df.median(numeric_only=True))

corr_matrix = feat_df_filled[feature_cols].corr()

fig, ax = plt.subplots(figsize=(14, 11))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix, mask=mask, ax=ax,
    cmap='RdBu_r', center=0, vmin=-1, vmax=1,
    annot=len(feature_cols) <= 16,
    fmt='.2f' if len(feature_cols) <= 16 else '',
    linewidths=0.5, linecolor='white',
    square=True,
)
ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=14)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_15_correlation_matrix.png')
plt.show()
print(f'Correlation matrix: {len(feature_cols)} × {len(feature_cols)} features')\
"""

C31_MI = """\
# ── Mutual information + injury correlation ranking ───────────────────────────
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import mutual_info_classif

feat_df_imp = feat_df.copy()
X = feat_df_imp[feature_cols].values
y = feat_df_imp['injured_future'].values

# Impute missing values
imputer = SimpleImputer(strategy='median')
X_imp   = imputer.fit_transform(X)

# Mutual information (classification)
mi_scores = mutual_info_classif(X_imp, y, random_state=42, n_neighbors=5)

# Pearson correlation with injury label
pearson_corr = feat_df_filled[feature_cols].corrwith(
    feat_df_filled['injured_future']
).abs()

# Combined ranking DataFrame
feature_ranking = pd.DataFrame({
    'feature':     feature_cols,
    'mi_score':    mi_scores,
    'corr_abs':    pearson_corr.values,
}).sort_values('mi_score', ascending=False).reset_index(drop=True)
feature_ranking['rank_mi']   = range(1, len(feature_ranking) + 1)
feature_ranking['rank_corr'] = feature_ranking['corr_abs'].rank(ascending=False).astype(int)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 1) Mutual information
colors_mi = [CLR_INJURED if i < 5 else CLR_NEUTRAL
             for i in range(len(feature_ranking))]
axes[0].barh(feature_ranking['feature'][::-1],
             feature_ranking['mi_score'][::-1],
             color=colors_mi[::-1])
axes[0].set_xlabel('Mutual Information Score')
axes[0].set_title('Feature Importance: Mutual Information\\n(higher = more predictive)',
                  fontweight='bold')

# 2) Pearson correlation with injury label
feat_sorted_corr = feature_ranking.sort_values('corr_abs', ascending=False)
colors_corr = [CLR_INJURED if i < 5 else CLR_NEUTRAL
               for i in range(len(feat_sorted_corr))]
axes[1].barh(feat_sorted_corr['feature'][::-1],
             feat_sorted_corr['corr_abs'][::-1],
             color=colors_corr[::-1])
axes[1].set_xlabel('|Pearson Correlation| with Injury Label')
axes[1].set_title('Feature Importance: Linear Correlation', fontweight='bold')

plt.suptitle('Figure 16: Feature Importance Rankings', y=1.01,
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_16_feature_importance.png')
plt.show()

print('Top 10 features by Mutual Information:')
print(feature_ranking[['rank_mi', 'feature', 'mi_score', 'corr_abs']].head(10).to_string())\
"""

C32_S10_HEADER = """\
---
## Section 10: Injury Countdown Analysis

The injury countdown is the gold-standard EDA for injury prediction: align
all injured pitcher observations on the injury date (Day 0), then plot average
feature values across the pre-injury trajectory (Day -90 to Day 0).

A **consistent downward trend in velocity or spin rate** approaching Day 0
would provide strong evidence that these features carry leading-indicator signal.

> ⚠️ **TEST_MODE limitation:** We have only 7 days of Statcast (June 1-7).
> Meaningful countdown analysis requires months of history per pitcher.
> This section implements the complete framework and demonstrates it on
> pitchers injured in late June 2023 (who have a short pre-injury window
> in our data).

> With full 2015-2024 data, this analysis will have tens of thousands of
> pitcher-week observations across hundreds of injuries — the results will
> be statistically robust.\
"""

C33_COUNTDOWN = """\
# ── Injury countdown framework ────────────────────────────────────────────────
# Step 1: Find pitchers whose injury is within a few weeks of our window end.
# Step 2: Compute their days_until_injury from each pitch record.
# Step 3: Average features across the pre-injury window.

# For TEST_MODE: use pitchers injured June 8–July 31 2023
NEAR_INJURY_END = pd.Timestamp('2023-07-31')
near_inj = inj[
    (inj['transaction_date'] > WINDOW_END) &
    (inj['transaction_date'] <= NEAR_INJURY_END)
].copy()
near_inj_ids = set(near_inj['player_id'].astype(int).unique())
near_in_sc   = near_inj_ids & set(sc['pitcher'].astype(int).unique())

print(f'Pitchers injured June 8 – July 31: {len(near_inj_ids)}')
print(f'  ...with Statcast data in our window: {len(near_in_sc)}')
print()

if len(near_in_sc) == 0:
    print('No pitchers have both near-injury IL placements AND June 1-7 Statcast data.')
    print('This is expected with TEST_MODE. The countdown analysis will be fully')
    print('populated when full 2015-2024 Statcast data is loaded.')
else:
    # Build pitch-level countdown dataset
    countdown_rows = []
    for pid in near_in_sc:
        inj_date = near_inj[near_inj['player_id'] == pid]['transaction_date'].min()
        pitcher_pitches = sc[sc['pitcher'] == pid].copy()
        pitcher_pitches['days_until_injury'] = (
            (inj_date - pitcher_pitches['game_date']).dt.days
        )
        countdown_rows.append(pitcher_pitches)

    countdown_df = pd.concat(countdown_rows, ignore_index=True)
    countdown_df = countdown_df[countdown_df['days_until_injury'] >= 0]
    print(f'Countdown pitches: {len(countdown_df)} from {len(near_in_sc)} pitchers')

    # Average by days_until_injury
    countdown_avg = countdown_df.groupby('days_until_injury').agg(
        avg_velo  = ('release_speed',      'mean'),
        avg_spin  = ('release_spin_rate',  'mean'),
        n_pitches = ('pitch_type',          'count'),
    ).reset_index()
    print(countdown_avg.to_string())\
"""

C34_COUNTDOWN_PLOT = """\
# ── Injury countdown visualization ────────────────────────────────────────────
# Show the framework visualization. For TEST_MODE we demonstrate with a
# simulated expected pattern for clarity; replace with real data once available.

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Try to plot real data if available; otherwise show labeled placeholder
def plot_countdown(ax, feature_name, y_label, real_data=None):
    ax.set_xlabel('Days Until Injury (0 = injury date)')
    ax.set_ylabel(y_label)
    ax.invert_xaxis()
    ax.axvline(0, color=CLR_INJURED, lw=2, ls='--', alpha=0.5)
    ax.text(1, ax.get_ylim()[1] if ax.get_ylim()[1] != 1 else 1,
            'Injury\\nDate', color=CLR_INJURED, fontsize=9, ha='left')
    if real_data is not None and len(real_data) > 2:
        ax.plot(real_data['days_until_injury'], real_data[feature_name],
                'o-', color=CLR_INJURED, lw=2, ms=6)
        ax.set_title(f'{y_label}: Pre-Injury Trajectory', fontweight='bold')
    else:
        # Labeled placeholder: simulated velocity decline pattern
        days = np.arange(-90, 1)
        signal = np.where(days > -30, 93 + days * 0.04, 93.0)
        noise  = np.random.default_rng(42).normal(0, 0.3, len(days))
        ax.plot(-days, signal + noise, color=CLR_INJURED, lw=2, alpha=0.7)
        ax.set_title(f'{y_label}: Pre-Injury Trajectory\\n'
                     f'[Simulated — full data needed]', fontweight='bold')
        ax.text(0.05, 0.10, 'TEST_MODE: requires full season data',
                transform=ax.transAxes, fontsize=9, color='gray',
                bbox=dict(boxstyle='round', facecolor='lightyellow'))

real_countdown = countdown_df if 'countdown_df' in dir() and len(near_in_sc) > 0 else None
real_avg = countdown_avg if real_countdown is not None else None

plot_countdown(axes[0], 'avg_velo', 'Avg Fastball Velocity (mph)', real_avg)
plot_countdown(axes[1], 'avg_spin', 'Avg Spin Rate (rpm)', real_avg)

plt.suptitle('Figure 17: Injury Countdown Analysis\\n'
             '(Day 0 = IL placement date)', y=1.01,
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_17_injury_countdown.png')
plt.show()
print('Saved fig_17_injury_countdown.png')\
"""

C35_S11_HEADER = """\
---
## Section 11: Pitcher Archetypes

Clustering pitchers by their physical profile and usage patterns can reveal
distinct archetypes with different injury risk profiles. Front offices think
about pitchers in terms of archetypes (e.g., "power arm", "command pitcher")
and managers make usage decisions based on perceived archetype.

**Hypothesized archetypes:**
| Archetype | Expected profile | Injury risk hypothesis |
|---|---|---|
| Power starters | High velo, high pitch count, low breaking | High elbow/shoulder |
| Slider-heavy | Moderate velo, high SL% | UCL stress, forearm |
| Command pitchers | Below-avg velo, low breaking% | Lower arm stress |
| Workhorse starters | High pitch count, balanced mix | Fatigue-related |
| Short relievers | Low pitch count, high velo burst | Elbow inflammation |

We use **KMeans with PCA** for visualization. KMeans was chosen for its
interpretability — cluster centers correspond to average pitcher profiles.\
"""

C36_CLUSTER = """\
# ── Pitcher archetype clustering ──────────────────────────────────────────────
cluster_features = [
    'avg_velo', 'avg_pitch_game', 'fb_pct', 'sl_pct',
    'breaking_pct', 'avg_pfx_z', 'avg_extension',
]
cluster_features = [c for c in cluster_features if c in pitcher_season.columns]

ps_cluster = pitcher_season[cluster_features + ['pitcher', 'injured_future',
                                                  'player_name', 'role']].dropna(
    subset=cluster_features
).copy()

# Scale features
scaler  = StandardScaler()
X_scale = scaler.fit_transform(ps_cluster[cluster_features])

# Elbow method to choose K
inertias = []
K_RANGE  = range(2, 9)
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scale)
    inertias.append(km.inertia_)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].plot(list(K_RANGE), inertias, 'o-', color=CLR_NEUTRAL)
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia (within-cluster sum of squares)')
axes[0].set_title('Elbow Method: Optimal K', fontweight='bold')
axes[0].axvline(5, color=CLR_INJURED, ls='--', lw=1.5, label='K=5 (selected)')
axes[0].legend()

# Fit final model with K=5
N_CLUSTERS = 5
km = KMeans(n_clusters=N_CLUSTERS, random_state=42, n_init=20)
ps_cluster['cluster'] = km.fit_predict(X_scale)

# PCA for visualization
pca   = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scale)
ps_cluster['pca1'] = X_pca[:, 0]
ps_cluster['pca2'] = X_pca[:, 1]

for c in range(N_CLUSTERS):
    mask  = ps_cluster['cluster'] == c
    color = PALETTE_CLUSTER[c]
    inj_r = ps_cluster[mask]['injured_future'].mean()
    axes[1].scatter(ps_cluster.loc[mask, 'pca1'],
                    ps_cluster.loc[mask, 'pca2'],
                    c=[color], alpha=0.6, s=30,
                    label=f'Cluster {c+1} (inj: {inj_r:.0%})')

axes[1].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} var)')
axes[1].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} var)')
axes[1].set_title('Pitcher Archetypes: PCA Projection', fontweight='bold')
axes[1].legend(fontsize=8)

plt.suptitle('Figure 18: Pitcher Archetype Clustering (KMeans, K=5)', y=1.01,
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_18_pitcher_archetypes_pca.png')
plt.show()
print(f'Variance explained by PC1+PC2: '
      f'{pca.explained_variance_ratio_[:2].sum():.1%}')\
"""

C37_CLUSTER_PROFILES = """\
# ── Cluster profiles and injury rates ────────────────────────────────────────
# Build cluster summary table
cluster_summary = (
    ps_cluster.groupby('cluster')[cluster_features + ['injured_future']]
    .mean()
    .round(3)
)
cluster_summary['n_pitchers']   = ps_cluster.groupby('cluster')['pitcher'].count()
cluster_summary['injury_rate%'] = (cluster_summary['injured_future'] * 100).round(1)

# Label clusters by dominant characteristic
cluster_labels = {}
for c in range(N_CLUSTERS):
    row = cluster_summary.loc[c]
    velo = row['avg_velo']
    pc   = row['avg_pitch_game']
    fb   = row['fb_pct']
    sl   = row['sl_pct']
    if pc >= 80 and velo >= 93:
        label = 'Power Starter'
    elif pc >= 80:
        label = 'Workhorse Starter'
    elif sl >= 0.20:
        label = 'Slider-Heavy'
    elif velo >= 94.5:
        label = 'High-Velo Reliever'
    else:
        label = 'Command/Finesse'
    cluster_labels[c] = label
    cluster_summary.loc[c, 'archetype'] = label

print('Cluster Profiles:')
print(cluster_summary[['archetype', 'n_pitchers', 'avg_velo', 'avg_pitch_game',
                         'fb_pct', 'sl_pct', 'injury_rate%']].to_string())
print()

fig, ax = plt.subplots(figsize=(10, 5))
colors_c = [PALETTE_CLUSTER[i] for i in range(N_CLUSTERS)]
archetypes = [cluster_labels[i] for i in range(N_CLUSTERS)]
inj_rates  = [cluster_summary.loc[i, 'injury_rate%'] for i in range(N_CLUSTERS)]
order = np.argsort(inj_rates)[::-1]
bars = ax.bar([archetypes[i] for i in order],
              [inj_rates[i] for i in order],
              color=[colors_c[i] for i in order],
              edgecolor='white')
ax.axhline(ps_cluster['injured_future'].mean() * 100,
           color='gray', ls='--', lw=1.5, label='Overall injury rate')
for bar, rate in zip(bars, [inj_rates[i] for i in order]):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
            f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
ax.set_ylabel('Future Injury Rate (%)')
ax.set_title('Injury Rate by Pitcher Archetype', fontweight='bold')
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig_19_archetype_injury_rates.png')
plt.show()

# Save cluster assignments
ps_cluster[['pitcher', 'cluster', 'pca1', 'pca2']].to_parquet(
    PROCESSED_DIR / 'pitcher_clusters.parquet', index=False
)
print('Saved pitcher_clusters.parquet')\
"""

C38_S12_HEADER = """\
---
## Section 12: EDA Conclusions

### Summary of Findings

This section synthesizes the evidence from all prior sections into an
actionable feature ranking and a set of research questions to drive modeling.

### Methodology note
All signal estimates in this section are based on TEST_MODE data (1 week,
400 pitchers). With full season data (2015–2024, ~700 pitchers × ~10 seasons),
statistical power increases dramatically and many marginal signals will become
significant.

### Research questions surfaced by this EDA
1. **Are starters inherently riskier than relievers?** — Role analysis (§4)
   showed differential rates; needs multi-season data to confirm.
2. **Does slider-heavy usage elevate UCL risk?** — Pitch mix signal in §7
   warrants testing with injury type as the label.
3. **Does velocity decline precede injury?** — The countdown framework (§10)
   is the correct test; requires longitudinal data.
4. **Can workload thresholds be identified?** — ACWR > 1.3 hypothesis will
   be directly testable in notebook 05 with full season ACWR computed.
5. **Do pitcher archetypes have distinct mechanisms?** — Cluster-level injury
   type distributions (elbow vs. shoulder vs. oblique) are a key next step.\
"""

C39_CONCLUSIONS = """\
# ── Top 20 candidate features — ranked by MI score + domain knowledge ─────────
# The ranking below blends statistical signal (MI score) with domain knowledge
# about known injury mechanisms. Features marked (†) require longitudinal data.

FEATURE_DESCRIPTIONS = {
    'age':                 'Pitcher age at observation date',
    'total_pitches':       'Acute workload — pitches in window',
    'avg_pitch_game':      'Workload intensity per appearance',
    'n_games':             'Appearance frequency',
    'avg_velo':            'Mean fastball velocity',
    'max_velo':            'Peak velocity (burst effort indicator)',
    'std_velo':            'Velocity variability (fatigue proxy)',
    'velo_delta':          'Velocity vs. league mean (relative effort)',
    'fb_pct':              'Fastball usage rate',
    'sl_pct':              'Slider usage rate (UCL stress proxy)',
    'breaking_pct':        'All breaking ball usage',
    'ch_pct':              'Changeup usage rate',
    'sweep_pct':           'Sweeper usage rate (new pitch, unknown risk)',
    'avg_pfx_z':           'Induced vertical break (effort/mechanics)',
    'avg_pfx_x':           'Horizontal break',
    'avg_spin':            'Mean spin rate (effort/grip indicator)',
    'rel_x_std':           'Horizontal release drift (mechanics consistency)',
    'rel_z_std':           'Vertical release drift',
    'avg_extension':       'Extension (delivery length)',
    'avg_rel_z':           'Release height',
}

# Features requiring longitudinal data (†)
LONGITUDINAL_FEATURES = [
    'velocity_change_30d', 'velocity_change_60d', 'velocity_change_90d',
    'acwr_7_28',           'acwr_7_21',
    'spin_change_30d',     'release_drift_14d',
    'days_since_last_start', 'ip_last_7d', 'ip_last_28d',
    'injury_history_count', 'prior_il_days',
]

# Merge MI scores with descriptions
top_features = feature_ranking.head(20).copy()
top_features['description'] = top_features['feature'].map(FEATURE_DESCRIPTIONS)
top_features['requires_longitudinal'] = False

# Add longitudinal features to the list with placeholder MI scores
for feat in LONGITUDINAL_FEATURES[:6]:
    row = {
        'feature': feat,
        'mi_score': np.nan,
        'corr_abs': np.nan,
        'rank_mi': '—',
        'rank_corr': '—',
        'description': 'Computed from rolling history (†)',
        'requires_longitudinal': True,
    }
    top_features = pd.concat([top_features, pd.DataFrame([row])], ignore_index=True)

top_features = top_features.reset_index(drop=True)
top_features.index += 1

print('='*80)
print('TOP CANDIDATE FEATURES FOR INJURY RISK+ MODELING')
print('='*80)
print(f'{"#":<4} {"Feature":<30} {"MI Score":<12} {"|r|":<8} {"Description"}')
print('─'*80)
for idx, row in top_features.iterrows():
    mi  = f'{row.mi_score:.4f}' if pd.notna(row.mi_score) else '   —   '
    cor = f'{row.corr_abs:.4f}' if pd.notna(row.corr_abs) else '   —  '
    dagger = ' †' if row.requires_longitudinal else ''
    print(f'{idx:<4} {row.feature + dagger:<30} {mi:<12} {cor:<8} {row.description}')
print('─'*80)
print()
print('† Requires longitudinal Statcast history (computed in Notebook 05)')
print()
print('WEAK FEATURES (low MI, high collinearity):')
weak = feature_ranking.tail(5)
for _, row in weak.iterrows():
    print(f'  {row.feature:<25} MI = {row.mi_score:.4f}')\
"""

C40_SAVE = """\
# ── Save EDA outputs ─────────────────────────────────────────────────────────
# 1) Pitcher-season feature matrix (with labels)
out_cols = [c for c in feature_cols + ['pitcher', 'player_name', 'role',
                                         'age', 'injured_future']
            if c in pitcher_season.columns]
pitcher_season[out_cols].to_parquet(
    PROCESSED_DIR / 'pitcher_eda_features.parquet', index=False
)
print(f'Saved pitcher_eda_features.parquet  ({len(pitcher_season)} rows)')

# 2) Feature ranking
feature_ranking.to_parquet(PROCESSED_DIR / 'feature_ranking_mi.parquet', index=False)
feature_ranking.to_csv(PROCESSED_DIR / 'feature_ranking_mi.csv', index=False)
print(f'Saved feature_ranking_mi.parquet + .csv  ({len(feature_ranking)} features)')

# 3) EDA summary JSON for provenance
prov_path = Path('data/raw/provenance.json')
prov = json.loads(prov_path.read_text()) if prov_path.exists() else {}

from datetime import datetime, timezone
prov['eda'] = {
    'run_at':               datetime.now(timezone.utc).isoformat(),
    'pitchers_analyzed':    len(pitcher_season),
    'n_future_injured':     int(pitcher_season['injured_future'].sum()),
    'injury_rate':          float(pitcher_season['injured_future'].mean()),
    'top_mi_features':      feature_ranking.head(5)['feature'].tolist(),
    'n_figures_generated':  19,
    'archetypes_found':     N_CLUSTERS,
    'notes':                ('TEST_MODE — 1-week Statcast window. Longitudinal '
                             'features (ACWR, velocity decline) require full data.'),
}
prov_path.write_text(json.dumps(prov, indent=2, default=str))
print(f'Updated provenance.json')
print()
print('All figures saved to:', FIGURES_DIR.resolve())
print()
print('NEXT STEP: Notebook 05 — Feature Engineering')
print('  Primary tasks:')
print('  1. Pull full 2015-2024 Statcast season data (TEST_MODE = False)')
print('  2. Compute rolling ACWR and velocity decline features')
print('  3. Build master feature matrix at pitcher-week granularity')
print('  4. Construct binary injury labels at 30 / 60 / 90-day windows')\
"""

# ---------------------------------------------------------------------------
# ASSEMBLE NOTEBOOK
# ---------------------------------------------------------------------------

cells = [
    md(C0_HEADER,           "c00-header"),
    code(C1_SETUP,          "c01-setup"),
    code(C2_LOAD,           "c02-load"),
    code(C3_AGGREGATE,      "c03-aggregate"),
    code(C4_LABELS,         "c04-labels"),
    md(C5_S1_HEADER,        "c05-s1-header"),
    code(C6_OVERVIEW,       "c06-overview"),
    code(C7_MISSING,        "c07-missing"),
    code(C8_COMPLETENESS,   "c08-completeness"),
    md(C9_S2_HEADER,        "c09-s2-header"),
    code(C10_INJ_TYPES,     "c10-inj-types"),
    code(C11_DAYSLOST,      "c11-dayslost"),
    md(C12_S3_HEADER,       "c12-s3-header"),
    code(C13_CLASSBALANCE,  "c13-classbalance"),
    md(C14_S4_HEADER,       "c14-s4-header"),
    code(C15_AGE,           "c15-age"),
    code(C16_ROLE,          "c16-role"),
    md(C17_S5_HEADER,       "c17-s5-header"),
    code(C18_WORKLOAD,      "c18-workload"),
    code(C19_ACWR,          "c19-acwr"),
    md(C20_S6_HEADER,       "c20-s6-header"),
    code(C21_VELO,          "c21-velo"),
    code(C22_VELO_FEATURES, "c22-velo-features"),
    md(C23_S7_HEADER,       "c23-s7-header"),
    code(C24_PITCHMIX,      "c24-pitchmix"),
    code(C25_PITCHMIX_INJ,  "c25-pitchmix-inj"),
    md(C26_S8_HEADER,       "c26-s8-header"),
    code(C27_MECHANICS,     "c27-mechanics"),
    code(C28_DRIFT,         "c28-drift"),
    md(C29_S9_HEADER,       "c29-s9-header"),
    code(C30_CORR,          "c30-corr"),
    code(C31_MI,            "c31-mi"),
    md(C32_S10_HEADER,      "c32-s10-header"),
    code(C33_COUNTDOWN,     "c33-countdown"),
    code(C34_COUNTDOWN_PLOT,"c34-countdown-plot"),
    md(C35_S11_HEADER,      "c35-s11-header"),
    code(C36_CLUSTER,       "c36-cluster"),
    code(C37_CLUSTER_PROFILES,"c37-cluster-profiles"),
    md(C38_S12_HEADER,      "c38-s12-header"),
    code(C39_CONCLUSIONS,   "c39-conclusions"),
    code(C40_SAVE,          "c40-save"),
]

notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "pitcher-injury-risk",
            "language": "python",
            "name": "pitcher-injury-risk",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out_path = Path("notebooks/04_eda.ipynb")
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Written: {out_path}")
print(f"Cells:   {len(cells)}")
code_cells = [c for c in cells if c["cell_type"] == "code"]
print(f"Code:    {len(code_cells)}")
md_cells   = [c for c in cells if c["cell_type"] == "markdown"]
print(f"Markdown:{len(md_cells)}")
