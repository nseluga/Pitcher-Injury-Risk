"""
Generate notebooks/13_dashboard.ipynb programmatically.
Run from project root: python scripts/generate_notebook_13.py
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
# 13 — Dashboard

## Purpose
Prototype the interactive Injury Risk+ dashboard using Plotly. This notebook
implements static versions of all core dashboard components, validating that
the data pipeline produces output suitable for an interactive application.

The four components implemented here mirror the planned dashboard layout:
1. **Season leaderboard** — top/bottom 25 Injury Risk+ pitchers with filters
2. **Pitcher lookup** — per-pitcher score with archetype-relative context
3. **Trend explorer** — Injury Risk+ over career for selected pitchers
4. **Component breakdown** — probability / days missed / severity contribution view

## Inputs
- `data/processed/injury_risk_plus_scores.parquet`
- `data/processed/player_metadata_clean.parquet` (for pitcher names)

## Outputs
All outputs are HTML files in `reports/figures/dashboard/`. No parquet or CSV
files are required — the execution-only contract means the notebook need only
run clean end-to-end.

## Note on interactivity
Plotly figures are saved as self-contained HTML with embedded JavaScript.
A future deployment step (Streamlit or Dash) can re-use these components with
live callbacks. The slider/filter logic shown here would move into callback
functions in that deployment.\
"""

C1 = """\
import sys
import warnings
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', 20)

PROJECT_ROOT = str(Path('.').resolve())
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

FIGURES_DIR = Path('reports/figures/dashboard')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print(f'Dashboard output dir: {FIGURES_DIR}')
print(f'Plotly imported OK')\
"""

C2_MD = """\
## 1. Load and Prepare Data

We load the per-appearance Injury Risk+ scores and aggregate to the
pitcher-season level (mean IR+ and components per pitcher per season). We also
join player name metadata so the leaderboard is human-readable.

Aggregating to pitcher-season (rather than using raw per-appearance rows)
is the correct unit for a season-level dashboard: a single pitcher appearing
200 times in 2022 should count once, not 200 times.\
"""

C2 = """\
scores_path = Path('data/processed/injury_risk_plus_scores.parquet')
if not scores_path.exists():
    raise FileNotFoundError('Run notebook 09 first to build injury_risk_plus_scores.parquet')

scores_raw = pd.read_parquet(scores_path)
scores_raw['pitcher_id'] = scores_raw['pitcher_id'].astype(int)
scores_raw['season'] = scores_raw['season'].astype(int)

print(f'Raw scores: {scores_raw.shape[0]:,} rows (per-appearance)')
print(f'Seasons: {sorted(scores_raw["season"].unique())}')
print(f'Unique pitchers: {scores_raw["pitcher_id"].nunique():,}')

# Aggregate to pitcher-season level.
agg = {
    'injury_risk_plus': 'mean',
    'risk_percentile': 'mean',
    'injury_prob_30d': 'mean',
    'expected_days_lost': 'mean',
    'hazard_rate': 'mean',
    'raw_risk_score': 'mean',
    'archetype': 'first',
}
ps = (
    scores_raw.groupby(['pitcher_id', 'season'], as_index=False)
    .agg(agg)
    .rename(columns={
        'injury_risk_plus': 'ir_plus',
        'risk_percentile': 'percentile',
    })
)
ps['ir_plus'] = ps['ir_plus'].round(1)
ps['percentile'] = ps['percentile'].round(1)

# Merge player names.
meta_path = Path('data/processed/player_metadata_clean.parquet')
if meta_path.exists():
    meta = pd.read_parquet(meta_path, columns=['player_id', 'player_name'])
    meta = meta.rename(columns={'player_id': 'pitcher_id'})
    meta['pitcher_id'] = meta['pitcher_id'].astype(int)
    ps = ps.merge(meta, on='pitcher_id', how='left')
    ps['player_name'] = ps['player_name'].fillna(ps['pitcher_id'].astype(str))
else:
    ps['player_name'] = ps['pitcher_id'].astype(str)

print(f'Pitcher-season records: {len(ps):,}')
print(f'Name coverage: {ps["player_name"].str.isnumeric().mean():.1%} unnamed')

LATEST_SEASON = int(ps['season'].max())
print(f'Latest season: {LATEST_SEASON}')\
"""

C3_MD = """\
## 2. Season Leaderboard

The leaderboard ranks pitchers by Injury Risk+ for a given season. We show the
top 25 (highest risk) and bottom 25 (lowest risk) for the latest season, split
by archetype.

In the production dashboard, a dropdown would let the user select the season
and filter by team, archetype, or age band. Here we prototype the static
version with the full pitcher population.\
"""

C3 = """\
def build_leaderboard_fig(df: pd.DataFrame, season: int, archetype: str | None = None) -> go.Figure:
    sub = df[df['season'] == season].copy()
    if archetype:
        sub = sub[sub['archetype'] == archetype]
    sub = sub.sort_values('ir_plus', ascending=False).reset_index(drop=True)

    top25  = sub.head(25)
    bot25  = sub.tail(25).sort_values('ir_plus')

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            f'Top 25 Highest Risk — {season}' + (f' ({archetype})' if archetype else ''),
            f'Top 25 Lowest Risk — {season}' + (f' ({archetype})' if archetype else ''),
        ),
        horizontal_spacing=0.12,
    )

    fig.add_trace(
        go.Bar(
            x=top25['ir_plus'], y=top25['player_name'],
            orientation='h', marker_color='crimson',
            text=top25['ir_plus'].map('{:.1f}'.format),
            textposition='outside',
            name='Highest Risk',
        ),
        row=1, col=1,
    )
    fig.add_trace(
        go.Bar(
            x=bot25['ir_plus'], y=bot25['player_name'],
            orientation='h', marker_color='steelblue',
            text=bot25['ir_plus'].map('{:.1f}'.format),
            textposition='outside',
            name='Lowest Risk',
        ),
        row=1, col=2,
    )

    fig.add_vline(x=100, line_dash='dash', line_color='gray', row=1, col=1)
    fig.add_vline(x=100, line_dash='dash', line_color='gray', row=1, col=2)

    fig.update_layout(
        title_text=f'Injury Risk+ Leaderboard — {season}',
        height=700, width=1100,
        showlegend=False,
        font=dict(size=11),
    )
    fig.update_xaxes(title_text='Injury Risk+')
    return fig

lb_fig = build_leaderboard_fig(ps, LATEST_SEASON)
out = FIGURES_DIR / 'leaderboard.html'
lb_fig.write_html(str(out))
print(f'Saved {out}')

# Stats summary
season_df = ps[ps['season'] == LATEST_SEASON]
print(f'{LATEST_SEASON} season: {len(season_df)} pitchers | mean IR+ = {season_df["ir_plus"].mean():.1f}')
print(f'  Starters: {(season_df["archetype"]=="starter").sum()} | '
      f'Relievers: {(season_df["archetype"]=="reliever").sum()}')
print(season_df[['player_name','archetype','ir_plus','percentile']].head(10).to_string(index=False))\
"""

C4_MD = """\
## 3. Pitcher Lookup

The pitcher lookup shows a single pitcher's Injury Risk+ score alongside the
archetype-average for context. We display all seasons for that pitcher, with a
horizontal reference band marking the archetype average ± 1 standard deviation.

In the production dashboard, the pitcher would be selected from a searchable
dropdown. Here we choose the pitcher with the most career seasons as a
representative example.\
"""

C4 = """\
def lookup_pitcher(df: pd.DataFrame, pitcher_id: int) -> go.Figure:
    rows = df[df['pitcher_id'] == pitcher_id].sort_values('season')
    if rows.empty:
        raise ValueError(f'No data for pitcher_id={pitcher_id}')

    name = rows['player_name'].iloc[-1]
    arch = rows['archetype'].mode().iloc[0]

    arch_stats = (
        df[df['archetype'] == arch]
        .groupby('season')['ir_plus']
        .agg(['mean', 'std'])
        .reset_index()
    )
    arch_stats.columns = ['season', 'arch_mean', 'arch_std']
    arch_stats['arch_lo'] = arch_stats['arch_mean'] - arch_stats['arch_std']
    arch_stats['arch_hi'] = arch_stats['arch_mean'] + arch_stats['arch_std']

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=arch_stats['season'].tolist() + arch_stats['season'].tolist()[::-1],
        y=arch_stats['arch_hi'].tolist() + arch_stats['arch_lo'].tolist()[::-1],
        fill='toself', fillcolor='rgba(100,100,200,0.12)',
        line=dict(color='rgba(0,0,0,0)'),
        name=f'{arch.capitalize()} ±1 SD',
        showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=arch_stats['season'], y=arch_stats['arch_mean'],
        mode='lines', line=dict(color='rgba(100,100,200,0.5)', dash='dot'),
        name=f'{arch.capitalize()} avg',
    ))
    fig.add_trace(go.Scatter(
        x=rows['season'], y=rows['ir_plus'],
        mode='lines+markers', line=dict(color='crimson', width=2.5),
        marker=dict(size=8), name=name,
        text=rows['ir_plus'].map('{:.1f}'.format),
        textposition='top center',
    ))
    fig.add_hline(y=100, line_dash='dash', line_color='gray',
                  annotation_text='League avg (100)')
    fig.update_layout(
        title=f'Pitcher Lookup: {name} — Injury Risk+ by Season',
        xaxis_title='Season', yaxis_title='Injury Risk+',
        height=500, width=900, legend=dict(x=0.01, y=0.99),
    )
    return fig, rows

# Pick pitcher with most seasons as representative example.
career_seasons = ps.groupby('pitcher_id')['season'].nunique()
sample_pid = int(career_seasons.idxmax())
lookup_fig, lookup_rows = lookup_pitcher(ps, sample_pid)

out = FIGURES_DIR / 'pitcher_lookup.html'
lookup_fig.write_html(str(out))
print(f'Saved {out}')
print(f'Sample pitcher: {lookup_rows["player_name"].iloc[0]}  (id={sample_pid})')
print(lookup_rows[['season','archetype','ir_plus','percentile']].to_string(index=False))\
"""

C5_MD = """\
## 4. Injury Risk+ Trend Explorer

The trend explorer overlays multiple pitchers on a single chart for comparison.
We select four representative pitchers: the one with the highest career peak
IR+, the one with the lowest, and two with the longest careers.

In the production dashboard, users would select pitchers from a multi-select
dropdown and optionally overlay IL stints from the injury data.\
"""

C5 = """\
def build_trend_fig(df: pd.DataFrame, pitcher_ids: list[int]) -> go.Figure:
    fig = go.Figure()
    colors = px.colors.qualitative.Plotly

    for i, pid in enumerate(pitcher_ids):
        rows = df[df['pitcher_id'] == pid].sort_values('season')
        if rows.empty:
            continue
        name = rows['player_name'].iloc[-1]
        color = colors[i % len(colors)]
        fig.add_trace(go.Scatter(
            x=rows['season'], y=rows['ir_plus'],
            mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(size=7),
            name=name,
        ))

    fig.add_hline(y=100, line_dash='dash', line_color='gray',
                  annotation_text='League avg (100)')
    fig.update_layout(
        title='Injury Risk+ Career Trends — Sample Pitchers',
        xaxis_title='Season', yaxis_title='Injury Risk+',
        height=500, width=950,
        legend=dict(x=0.01, y=0.99),
    )
    return fig

# Select 4 interesting pitchers: highest peak, lowest trough, and 2 long careers.
peak_pid   = int(ps.loc[ps['ir_plus'].idxmax(), 'pitcher_id'])
trough_pid = int(ps.loc[ps['ir_plus'].idxmin(), 'pitcher_id'])
long_pids  = career_seasons.nlargest(4).index.tolist()
pids_to_show = list(dict.fromkeys([peak_pid, trough_pid] + long_pids))[:4]

trend_fig = build_trend_fig(ps, pids_to_show)
out = FIGURES_DIR / 'trend_explorer.html'
trend_fig.write_html(str(out))
print(f'Saved {out}')

names = [ps[ps['pitcher_id']==p]['player_name'].iloc[0] for p in pids_to_show if not ps[ps['pitcher_id']==p].empty]
print(f'Trend comparison: {names}')\
"""

C6_MD = """\
## 5. Component Breakdown View

The component breakdown decomposes the Injury Risk+ score into its three
contributing signals for a given pitcher and season:
- **Injury probability (30d)** — XGBoost classifier output (50% weight)
- **Expected days lost** — regression head from multitask model (30% weight)
- **Survival hazard rate** — Cox/RSF-derived hazard proxy (20% weight)

This view helps users understand *why* a pitcher has a high or low score, not
just what their score is.\
"""

C6 = """\
def component_breakdown_fig(df: pd.DataFrame, pitcher_id: int, season: int) -> go.Figure:
    row = df[(df['pitcher_id'] == pitcher_id) & (df['season'] == season)]
    if row.empty:
        raise ValueError(f'No data for pitcher_id={pitcher_id}, season={season}')
    row = row.iloc[0]
    name = row['player_name']

    league_row = df[df['season'] == season].mean(numeric_only=True)

    components = ['injury_prob_30d', 'expected_days_lost', 'hazard_rate']
    labels = ['Injury Prob (30d)', 'Expected Days Lost', 'Hazard Rate']
    weights = [0.50, 0.30, 0.20]
    pitcher_vals = [float(row[c]) if pd.notna(row[c]) else 0.0 for c in components]
    league_vals  = [float(league_row[c]) if pd.notna(league_row[c]) else 0.0 for c in components]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=name, x=labels, y=pitcher_vals,
        marker_color='crimson', opacity=0.85,
        text=[f'{v:.3f}' for v in pitcher_vals], textposition='outside',
    ))
    fig.add_trace(go.Bar(
        name=f'League avg ({season})', x=labels, y=league_vals,
        marker_color='steelblue', opacity=0.6,
        text=[f'{v:.3f}' for v in league_vals], textposition='outside',
    ))

    for label, w in zip(labels, weights):
        fig.add_annotation(
            x=label, y=0, text=f'wt={w:.0%}',
            showarrow=False, yshift=-25, font=dict(size=10, color='gray'),
        )

    fig.update_layout(
        barmode='group',
        title=f'Component Breakdown: {name} — {season} (IR+ = {row["ir_plus"]:.1f})',
        yaxis_title='Component Value',
        height=480, width=850,
        legend=dict(x=0.01, y=0.99),
    )
    return fig

# Use the sample pitcher from the lookup section, most recent season.
sample_season = int(lookup_rows['season'].max())
comp_fig = component_breakdown_fig(ps, sample_pid, sample_season)
out = FIGURES_DIR / 'component_breakdown.html'
comp_fig.write_html(str(out))
print(f'Saved {out}')

sample_row = ps[(ps['pitcher_id'] == sample_pid) & (ps['season'] == sample_season)].iloc[0]
print(f'{sample_row["player_name"]} {sample_season}: IR+ = {sample_row["ir_plus"]:.1f}')
print(f'  injury_prob_30d   = {sample_row["injury_prob_30d"]:.4f}')
print(f'  expected_days_lost= {sample_row["expected_days_lost"]:.2f}')
print(f'  hazard_rate       = {sample_row["hazard_rate"]:.4f}')\
"""

C7_MD = """\
## 6. Archetype Comparison Panel

A final panel shows mean IR+ and component values by archetype across all
seasons, giving the user a baseline sense of how starters and relievers differ
in modeled risk. This is the "compare to context" view referenced in the
pitcher lookup.\
"""

C7 = """\
arch_agg = (
    ps.groupby(['archetype', 'season'])
    .agg(
        mean_ir_plus=('ir_plus', 'mean'),
        mean_prob=('injury_prob_30d', 'mean'),
        mean_days=('expected_days_lost', 'mean'),
        n=('pitcher_id', 'count'),
    )
    .reset_index()
)

fig = make_subplots(
    rows=1, cols=2,
    subplot_titles=['Mean Injury Risk+ by Season', 'Mean 30-Day Injury Probability'],
    horizontal_spacing=0.10,
)
colors_by_arch = {'starter': 'steelblue', 'reliever': 'darkorange'}

for arch, grp in arch_agg.groupby('archetype'):
    color = colors_by_arch.get(arch, 'gray')
    fig.add_trace(
        go.Scatter(x=grp['season'], y=grp['mean_ir_plus'],
                   mode='lines+markers', name=arch.capitalize(),
                   line=dict(color=color, width=2), marker=dict(size=6),
                   legendgroup=arch, showlegend=True),
        row=1, col=1,
    )
    fig.add_trace(
        go.Scatter(x=grp['season'], y=grp['mean_prob'] * 100,
                   mode='lines+markers', name=arch.capitalize(),
                   line=dict(color=color, width=2), marker=dict(size=6),
                   legendgroup=arch, showlegend=False),
        row=1, col=2,
    )

fig.add_hline(y=100, line_dash='dash', line_color='gray', row=1, col=1)
fig.update_layout(
    title='Archetype Comparison: Starters vs. Relievers',
    height=450, width=1000, legend=dict(x=0.01, y=0.99),
)
fig.update_xaxes(title_text='Season')
fig.update_yaxes(title_text='Mean IR+', row=1, col=1)
fig.update_yaxes(title_text='Mean 30d Injury Prob (%)', row=1, col=2)

out = FIGURES_DIR / 'archetype_comparison.html'
fig.write_html(str(out))
print(f'Saved {out}')
print(arch_agg[arch_agg['season'] == LATEST_SEASON][['archetype','mean_ir_plus','mean_prob','n']].to_string(index=False))\
"""

C8_MD = """\
## 7. Dashboard Summary

All four prototype components have been generated as self-contained Plotly HTML
files. Each file can be opened in any browser and contains the full interactive
chart (pan, zoom, hover) without a server.

### Files Generated

| Component | File |
|-----------|------|
| Season leaderboard | `reports/figures/dashboard/leaderboard.html` |
| Pitcher lookup | `reports/figures/dashboard/pitcher_lookup.html` |
| Trend explorer | `reports/figures/dashboard/trend_explorer.html` |
| Component breakdown | `reports/figures/dashboard/component_breakdown.html` |
| Archetype comparison | `reports/figures/dashboard/archetype_comparison.html` |

### Next Steps for Production

1. Move each figure-building function into `src/dashboard/` module
2. Wrap with a Streamlit app (`streamlit run app.py`)
3. Add real-time simulation sliders (calls into `src/simulation/`)
4. Deploy to Streamlit Cloud or a hosted server
5. Add daily score refresh from latest Statcast pull\
"""

C8 = """\
html_files = list(FIGURES_DIR.glob('*.html'))
print(f'Dashboard prototype complete: {len(html_files)} HTML files in {FIGURES_DIR}')
for f in sorted(html_files):
    size_kb = f.stat().st_size / 1024
    print(f'  {f.name:<40} {size_kb:6.0f} KB')

provenance = {
    'notebook': '13_dashboard',
    'run_at': datetime.now(timezone.utc).isoformat(),
    'latest_season': LATEST_SEASON,
    'pitcher_season_pairs': len(ps),
    'html_files': [str(f.name) for f in sorted(html_files)],
}
import json
(FIGURES_DIR / 'dashboard_provenance.json').write_text(json.dumps(provenance, indent=2))
print('Notebook 13 complete.')\
"""

# ---------------------------------------------------------------------------
# ASSEMBLE
# ---------------------------------------------------------------------------

cells = [
    md(C0,    "c00-header"),
    code(C1,  "c01-setup"),
    md(C2_MD, "c02-load-header"),
    code(C2,  "c02-load"),
    md(C3_MD, "c03-leaderboard-header"),
    code(C3,  "c03-leaderboard"),
    md(C4_MD, "c04-lookup-header"),
    code(C4,  "c04-lookup"),
    md(C5_MD, "c05-trend-header"),
    code(C5,  "c05-trend"),
    md(C6_MD, "c06-breakdown-header"),
    code(C6,  "c06-breakdown"),
    md(C7_MD, "c07-archetype-header"),
    code(C7,  "c07-archetype"),
    md(C8_MD, "c08-summary-header"),
    code(C8,  "c08-summary"),
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

out = Path("notebooks/13_dashboard.ipynb")
out.parent.mkdir(exist_ok=True)
with open(out, "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Written: {out}  ({len(cells)} cells)")
