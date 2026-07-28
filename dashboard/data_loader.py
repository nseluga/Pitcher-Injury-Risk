from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent

# Notebook-11 taxonomy (notebooks/11_baseball_specific_insights.ipynb, Section 8).
# Order matters: the first matching rule wins.
ARCHETYPES = [
    "Power Starter",
    "Slider-Heavy Starter",
    "Workhorse Starter",
    "Command Starter",
    "High-Leverage Reliever",
    "Bulk Reliever",
    "Standard",
]


def _flip_name(name: str) -> str:
    """Statcast stores 'Last, First'; metadata and the UI use 'First Last'."""
    if isinstance(name, str) and "," in name:
        last, first = name.split(",", 1)
        return f"{first.strip()} {last.strip()}"
    return name


@st.cache_data
def load_player_names() -> pd.Series:
    """pitcher_id -> display name, covering every pitcher that has a score.

    player_metadata_clean is preferred because it carries correct accents and
    punctuation ("Bartolo Colón", "A. J. Burnett"), but it only covers 2,490 of
    the 3,249 scored pitchers. statcast_clean covers all of them, so it fills
    the gap. Without the fallback ~23% of pitchers rendered as a bare numeric
    id, which also broke the leaderboard's y-axis (see leaderboard.py).
    """
    names = pd.Series(dtype="object")

    meta_path = PROJECT_ROOT / "data/processed/player_metadata_clean.parquet"
    if meta_path.exists():
        meta = pd.read_parquet(meta_path, columns=["player_id", "player_name"]).dropna()
        meta = meta.drop_duplicates("player_id")
        names = meta.set_index(meta["player_id"].astype(int))["player_name"]

    statcast_path = PROJECT_ROOT / "data/processed/statcast_clean.parquet"
    if statcast_path.exists():
        sc = pd.read_parquet(statcast_path, columns=["pitcher", "player_name"]).dropna()
        sc = sc.drop_duplicates("pitcher")
        fallback = sc.set_index(sc["pitcher"].astype(int))["player_name"].map(_flip_name)
        names = names.combine_first(fallback) if len(names) else fallback

    return names


@st.cache_data
def load_archetypes() -> pd.DataFrame:
    """Pitcher-season archetype using the notebook-11 taxonomy.

    The scores parquet only carries the coarse starter/reliever split, so the
    seven real archetypes are derived here from season-average workload and
    pitch mix.

    Deviation from notebook 11: that version classified individual game rows
    against a single league-wide fastball median. League median FB velo climbs
    from 92.3 to 93.3 mph over 2015-2024, so one global baseline would tag
    recent pitchers as "power" purely from era drift. The baseline is
    per-season here, and rows are aggregated to pitcher-season before
    classifying, which is the unit the dashboard displays.
    """
    fm = pd.read_parquet(
        PROJECT_ROOT / "data/processed/feature_matrix.parquet",
        columns=["pitcher", "season", "pitch_count", "fb_velo_mean", "sl_pct"],
    )
    fm["pitcher"] = fm["pitcher"].astype(int)
    fm["season"] = fm["season"].astype(int)

    arch = fm.groupby(["pitcher", "season"], as_index=False).agg(
        pitch_count=("pitch_count", "mean"),
        fb_velo_mean=("fb_velo_mean", "mean"),
        sl_pct=("sl_pct", "mean"),
    )
    league_velo = arch.groupby("season")["fb_velo_mean"].transform("median")

    conditions = [
        (arch["pitch_count"] >= 75) & (arch["fb_velo_mean"] >= league_velo + 1.5),
        (arch["pitch_count"] >= 75) & (arch["sl_pct"] >= 0.28),
        (arch["pitch_count"] >= 80),
        (arch["pitch_count"] >= 60),
        (arch["pitch_count"] < 30) & (arch["fb_velo_mean"] >= league_velo + 1.0),
        (arch["pitch_count"] < 45),
    ]
    arch["archetype"] = np.select(conditions, ARCHETYPES[:-1], default=ARCHETYPES[-1])
    return arch.rename(columns={"pitcher": "pitcher_id"})[
        ["pitcher_id", "season", "archetype"]
    ]


@st.cache_data
def load_pitcher_season() -> pd.DataFrame:
    """Pitcher-season aggregated IR+ scores, with names and archetypes joined."""
    scores = pd.read_parquet(PROJECT_ROOT / "data/processed/injury_risk_plus_scores.parquet")
    scores["pitcher_id"] = scores["pitcher_id"].astype(int)
    scores["season"] = scores["season"].astype(int)

    agg = {
        "injury_risk_plus": "mean",
        "risk_percentile": "mean",
        "injury_prob_30d": "mean",
        "expected_days_lost": "mean",
        "hazard_rate": "mean",
        "raw_risk_score": "mean",
        "archetype": "first",
    }
    ps = (
        scores.groupby(["pitcher_id", "season"], as_index=False)
        .agg(agg)
        # The scores file's "archetype" is only starter/reliever — that is a
        # role, not an archetype. The seven real archetypes are joined below.
        .rename(
            columns={
                "injury_risk_plus": "ir_plus",
                "risk_percentile": "percentile",
                "archetype": "role",
            }
        )
    )
    ps["ir_plus"] = ps["ir_plus"].round(1)
    ps["percentile"] = ps["percentile"].round(1)

    ps = ps.merge(load_archetypes(), on=["pitcher_id", "season"], how="left")
    ps["archetype"] = ps["archetype"].fillna(ARCHETYPES[-1])

    ps["player_name"] = ps["pitcher_id"].map(load_player_names())
    ps["player_name"] = ps["player_name"].fillna(ps["pitcher_id"].astype(str))

    # 19 pairs of distinct pitchers share a name (two Luis Castillos, two Justin
    # Wilsons...). The profile and comparison panels select rows by name, so
    # without this their careers would be merged into a single series.
    collides = ps.groupby("player_name")["pitcher_id"].transform("nunique") > 1
    ps.loc[collides, "player_name"] += " (" + ps.loc[collides, "pitcher_id"].astype(str) + ")"

    return ps


@st.cache_data
def load_workload_by_season() -> pd.DataFrame:
    """Per-season workload summary for each pitcher (mean rolling loads)."""
    fm = pd.read_parquet(
        PROJECT_ROOT / "data/processed/feature_matrix.parquet",
        columns=["pitcher", "season", "pitches_7d", "pitches_28d", "pitches_90d",
                 "acwr_7_28", "pitches_season_to_date", "player_name"],
    )
    fm["pitcher"] = fm["pitcher"].astype(int)
    fm["season"] = fm["season"].astype(int)
    wl = (
        fm.groupby(["pitcher", "season"], as_index=False)
        .agg(
            pitches_7d=("pitches_7d", "mean"),
            pitches_28d=("pitches_28d", "mean"),
            pitches_90d=("pitches_90d", "mean"),
            acwr_7_28=("acwr_7_28", "mean"),
            total_pitches=("pitches_season_to_date", "max"),
            player_name=("player_name", "first"),
        )
        .rename(columns={"pitcher": "pitcher_id"})
    )
    return wl


@st.cache_data
def player_name_map() -> dict[int, str]:
    ps = load_pitcher_season()
    return dict(zip(ps["pitcher_id"], ps["player_name"]))
