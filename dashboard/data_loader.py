from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).parent.parent


@st.cache_data
def load_pitcher_season() -> pd.DataFrame:
    """Pitcher-season aggregated IR+ scores, with player names joined."""
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
        .rename(columns={"injury_risk_plus": "ir_plus", "risk_percentile": "percentile"})
    )
    ps["ir_plus"] = ps["ir_plus"].round(1)
    ps["percentile"] = ps["percentile"].round(1)

    meta_path = PROJECT_ROOT / "data/processed/player_metadata_clean.parquet"
    if meta_path.exists():
        meta = pd.read_parquet(meta_path, columns=["player_id", "player_name"])
        meta = meta.rename(columns={"player_id": "pitcher_id"})
        meta["pitcher_id"] = meta["pitcher_id"].astype(int)
        ps = ps.merge(meta, on="pitcher_id", how="left")
        ps["player_name"] = ps["player_name"].fillna(ps["pitcher_id"].astype(str))
    else:
        ps["player_name"] = ps["pitcher_id"].astype(str)

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
