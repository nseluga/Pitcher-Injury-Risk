"""
velocity_features.py

Extracts velocity-based features from Statcast pitch-level data.
Velocity changes — particularly declines — have been associated with arm
stress and are a key component of injury risk modeling.

Feature categories:
- Mean and max fastball velocity per game
- Rolling velocity trends (7 / 30 / 90 day windows)
- Velocity change vs. rolling baseline (delta features)
- Intra-game velocity drop (first vs. last inning)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FASTBALL_TYPES = {"FF", "SI", "FC"}
SLIDER_TYPES = {"SL", "ST"}


def compute_game_velocity_stats(pitch_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-game fastball velocity summary statistics per pitcher.

    Aggregates pitch-level velocity data to the game grain, tracking fastball
    mean, max, and std. Non-fastball pitches are excluded so the metric
    reflects arm speed rather than pitch selection.

    Args:
        pitch_df: Pitch-level DataFrame with columns pitcher (int), game_date
            (datetime), pitch_type (str), release_speed (float).

    Returns:
        Game-level DataFrame with columns:
            pitcher, game_date, fb_velo_mean, fb_velo_max, fb_velo_std,
            fb_count, all_velo_mean.
    """
    df = pitch_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Fastball-only subset
    fb = df[df["pitch_type"].isin(FASTBALL_TYPES)]

    fb_stats = (
        fb.groupby(["pitcher", "game_date"])["release_speed"]
        .agg(fb_velo_mean="mean", fb_velo_max="max", fb_velo_std="std",
             fb_count="count")
        .reset_index()
    )

    # All-pitch mean (for context)
    all_stats = (
        df.groupby(["pitcher", "game_date"])["release_speed"]
        .mean()
        .reset_index(name="all_velo_mean")
    )

    # Intra-game velocity drop: first two innings vs. last two innings.
    # Per-game max inning is used so starters exiting in inning 5-6 still get
    # a valid reading. Using global max() here was a bug — most starters pitch
    # fewer innings than the full-game max, so "late" would be empty for them.
    inning_col = "inning" if "inning" in df.columns else None
    if inning_col:
        game_max_inning = (
            df.groupby(["pitcher", "game_date"])["inning"]
            .max()
            .reset_index(name="_game_max_inning")
        )
        df_inning = df.merge(game_max_inning, on=["pitcher", "game_date"], how="left")

        early = df_inning[df_inning[inning_col] <= 2]
        late  = df_inning[df_inning[inning_col] >= (df_inning["_game_max_inning"] - 1)]
        early_velo = (
            early[early["pitch_type"].isin(FASTBALL_TYPES)]
            .groupby(["pitcher", "game_date"])["release_speed"]
            .mean()
            .reset_index(name="fb_velo_early_innings")
        )
        late_velo = (
            late[late["pitch_type"].isin(FASTBALL_TYPES)]
            .groupby(["pitcher", "game_date"])["release_speed"]
            .mean()
            .reset_index(name="fb_velo_late_innings")
        )
        inning_drop = early_velo.merge(late_velo, on=["pitcher", "game_date"], how="outer")
        inning_drop["intragame_velo_drop"] = (
            inning_drop["fb_velo_early_innings"] - inning_drop["fb_velo_late_innings"]
        )
    else:
        inning_drop = None

    out = fb_stats.merge(all_stats, on=["pitcher", "game_date"], how="outer")
    if inning_drop is not None:
        out = out.merge(inning_drop[["pitcher", "game_date", "intragame_velo_drop"]],
                        on=["pitcher", "game_date"], how="left")

    return out


def compute_rolling_velocity(
    game_df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling average fastball velocity over multiple time windows.

    Uses closed='left' so each row reflects velocity in the days *before*
    the current appearance — a proper leading indicator.

    Args:
        game_df: Game-level velocity DataFrame containing fb_velo_mean.
        windows: Lookback windows in days. Defaults to [7, 30, 90].

    Returns:
        game_df with new columns fb_velo_{N}d_avg for each window N.
    """
    if windows is None:
        windows = [7, 30, 90]

    if "fb_velo_mean" not in game_df.columns:
        raise ValueError("game_df must contain fb_velo_mean. "
                         "Run compute_game_velocity_stats first.")

    df = game_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df_s = df.sort_values(["pitcher", "game_date"]).set_index("game_date")

    for w in windows:
        result = (
            df_s.groupby("pitcher")["fb_velo_mean"]
            .rolling(f"{w}D", min_periods=1, closed="left")
            .mean()
            .reset_index()
        )
        result.columns = ["pitcher", "game_date", f"fb_velo_{w}d_avg"]
        df = df.merge(result, on=["pitcher", "game_date"], how="left")

    return df


def compute_velocity_delta(game_df: pd.DataFrame) -> pd.DataFrame:
    """Compute velocity change relative to rolling baselines.

    Captures velocity decline — a leading indicator of arm fatigue. Three
    delta features measure change over different time horizons:
        velo_change_7_30d:  this week vs. prior 30 days
        velo_change_30_90d: prior 30 days vs. prior 90 days

    Args:
        game_df: Game-level velocity DataFrame with rolling velocity columns
            produced by compute_rolling_velocity.

    Returns:
        game_df with velocity change columns appended.
    """
    df = game_df.copy()

    if "fb_velo_7d_avg" in df.columns and "fb_velo_30d_avg" in df.columns:
        df["velo_change_7_30d"] = df["fb_velo_7d_avg"] - df["fb_velo_30d_avg"]

    if "fb_velo_30d_avg" in df.columns and "fb_velo_90d_avg" in df.columns:
        df["velo_change_30_90d"] = df["fb_velo_30d_avg"] - df["fb_velo_90d_avg"]

    # Season-to-date baseline velocity (cumulative mean from season start)
    df_s = df.sort_values(["pitcher", "game_date"]).set_index("game_date")
    season_avg = (
        df_s.groupby("pitcher")["fb_velo_mean"]
        .expanding(min_periods=3)
        .mean()
        .reset_index()
    )
    season_avg.columns = ["pitcher", "game_date", "fb_velo_season_avg"]
    df = df.merge(season_avg, on=["pitcher", "game_date"], how="left")
    df["velo_delta_vs_season"] = df["fb_velo_mean"] - df["fb_velo_season_avg"]

    return df


def build_velocity_features(pitch_df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrate all velocity feature computations.

    Args:
        pitch_df: Raw pitch-level Statcast DataFrame with pitcher, game_date,
            pitch_type, release_speed, and optionally inning columns.

    Returns:
        Game-level DataFrame with all velocity feature columns.
    """
    game_velo = compute_game_velocity_stats(pitch_df)
    # 14-day window added alongside 7/30/90 — Logue et al. (2021) identified the
    # 15 games (~14 days for a starter) before surgery as the critical detection window.
    game_velo = compute_rolling_velocity(game_velo, windows=[7, 14, 30, 90])
    game_velo = compute_velocity_delta(game_velo)
    return game_velo
