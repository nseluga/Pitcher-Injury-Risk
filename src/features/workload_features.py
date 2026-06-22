"""
workload_features.py

Computes pitcher workload features at the game and rolling-window level.
Workload is one of the strongest known predictors of pitcher injury and
must be captured across multiple time horizons.

Feature categories:
- Raw pitch counts (rolling 7 / 28 / 90 days)
- Days of rest between appearances
- Acute:chronic workload ratio (ACWR) — industry-standard spike detector
- Season-to-date pitch accumulation
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _rolling_agg(
    df: pd.DataFrame,
    col: str,
    days: int,
    func: str = "sum",
    closed: str = "left",
) -> pd.Series:
    """Rolling aggregation of col over prior `days` days per pitcher.

    Uses pandas time-based rolling on a DatetimeIndex grouped by pitcher.
    closed='left' excludes the current row — features represent state
    *before* the current appearance, which is correct for injury prediction.
    """
    df_s = df.sort_values(["pitcher", "game_date"]).set_index("game_date")
    result = (
        df_s.groupby("pitcher")[col]
        .rolling(f"{days}D", min_periods=0, closed=closed)
        .agg(func)
        .reset_index()
    )
    result.columns = ["pitcher", "game_date", "_v"]
    merged = df[["pitcher", "game_date"]].merge(result, on=["pitcher", "game_date"], how="left")
    return merged["_v"].values


def compute_rolling_pitch_counts(
    game_df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling pitch count totals over multiple lookback windows.

    For each pitcher-game row, sums pitch counts over the preceding N days
    for each window size. Results exclude the current game (closed='left').

    Args:
        game_df: Game-level DataFrame with columns pitcher (int), game_date
            (datetime), pitch_count (int).
        windows: List of lookback windows in days. Defaults to [7, 28, 90].

    Returns:
        game_df with new columns pitches_{N}d for each window N.
    """
    if windows is None:
        windows = [7, 28, 90]

    df = game_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    for w in windows:
        df[f"pitches_{w}d"] = _rolling_agg(df, "pitch_count", w, "sum")

    return df


def compute_rest_days(game_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the number of rest days before each pitching appearance.

    For each row, finds the pitcher's previous appearance and returns the
    gap in calendar days. First appearance for a pitcher returns NaN.

    Args:
        game_df: Game-level DataFrame with columns pitcher, game_date.

    Returns:
        game_df with a new column days_rest.
    """
    df = game_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["pitcher", "game_date"])
    df["days_rest"] = df.groupby("pitcher")["game_date"].diff().dt.days
    return df


def compute_season_to_date_workload(game_df: pd.DataFrame) -> pd.DataFrame:
    """Accumulate season-to-date pitch count up to (but not including) each game.

    Tracks cumulative workload within a season, capturing the late-season
    fatigue effect common in starting pitchers.

    Args:
        game_df: Game-level DataFrame with columns pitcher, game_date,
            pitch_count. A 'season' column is added internally if absent.

    Returns:
        game_df with new column pitches_season_to_date.
    """
    df = game_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    if "season" not in df.columns:
        df["season"] = df["game_date"].dt.year

    df = df.sort_values(["pitcher", "game_date"])
    # Use a 365-day rolling sum as a season proxy (handles mid-year debuts)
    df["pitches_season_to_date"] = _rolling_agg(df, "pitch_count", 365, "sum")
    return df


def compute_yoy_workload_spike(game_df: pd.DataFrame) -> pd.DataFrame:
    """Add year-over-year workload spike features.

    Computes the prior season's total pitch count for each pitcher, then
    expresses the current season's accumulation as a ratio relative to that
    baseline. A ratio >> 1 early in the season signals a workload spike that
    is a known injury risk factor (RR ≈ 2.2 per PMC8721392).

    Requires 'pitches_season_to_date' to already be present in game_df.
    No leakage: prior_year_pitches is the FULL prior-season total, which is
    entirely in the past for any game in the current season.

    Args:
        game_df: Game-level DataFrame with pitcher, game_date, pitch_count,
            and pitches_season_to_date.

    Returns:
        game_df with new columns prior_year_pitches and yoy_workload_ratio.
    """
    df = game_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    if "season" not in df.columns:
        df["season"] = df["game_date"].dt.year

    # Total pitches per pitcher per season (full season, all games)
    season_totals = (
        df.groupby(["pitcher", "season"])["pitch_count"]
        .sum()
        .reset_index()
        .rename(columns={"pitch_count": "prior_year_pitches", "season": "prior_season"})
    )
    # Shift forward by 1 year so we can join onto current-season rows
    season_totals["season"] = season_totals["prior_season"] + 1

    df = df.merge(
        season_totals[["pitcher", "season", "prior_year_pitches"]],
        on=["pitcher", "season"],
        how="left",
    )

    # Ratio: how far into a repeat-workload season is this pitcher, relative
    # to what they threw all of last year?  NaN for debut seasons (no prior year).
    df["yoy_workload_ratio"] = (
        df["pitches_season_to_date"] / df["prior_year_pitches"].clip(lower=1)
    ).round(4)

    return df


def build_workload_features(game_df: pd.DataFrame) -> pd.DataFrame:
    """Run all workload feature computations and return the enriched DataFrame.

    Computes rolling pitch counts (7/28/90 days), rest days, ACWR,
    season-to-date accumulation, and year-over-year workload spike.

    Args:
        game_df: Game-level pitcher DataFrame with pitcher, game_date,
            pitch_count columns.

    Returns:
        game_df with all workload feature columns appended.
    """
    df = game_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Rolling pitch counts
    df = compute_rolling_pitch_counts(df, windows=[7, 28, 90])

    # Rest days
    df = compute_rest_days(df)

    # ACWR: acute (7d) / chronic_weekly (28d ÷ 4)
    # Clamp denominator to 1 to avoid division by zero for first appearances.
    chronic_weekly = (df["pitches_28d"] / 4).clip(lower=1)
    df["acwr_7_28"] = (df["pitches_7d"] / chronic_weekly).round(4)

    # Season-to-date
    df = compute_season_to_date_workload(df)

    # YoY workload spike
    df = compute_yoy_workload_spike(df)

    return df
