"""
workload_features.py

Computes pitcher workload features at the game and rolling-window level.
Workload is one of the strongest known predictors of pitcher injury and
must be captured across multiple time horizons.

Feature categories:
- Raw pitch counts (per game, rolling 7 / 15 / 30 / 90 days)
- Innings pitched and batters faced (rolling windows)
- Days of rest between appearances
- High-stress pitch exposure (pitches thrown at high effort / late in counts)
- Season-to-date accumulation vs. career baseline
- Workload relative to pitcher's historical norm (z-score)
"""

from __future__ import annotations

import pandas as pd


def compute_rolling_pitch_counts(
    game_df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling pitch count totals over multiple lookback windows.

    For each pitcher-game row, sums pitch counts over the preceding N days
    for each window size. Results are right-aligned (current game not included).

    Args:
        game_df: Game-level DataFrame with columns pitcher_id, game_date,
            pitch_count.
        windows: List of lookback windows in days. Defaults to [7, 15, 30, 90].

    Returns:
        game_df with new columns pitches_last_{N}d for each window N.
    """
    if windows is None:
        windows = [7, 15, 30, 90]
    raise NotImplementedError


def compute_rest_days(game_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the number of rest days before each pitching appearance.

    For each row, finds the pitcher's previous appearance and computes
    the gap in calendar days.

    Args:
        game_df: Game-level DataFrame with columns pitcher_id, game_date.

    Returns:
        game_df with a new column rest_days.
    """
    raise NotImplementedError


def compute_high_leverage_pitches(game_df: pd.DataFrame) -> pd.DataFrame:
    """Estimate the number of high-leverage or high-effort pitches per game.

    High-leverage pitches are those thrown in high-count situations (3-2),
    late innings, or beyond a threshold in the appearance. This proxy for
    stress may be more predictive than raw pitch count alone.

    Args:
        game_df: Game-level DataFrame including count-state and inning columns.

    Returns:
        game_df with new columns: high_leverage_pitches, high_leverage_pct.
    """
    raise NotImplementedError


def compute_workload_zscore(game_df: pd.DataFrame) -> pd.DataFrame:
    """Express each pitcher's recent workload relative to their career baseline.

    For each game, computes a z-score of the pitcher's 30-day rolling pitch
    count relative to their historical mean and standard deviation. A large
    positive z-score indicates an unusual workload spike.

    Args:
        game_df: Game-level DataFrame including rolling pitch count columns
            produced by compute_rolling_pitch_counts.

    Returns:
        game_df with new column workload_zscore_30d.
    """
    raise NotImplementedError


def compute_season_to_date_workload(game_df: pd.DataFrame) -> pd.DataFrame:
    """Accumulate season-to-date pitch count and innings pitched.

    Tracks cumulative season workload up to (but not including) each game.
    This captures the late-season fatigue effect.

    Args:
        game_df: Game-level DataFrame with columns pitcher_id, game_date,
            season, pitch_count, innings_pitched.

    Returns:
        game_df with new columns: season_pitches_to_date, season_ip_to_date.
    """
    raise NotImplementedError


def build_workload_features(game_df: pd.DataFrame) -> pd.DataFrame:
    """Run all workload feature computations and return the enriched DataFrame.

    Orchestrates all functions in this module in the correct order.

    Args:
        game_df: Game-level pitcher DataFrame.

    Returns:
        game_df with all workload feature columns appended.
    """
    raise NotImplementedError
