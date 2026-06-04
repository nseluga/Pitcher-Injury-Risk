"""
velocity_features.py

Extracts velocity-based features from Statcast pitch-level data.
Velocity changes — particularly spikes and declines — have been associated
with arm stress and are a key component of injury risk modeling.

Feature categories:
- Mean and max fastball velocity per game
- Rolling velocity trends (7 / 15 / 30 day windows)
- Velocity change vs. season average (delta)
- Velocity spikes (single-game jumps above threshold)
- Velocity decline curves (late-season decay patterns)
- Velocity differential between pitch types (FB vs. breaking ball gap)
- Intra-game velocity drop (first inning vs. last inning)
"""

from __future__ import annotations

import pandas as pd


def compute_game_velocity_stats(pitch_df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, max, min, and std of release_speed per game per pitcher.

    Aggregates pitch-level velocity data to the game grain, split by pitch
    type so fastball, slider, and changeup velocities are tracked separately.

    Args:
        pitch_df: Pitch-level DataFrame with columns pitcher_id, game_date,
            pitch_type, release_speed.

    Returns:
        Game-level DataFrame with velocity summary columns per pitch type.
    """
    raise NotImplementedError


def compute_rolling_velocity(
    game_df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling average fastball velocity over multiple time windows.

    Args:
        game_df: Game-level velocity DataFrame from compute_game_velocity_stats.
        windows: List of lookback windows in days. Defaults to [7, 15, 30].

    Returns:
        game_df with new columns velo_fb_mean_last_{N}d for each window N.
    """
    if windows is None:
        windows = [7, 15, 30]
    raise NotImplementedError


def compute_velocity_delta(game_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the change in fastball velocity relative to the season average.

    For each game, subtracts the pitcher's season-to-date mean velocity from
    the current game's velocity to capture deviations from baseline.

    Args:
        game_df: Game-level velocity DataFrame with rolling velocity columns.

    Returns:
        game_df with new column velo_delta_vs_season_avg.
    """
    raise NotImplementedError


def flag_velocity_spikes(
    game_df: pd.DataFrame,
    spike_threshold_mph: float = 2.0,
) -> pd.DataFrame:
    """Flag games where a pitcher's velocity exceeded their rolling mean by a threshold.

    Velocity spikes may indicate unusual arm effort. This feature captures
    single-game outliers that might otherwise be smoothed away in rolling averages.

    Args:
        game_df: Game-level velocity DataFrame.
        spike_threshold_mph: MPH above the 30-day rolling mean to qualify as
            a spike. Default is 2.0 mph.

    Returns:
        game_df with new boolean column velocity_spike.
    """
    raise NotImplementedError


def compute_intragame_velocity_drop(pitch_df: pd.DataFrame) -> pd.DataFrame:
    """Measure velocity loss within a single appearance (first vs. last inning).

    Computes the difference between mean fastball velocity in the first inning
    and the last inning of each appearance, as a proxy for fatigue within a game.

    Args:
        pitch_df: Pitch-level DataFrame with inning and velocity columns.

    Returns:
        Game-level DataFrame with new column intragame_velo_drop.
    """
    raise NotImplementedError


def build_velocity_features(pitch_df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrate all velocity feature computations.

    Args:
        pitch_df: Raw pitch-level Statcast DataFrame.

    Returns:
        Game-level DataFrame with all velocity feature columns.
    """
    raise NotImplementedError
