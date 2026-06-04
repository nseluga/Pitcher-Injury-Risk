"""
pitch_mix_features.py

Computes features describing a pitcher's pitch mix composition and how
it changes over time. Pitch mix shifts can reflect strategic changes,
fatigue, mechanical adjustments, or attempts to hide injury.

Feature categories:
- Usage rate by pitch type (per game and rolling)
- Rolling change in pitch mix (delta vs. 30-day baseline)
- Pitch mix diversity index (entropy of usage distribution)
- High-stress pitch type ratios (e.g. slider rate — associated with elbow risk)
- Shift detection: statistically significant changes in mix within a season
"""

from __future__ import annotations

import pandas as pd


def compute_game_pitch_mix(pitch_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-game pitch type usage rates for each pitcher.

    For each pitcher-game, calculates the proportion of pitches thrown for
    each pitch type: FF, SL, CH, CU, SI, FC, etc.

    Args:
        pitch_df: Pitch-level DataFrame with columns pitcher_id, game_date,
            pitch_type.

    Returns:
        Game-level DataFrame with columns usage_{pitch_type} for each type.
    """
    raise NotImplementedError


def compute_rolling_pitch_mix(
    game_df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling average pitch mix over multiple lookback windows.

    Args:
        game_df: Game-level pitch mix DataFrame from compute_game_pitch_mix.
        windows: Lookback windows in days. Defaults to [15, 30].

    Returns:
        game_df with rolling mix columns appended.
    """
    if windows is None:
        windows = [15, 30]
    raise NotImplementedError


def compute_pitch_mix_delta(game_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the deviation of current pitch mix from the pitcher's season baseline.

    Large deviations may indicate strategic adjustments or attempts to compensate
    for injury. Computes the absolute and signed change in each pitch type's
    usage rate versus the season-to-date average.

    Args:
        game_df: Game-level pitch mix DataFrame with rolling mix columns.

    Returns:
        game_df with delta columns appended per pitch type.
    """
    raise NotImplementedError


def compute_pitch_mix_entropy(game_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the Shannon entropy of the per-game pitch mix.

    A higher entropy indicates a more diverse arsenal; a lower entropy
    indicates heavy reliance on one or two pitch types. This can reflect
    pitcher archetype and may interact with injury risk.

    Args:
        game_df: Game-level pitch mix DataFrame.

    Returns:
        game_df with a new column pitch_mix_entropy.
    """
    raise NotImplementedError


def flag_slider_heavy_usage(
    game_df: pd.DataFrame,
    threshold: float = 0.35,
) -> pd.DataFrame:
    """Flag games where the pitcher threw an unusually high rate of sliders.

    Slider overuse is associated with elevated UCL stress. This flag can be
    used as an interaction feature with workload and velocity features.

    Args:
        game_df: Game-level pitch mix DataFrame.
        threshold: Slider usage rate above which the flag is set. Default 0.35.

    Returns:
        game_df with a new boolean column slider_heavy.
    """
    raise NotImplementedError


def build_pitch_mix_features(pitch_df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrate all pitch mix feature computations.

    Args:
        pitch_df: Raw pitch-level Statcast DataFrame.

    Returns:
        Game-level DataFrame with all pitch mix feature columns.
    """
    raise NotImplementedError
