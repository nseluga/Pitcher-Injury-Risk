"""
injury_history_features.py

Derives features from a pitcher's prior injury history. Prior injury is
one of the strongest known predictors of future injury, and the nature,
recency, and frequency of past injuries all carry signal.

Feature categories:
- Career injury count (total and by body part)
- Days since last injury (recency)
- Days lost in previous IL stint
- Injury recurrence flags (same body part injured again)
- Career total days lost
- Injury type one-hot encoding (elbow, shoulder, forearm, etc.)
- Injury frequency rate (injuries per 162 games or per season)
"""

from __future__ import annotations

import pandas as pd


def compute_injury_counts(
    game_df: pd.DataFrame,
    injuries_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute cumulative injury counts up to each game date for each pitcher.

    For each row in game_df, counts all prior IL stints for that pitcher —
    total and split by injury category (elbow, shoulder, forearm, etc.).

    Args:
        game_df: Game-level DataFrame with columns pitcher_id, game_date.
        injuries_df: Injury database from injury_loader.

    Returns:
        game_df with new columns: prior_il_stints_total, prior_il_elbow,
        prior_il_shoulder, prior_il_forearm, prior_il_other.
    """
    raise NotImplementedError


def compute_days_since_last_injury(
    game_df: pd.DataFrame,
    injuries_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the number of days since a pitcher's most recent IL stint.

    A pitcher who recently returned from injury may face elevated risk of
    recurrence. Returns NaN for pitchers with no prior injury history.

    Args:
        game_df: Game-level DataFrame with pitcher_id, game_date.
        injuries_df: Injury database from injury_loader.

    Returns:
        game_df with new column days_since_last_injury.
    """
    raise NotImplementedError


def compute_prior_days_lost(
    game_df: pd.DataFrame,
    injuries_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach the number of days lost in the pitcher's most recent IL stint.

    Captures severity of the most recent injury. May interact with
    days_since_last_injury as a recovery completeness proxy.

    Args:
        game_df: Game-level DataFrame.
        injuries_df: Injury database from injury_loader.

    Returns:
        game_df with new column prior_days_lost.
    """
    raise NotImplementedError


def flag_same_body_part_recurrence(
    game_df: pd.DataFrame,
    injuries_df: pd.DataFrame,
) -> pd.DataFrame:
    """Flag pitchers who have previously been injured in the same body part.

    Recurrence risk is substantially elevated for the same location. This
    flag creates an interaction term that can be combined with current
    injury-type predictions.

    Args:
        game_df: Game-level DataFrame with pitcher_id, game_date.
        injuries_df: Injury database including injury_type column.

    Returns:
        game_df with boolean columns: recurring_elbow_risk,
        recurring_shoulder_risk, recurring_forearm_risk.
    """
    raise NotImplementedError


def compute_injury_frequency_rate(
    game_df: pd.DataFrame,
    injuries_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute a pitcher's historical injury frequency rate.

    Injuries per 162 games (or per season) as a normalized rate that accounts
    for differences in career length and playing time.

    Args:
        game_df: Game-level DataFrame.
        injuries_df: Injury database from injury_loader.

    Returns:
        game_df with new column injury_frequency_rate.
    """
    raise NotImplementedError


def build_injury_history_features(
    game_df: pd.DataFrame,
    injuries_df: pd.DataFrame,
) -> pd.DataFrame:
    """Orchestrate all injury history feature computations.

    Args:
        game_df: Game-level pitcher DataFrame.
        injuries_df: Injury database from injury_loader.

    Returns:
        game_df with all injury history feature columns appended.
    """
    raise NotImplementedError
