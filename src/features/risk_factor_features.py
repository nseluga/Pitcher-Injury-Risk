"""
risk_factor_features.py

Computes aggregate and composite risk factor features that combine
information from workload, velocity, movement, pitch mix, and injury
history into higher-order signals.

This module also encodes player-level attributes (age, physical build,
pitcher archetype) and environmental context (team, ballpark, schedule
density) that modulate baseline risk.

Feature categories:
- Age and career-stage effects
- Physical attributes (height, weight, BMI from player metadata)
- Pitcher archetype encoding (starter / hybrid / reliever / closer)
- Schedule density (back-to-back appearances, road trip length)
- Composite stress index (multi-signal combination)
- Interaction features (e.g. high workload × velocity spike)
"""

from __future__ import annotations

import pandas as pd


def attach_player_attributes(
    game_df: pd.DataFrame,
    player_metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach static player attributes (age at game date, height, weight, throws).

    Age is computed dynamically based on game_date vs. birth_date so it
    changes correctly across seasons.

    Args:
        game_df: Game-level DataFrame with pitcher_id and game_date.
        player_metadata_df: Player metadata from pybaseball or MLB Stats API.

    Returns:
        game_df with new columns: age, height_in, weight_lbs, bmi, throws.
    """
    raise NotImplementedError


def encode_pitcher_archetype(game_df: pd.DataFrame) -> pd.DataFrame:
    """Classify each pitcher-season into a role archetype.

    Categories: 'starter', 'hybrid_starter', 'long_reliever', 'middle_reliever',
    'setup', 'closer'. Classification is based on appearances, innings per
    appearance, and role transition flags from transactions_loader.

    Args:
        game_df: Game-level DataFrame with appearance-level context.

    Returns:
        game_df with new column pitcher_archetype (categorical).
    """
    raise NotImplementedError


def compute_schedule_density(game_df: pd.DataFrame) -> pd.DataFrame:
    """Quantify the density of a pitcher's recent schedule.

    Features:
    - appearances_last_7d: number of outings in the past 7 days
    - road_trip_day: day number within a current road trip
    - time_zone_changes: proxy for travel fatigue across recent games

    Args:
        game_df: Game-level DataFrame with game_date and team schedule metadata.

    Returns:
        game_df with schedule density columns appended.
    """
    raise NotImplementedError


def compute_composite_stress_index(game_df: pd.DataFrame) -> pd.DataFrame:
    """Combine workload, velocity, and movement signals into a composite stress score.

    This is a hand-engineered precursor to the model-derived Injury Risk+ score.
    It serves as a useful EDA tool and baseline feature. The exact formula is
    defined in docs/injury_risk_plus_design.md.

    Args:
        game_df: Game-level DataFrame with workload, velocity, and movement
            feature columns already attached.

    Returns:
        game_df with new column composite_stress_index.
    """
    raise NotImplementedError


def compute_interaction_features(game_df: pd.DataFrame) -> pd.DataFrame:
    """Create interaction terms between high-signal feature pairs.

    Examples:
    - workload_spike × velocity_spike
    - age × rolling_pitch_count_30d
    - slider_heavy × prior_elbow_injury
    - rest_days × season_to_date_pitches

    Args:
        game_df: Game-level DataFrame with all base feature columns present.

    Returns:
        game_df with interaction feature columns appended.
    """
    raise NotImplementedError


def build_risk_factor_features(
    game_df: pd.DataFrame,
    player_metadata_df: pd.DataFrame,
) -> pd.DataFrame:
    """Orchestrate all risk factor feature computations.

    Args:
        game_df: Game-level DataFrame with workload, velocity, movement, and
            pitch mix features already attached.
        player_metadata_df: Player metadata DataFrame.

    Returns:
        game_df with all risk factor feature columns appended.
    """
    raise NotImplementedError
