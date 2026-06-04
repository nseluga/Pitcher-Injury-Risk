"""
injury_risk_plus.py

Constructs the Injury Risk+ score — a composite, normalized metric where:
  100 = league-average injury risk for that pitcher archetype and era
  > 100 = riskier than average
  < 100 = safer than average

Design philosophy:
- Scores are era-adjusted so they are comparable across seasons
- Scores are archetype-adjusted (starters vs. relievers have different baselines)
- The score blends multiple model outputs: injury probability, time-to-injury,
  and severity (expected days lost)
- 100 is calibrated to the population mean each season to prevent score drift

Full design specification: docs/injury_risk_plus_design.md
"""

from __future__ import annotations

import pandas as pd


def compute_raw_risk_score(predictions_df: pd.DataFrame) -> pd.DataFrame:
    """Combine multi-task model predictions into a single raw risk index.

    Blends:
    - Injury probability (30-day) — primary component
    - Expected days lost (severity weight)
    - Survival hazard rate (time-to-event signal)

    Blend weights are defined in score_calibration.py and may be tuned.

    Args:
        predictions_df: DataFrame with columns: injury_prob_30d,
            expected_days_lost, hazard_rate.

    Returns:
        predictions_df with new column raw_risk_score.
    """
    raise NotImplementedError


def normalize_to_injury_risk_plus(
    raw_scores: pd.Series,
    season: int,
    archetype: str | None = None,
    reference_df: pd.DataFrame | None = None,
) -> pd.Series:
    """Scale raw risk scores so that the population mean equals 100.

    For a given season and optionally archetype, divides each raw score by
    the population mean and multiplies by 100. The reference mean can be
    provided explicitly (for inference on new data) or computed from
    reference_df.

    Args:
        raw_scores: Series of raw risk scores.
        season: Season year for era-adjustment lookup.
        archetype: If provided, normalizes within the archetype group.
        reference_df: Historical reference DataFrame used to compute the
            normalization denominator. If None, computed from raw_scores.

    Returns:
        Series of Injury Risk+ scores (mean ≈ 100).
    """
    raise NotImplementedError


def compute_seasonal_risk_plus(
    master_df: pd.DataFrame,
    model_dict: dict[str, object],
) -> pd.DataFrame:
    """Compute Injury Risk+ for all pitchers for each season in the dataset.

    Runs model inference, computes raw scores, and normalizes within each
    season-archetype group.

    Args:
        master_df: Master modeling DataFrame.
        model_dict: Dictionary of fitted model objects from multitask_models.

    Returns:
        DataFrame with columns: pitcher_id, season, injury_risk_plus,
        raw_risk_score, archetype, plus all component scores.
    """
    raise NotImplementedError


def get_top_risk_pitchers(
    risk_df: pd.DataFrame,
    season: int,
    n: int = 20,
    archetype: str | None = None,
) -> pd.DataFrame:
    """Return the N highest Injury Risk+ pitchers for a given season.

    Args:
        risk_df: Output of compute_seasonal_risk_plus.
        season: Season to filter to.
        n: Number of pitchers to return.
        archetype: Optional archetype filter.

    Returns:
        DataFrame of top-N pitchers sorted by Injury Risk+ descending.
    """
    raise NotImplementedError


def compute_risk_percentile(risk_df: pd.DataFrame) -> pd.DataFrame:
    """Attach a percentile rank (0–100) alongside each Injury Risk+ score.

    Percentile is computed within the same season and archetype group.

    Args:
        risk_df: Output of compute_seasonal_risk_plus.

    Returns:
        risk_df with new column risk_percentile.
    """
    raise NotImplementedError
