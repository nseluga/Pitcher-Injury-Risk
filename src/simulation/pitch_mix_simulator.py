"""
pitch_mix_simulator.py

Simulates changes to a pitcher's pitch mix and estimates the resulting
change in injury risk. Explores research questions such as:

  "Does reducing slider usage lower long-term injury risk?"
  "What pitch mix minimizes risk for a power sinker-baller?"
  "Can certain pitch mix profiles reduce UCL stress while preserving
   effectiveness?"

Approach: modify pitch mix features in the pitcher's profile vector,
propagate through the model pipeline, and compare Injury Risk+ scores.
"""

from __future__ import annotations

import pandas as pd


def simulate_pitch_mix_change(
    pitcher_profile: pd.Series,
    new_mix: dict[str, float],
    model_dict: dict[str, object],
) -> dict[str, float]:
    """Estimate injury risk under a hypothetical pitch mix.

    Args:
        pitcher_profile: A single row of the master DataFrame (one game).
        new_mix: Dictionary mapping pitch type to desired usage rate.
            Must sum to 1.0. Example: {'FF': 0.50, 'SL': 0.20, 'CH': 0.30}.
        model_dict: Fitted model objects from multitask_models.

    Returns:
        Dictionary with keys: original_risk_plus, counterfactual_risk_plus,
        risk_reduction_pct.
    """
    raise NotImplementedError


def simulate_slider_reduction(
    pitcher_profile: pd.Series,
    target_slider_rate: float,
    model_dict: dict[str, object],
) -> dict[str, float]:
    """Estimate injury risk change from reducing slider usage to a target rate.

    Redistributes the removed slider usage proportionally to the pitcher's
    other pitch types to maintain a valid probability distribution.

    Args:
        pitcher_profile: Feature profile for the pitcher.
        target_slider_rate: Target slider usage rate (0.0–1.0).
        model_dict: Fitted model objects.

    Returns:
        Dictionary with original and counterfactual risk scores plus
        the redistributed pitch mix.
    """
    raise NotImplementedError


def find_risk_minimizing_pitch_mix(
    pitcher_profile: pd.Series,
    available_pitch_types: list[str],
    model_dict: dict[str, object],
    n_samples: int = 1000,
) -> dict[str, object]:
    """Search for the pitch mix that minimizes injury risk for a given pitcher.

    Uses Monte Carlo sampling of valid pitch mixes (simplex sampling) and
    returns the mix with the lowest predicted Injury Risk+.

    Args:
        pitcher_profile: Feature profile for the pitcher.
        available_pitch_types: List of pitch types the pitcher can throw.
        model_dict: Fitted model objects.
        n_samples: Number of random mixes to evaluate.

    Returns:
        Dictionary with keys: optimal_mix (dict), min_risk_plus,
        original_risk_plus, risk_reduction_pct.
    """
    raise NotImplementedError


def compute_pitch_type_risk_sensitivity(
    pitcher_profile: pd.Series,
    model_dict: dict[str, object],
    delta: float = 0.05,
) -> pd.DataFrame:
    """Compute the sensitivity of Injury Risk+ to each pitch type's usage rate.

    For each pitch type, increases usage by delta (and decreases another
    proportionally) and records the change in risk. This is a local
    sensitivity / gradient analysis.

    Args:
        pitcher_profile: Feature profile for the pitcher.
        model_dict: Fitted model objects.
        delta: Usage rate change to apply per pitch type.

    Returns:
        DataFrame with columns: pitch_type, risk_sensitivity (change in
        Injury Risk+ per unit change in usage rate).
    """
    raise NotImplementedError
