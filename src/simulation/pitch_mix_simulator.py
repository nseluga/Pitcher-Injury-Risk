"""
pitch_mix_simulator.py

Simulates changes to a pitcher's pitch mix and estimates the resulting
change in injury risk. Explores research questions such as:

  "Does reducing slider usage lower long-term injury risk?"
  "What pitch mix minimizes risk for a power sinker-baller?"

Approach: modify pitch mix features in the pitcher's profile vector,
propagate through the model pipeline, and compare predicted injury probabilities.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Map pitch-type abbreviations to feature-matrix column names.
_PITCH_TYPE_COLS = {
    "FF": "fb_pct",
    "SL": "sl_pct",
    "CH": "ch_pct",
    "CU": "cu_pct",
}


def _predict_prob(profile: pd.Series, model_dict: dict) -> float:
    """Single-row inference via the XGBoost pipeline in model_dict."""
    pipeline = model_dict["xgb_pipeline"]
    feature_cols = model_dict["feature_cols"]
    row_df = pd.DataFrame(
        [profile[feature_cols].values], columns=feature_cols
    )
    return float(pipeline.predict_proba(row_df)[0, 1])


def simulate_pitch_mix_change(
    pitcher_profile: pd.Series,
    new_mix: dict[str, float],
    model_dict: dict,
) -> dict[str, float]:
    """Estimate injury risk under a hypothetical pitch mix.

    Args:
        pitcher_profile: A single row of the feature matrix.
        new_mix: Dictionary mapping pitch-type abbreviation to desired usage rate.
            Must sum to ≤ 1.0. Example: {'FF': 0.50, 'SL': 0.20, 'CH': 0.30}.
        model_dict: Must contain 'xgb_pipeline' and 'feature_cols'.

    Returns:
        Dictionary with original_prob, counterfactual_prob, risk_reduction_pct.
    """
    orig_prob = _predict_prob(pitcher_profile, model_dict)
    mod = pitcher_profile.copy()

    for pt, rate in new_mix.items():
        col = _PITCH_TYPE_COLS.get(pt)
        if col and col in mod.index:
            mod[col] = float(rate)

    cf_prob = _predict_prob(mod, model_dict)
    pct = (orig_prob - cf_prob) / max(orig_prob, 1e-9) * 100.0

    return {
        "original_prob": orig_prob,
        "counterfactual_prob": cf_prob,
        "risk_reduction_pct": pct,
    }


def simulate_slider_reduction(
    pitcher_profile: pd.Series,
    target_slider_rate: float,
    model_dict: dict,
) -> dict[str, float]:
    """Estimate injury risk change from reducing slider usage to a target rate.

    The removed slider usage is redistributed to the fastball percentage to
    keep the pitch distribution valid.

    Args:
        pitcher_profile: Feature profile for the pitcher.
        target_slider_rate: Target slider usage rate (0.0–1.0).
        model_dict: Must contain 'xgb_pipeline' and 'feature_cols'.

    Returns:
        Dictionary with original_prob, counterfactual_prob, risk_reduction_pct,
        original_sl_pct, target_sl_pct.
    """
    orig_prob = _predict_prob(pitcher_profile, model_dict)
    mod = pitcher_profile.copy()

    orig_sl = float(mod.get("sl_pct", 0.15) or 0.0)
    delta = orig_sl - target_slider_rate  # amount removed from slider

    mod["sl_pct"] = float(target_slider_rate)
    if "fb_pct" in mod.index:
        mod["fb_pct"] = float(mod.get("fb_pct", 0.5) or 0.5) + delta

    cf_prob = _predict_prob(mod, model_dict)
    pct = (orig_prob - cf_prob) / max(orig_prob, 1e-9) * 100.0

    return {
        "original_prob": orig_prob,
        "counterfactual_prob": cf_prob,
        "risk_reduction_pct": pct,
        "original_sl_pct": orig_sl,
        "target_sl_pct": target_slider_rate,
    }


def find_risk_minimizing_pitch_mix(
    pitcher_profile: pd.Series,
    available_pitch_types: list[str],
    model_dict: dict,
    n_samples: int = 1000,
) -> dict:
    """Search for the pitch mix that minimizes injury risk for a given pitcher.

    Uses Dirichlet sampling to generate valid pitch mix combinations and
    returns the mix with the lowest predicted injury probability.

    Args:
        pitcher_profile: Feature profile for the pitcher.
        available_pitch_types: List of pitch-type abbreviations (e.g. ['FF','SL','CH']).
        model_dict: Must contain 'xgb_pipeline' and 'feature_cols'.
        n_samples: Number of random mixes to evaluate.

    Returns:
        Dictionary with optimal_mix, min_prob, original_prob, risk_reduction_pct.
    """
    orig_prob = _predict_prob(pitcher_profile, model_dict)
    active_cols = [
        (pt, _PITCH_TYPE_COLS[pt])
        for pt in available_pitch_types
        if pt in _PITCH_TYPE_COLS and _PITCH_TYPE_COLS[pt] in pitcher_profile.index
    ]

    if not active_cols:
        return {
            "original_prob": orig_prob,
            "optimal_mix": {},
            "min_prob": orig_prob,
            "risk_reduction_pct": 0.0,
        }

    rng = np.random.default_rng(42)
    best_prob = orig_prob
    best_mix: dict[str, float] = {}

    for _ in range(n_samples):
        fracs = rng.dirichlet(np.ones(len(active_cols)))
        mod = pitcher_profile.copy()
        for (pt, col), frac in zip(active_cols, fracs):
            mod[col] = float(frac)
        p = _predict_prob(mod, model_dict)
        if p < best_prob:
            best_prob = p
            best_mix = {pt: float(frac) for (pt, _col), frac in zip(active_cols, fracs)}

    pct = (orig_prob - best_prob) / max(orig_prob, 1e-9) * 100.0

    return {
        "original_prob": orig_prob,
        "optimal_mix": best_mix,
        "min_prob": best_prob,
        "risk_reduction_pct": pct,
    }


def compute_pitch_type_risk_sensitivity(
    pitcher_profile: pd.Series,
    model_dict: dict,
    delta: float = 0.05,
) -> pd.DataFrame:
    """Compute sensitivity of injury probability to each pitch type's usage rate.

    For each pitch type, increases usage by delta (redistributing from fastball)
    and records the change in predicted injury probability.

    Args:
        pitcher_profile: Feature profile for the pitcher.
        model_dict: Must contain 'xgb_pipeline' and 'feature_cols'.
        delta: Usage rate change per pitch type.

    Returns:
        DataFrame with columns: pitch_type, column, original_usage, risk_sensitivity.
    """
    orig_prob = _predict_prob(pitcher_profile, model_dict)
    rows = []

    for pt, col in _PITCH_TYPE_COLS.items():
        if col not in pitcher_profile.index:
            continue
        orig_val = float(pitcher_profile.get(col, 0.0) or 0.0)
        mod = pitcher_profile.copy()
        mod[col] = min(1.0, orig_val + delta)
        # Reduce fastball to compensate (unless this IS the fastball).
        if col != "fb_pct" and "fb_pct" in mod.index:
            mod["fb_pct"] = max(0.0, float(mod.get("fb_pct", 0.5) or 0.5) - delta)
        cf_prob = _predict_prob(mod, model_dict)
        sensitivity = (cf_prob - orig_prob) / delta  # change per unit usage increase
        rows.append({
            "pitch_type": pt,
            "column": col,
            "original_usage": orig_val,
            "risk_sensitivity": sensitivity,
        })

    return pd.DataFrame(rows).sort_values("risk_sensitivity", ascending=False).reset_index(drop=True)
