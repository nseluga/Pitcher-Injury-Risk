"""
workload_simulator.py

Simulates alternative pitcher workload schedules to estimate their
effect on injury risk. Given a pitcher's feature profile, answers
counterfactual questions like:

  "What would this pitcher's injury probability be if he threw 15 fewer
   pitches per start?"

  "How does injury risk change if rest between starts is increased
   from 4 days to 5 days?"

This is the foundation for data-driven pitch count optimization and
rest schedule optimization research.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _predict_prob(profile: pd.Series, model_dict: dict) -> float:
    """Single-row inference via the XGBoost pipeline in model_dict."""
    pipeline = model_dict["xgb_pipeline"]
    feature_cols = model_dict["feature_cols"]
    row_df = pd.DataFrame(
        [profile[feature_cols].values], columns=feature_cols
    )
    return float(pipeline.predict_proba(row_df)[0, 1])


def simulate_pitch_count_reduction(
    pitcher_profile: pd.Series,
    reduction_pitches: int,
    model_dict: dict,
) -> dict[str, float]:
    """Estimate the change in injury risk from reducing pitch count per game.

    Modifies pitch_count and proportionally scales rolling pitch totals
    (pitches_7d, pitches_28d, pitches_90d) to represent consistent lower usage,
    then reruns model inference.

    Args:
        pitcher_profile: A single row of the feature matrix (pd.Series).
        reduction_pitches: Number of pitches to remove from the game.
        model_dict: Must contain 'xgb_pipeline' and 'feature_cols'.

    Returns:
        Dictionary with original_prob, counterfactual_prob, risk_reduction_pct.
    """
    orig_prob = _predict_prob(pitcher_profile, model_dict)
    mod = pitcher_profile.copy()

    orig_count = float(mod.get("pitch_count", 95.0) or 95.0)
    new_count = max(30.0, orig_count - reduction_pitches)
    scale = new_count / orig_count if orig_count > 0 else 1.0

    mod["pitch_count"] = new_count
    for col in ("pitches_7d", "pitches_28d", "pitches_90d", "pitches_season_to_date"):
        if col in mod.index and pd.notna(mod[col]):
            mod[col] = float(mod[col]) * scale

    cf_prob = _predict_prob(mod, model_dict)
    pct = (orig_prob - cf_prob) / max(orig_prob, 1e-9) * 100.0

    return {
        "original_prob": orig_prob,
        "counterfactual_prob": cf_prob,
        "risk_reduction_pct": pct,
    }


def simulate_rest_schedule(
    pitcher_profile: pd.Series,
    rest_days: int,
    model_dict: dict,
) -> dict[str, float]:
    """Estimate the change in injury risk from altering days of rest.

    Sets days_rest to the target value, holding all other features constant.

    Args:
        pitcher_profile: A single row of the feature matrix.
        rest_days: Target number of rest days to simulate.
        model_dict: Must contain 'xgb_pipeline' and 'feature_cols'.

    Returns:
        Dictionary with original_prob, counterfactual_prob, risk_reduction_pct.
    """
    orig_prob = _predict_prob(pitcher_profile, model_dict)
    mod = pitcher_profile.copy()
    mod["days_rest"] = float(rest_days)

    cf_prob = _predict_prob(mod, model_dict)
    pct = (orig_prob - cf_prob) / max(orig_prob, 1e-9) * 100.0

    return {
        "original_prob": orig_prob,
        "counterfactual_prob": cf_prob,
        "risk_reduction_pct": pct,
        "rest_days": rest_days,
    }


def simulate_season_workload_path(
    pitcher_id: int,
    season: int,
    modified_schedule: pd.DataFrame,
    master_df: pd.DataFrame,
    model_dict: dict,
) -> pd.DataFrame:
    """Simulate an entire season under a modified workload schedule.

    Given a hypothetical game-by-game workload plan, computes the rolling
    risk trajectory across the full season.

    Args:
        pitcher_id: Pitcher identifier.
        season: Season year.
        modified_schedule: DataFrame with one row per game containing the
            hypothetical workload parameters.
        master_df: Original master DataFrame for feature baseline.
        model_dict: Must contain 'xgb_pipeline' and 'feature_cols'.

    Returns:
        DataFrame with game_date, original_risk, simulated_risk, risk_delta.
    """
    pitcher_df = master_df[
        (master_df["pitcher"] == pitcher_id) & (master_df["season"] == season)
    ].sort_values("game_date").reset_index(drop=True)

    rows = []
    for i, game_row in pitcher_df.iterrows():
        orig_prob = _predict_prob(game_row, model_dict)
        if i < len(modified_schedule):
            mod = game_row.copy()
            for col, val in modified_schedule.iloc[i].items():
                if col in mod.index:
                    mod[col] = val
            sim_prob = _predict_prob(mod, model_dict)
        else:
            sim_prob = orig_prob
        rows.append({
            "game_date": game_row.get("game_date"),
            "original_risk": orig_prob,
            "simulated_risk": sim_prob,
            "risk_delta": sim_prob - orig_prob,
        })

    return pd.DataFrame(rows)


def find_optimal_pitch_count(
    pitcher_profile: pd.Series,
    model_dict: dict,
    search_range: tuple[int, int] = (50, 120),
) -> dict:
    """Find the pitch count that minimizes injury risk for a given pitcher profile.

    Performs a grid search over the specified pitch count range in steps of 5,
    simulating each workload level and returning the optimal count.

    Args:
        pitcher_profile: Feature profile for the pitcher.
        model_dict: Must contain 'xgb_pipeline' and 'feature_cols'.
        search_range: (min_pitches, max_pitches) to search.

    Returns:
        Dictionary with optimal_pitch_count, min_prob, original_prob, all_results.
    """
    orig_prob = _predict_prob(pitcher_profile, model_dict)
    orig_count = float(pitcher_profile.get("pitch_count", 95.0) or 95.0)

    results = []
    for count in range(search_range[0], search_range[1] + 1, 5):
        mod = pitcher_profile.copy()
        scale = count / orig_count if orig_count > 0 else 1.0
        mod["pitch_count"] = float(count)
        for col in ("pitches_7d", "pitches_28d"):
            if col in mod.index and pd.notna(mod[col]):
                mod[col] = float(mod[col]) * scale
        p = _predict_prob(mod, model_dict)
        results.append({"pitch_count": count, "predicted_prob": p})

    results_df = pd.DataFrame(results)
    best_row = results_df.loc[results_df["predicted_prob"].idxmin()]

    return {
        "optimal_pitch_count": int(best_row["pitch_count"]),
        "min_prob": float(best_row["predicted_prob"]),
        "original_prob": orig_prob,
        "all_results": results,
    }
