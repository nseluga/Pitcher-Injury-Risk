"""
usage_strategy_simulator.py

High-level simulation of pitcher usage strategies at the team and
individual level. Extends workload_simulator and pitch_mix_simulator
to answer broader organizational questions:

  "Are traditional starters inherently less healthy than hybrid starters?"
  "Should some pitchers transition between starting and relieving?"
  "Can workload be distributed differently across a pitching staff?"
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


def _archetype_medians(master_df: pd.DataFrame, workload_cols: list[str]) -> dict[str, dict]:
    """Compute median workload features per archetype."""
    if "archetype" not in master_df.columns:
        return {}
    result = {}
    for arch, grp in master_df.groupby("archetype"):
        result[arch] = grp[workload_cols].median().to_dict()
    return result


def simulate_role_transition(
    pitcher_id: int,
    current_role: str,
    target_role: str,
    season: int,
    master_df: pd.DataFrame,
    model_dict: dict,
) -> dict[str, float]:
    """Estimate the change in injury risk from transitioning a pitcher's role.

    Constructs a counterfactual feature vector by replacing the pitcher's
    workload features with the target-role median workload profile, then
    compares predicted injury probability.

    Args:
        pitcher_id: Pitcher identifier (matches 'pitcher' column).
        current_role: Current pitcher archetype (e.g. 'starter').
        target_role: Target archetype (e.g. 'reliever').
        season: Season to simulate.
        master_df: Feature matrix (must contain 'archetype' column).
        model_dict: Must contain 'xgb_pipeline' and 'feature_cols'.

    Returns:
        Dictionary with original_prob, simulated_prob, risk_reduction_pct,
        recommended_transition.
    """
    workload_cols = [
        "pitch_count", "pitches_7d", "pitches_28d", "pitches_90d", "days_rest",
    ]
    available = [c for c in workload_cols if c in master_df.columns]

    pitcher_rows = master_df[
        (master_df["pitcher"] == pitcher_id) & (master_df["season"] == season)
    ]
    if pitcher_rows.empty:
        # Fall back to any season
        pitcher_rows = master_df[master_df["pitcher"] == pitcher_id]

    if pitcher_rows.empty:
        return {
            "original_prob": float("nan"),
            "simulated_prob": float("nan"),
            "risk_reduction_pct": 0.0,
            "recommended_transition": False,
        }

    profile = pitcher_rows.iloc[len(pitcher_rows) // 2]  # median game
    orig_prob = _predict_prob(profile, model_dict)

    arch_medians = _archetype_medians(master_df, available)
    if target_role not in arch_medians:
        return {
            "original_prob": orig_prob,
            "simulated_prob": orig_prob,
            "risk_reduction_pct": 0.0,
            "recommended_transition": False,
        }

    mod = profile.copy()
    for col, val in arch_medians[target_role].items():
        if col in mod.index:
            mod[col] = val

    sim_prob = _predict_prob(mod, model_dict)
    pct = (orig_prob - sim_prob) / max(orig_prob, 1e-9) * 100.0

    return {
        "original_prob": orig_prob,
        "simulated_prob": sim_prob,
        "risk_reduction_pct": pct,
        "recommended_transition": sim_prob < orig_prob,
    }


def simulate_staff_workload_distribution(
    team: str,
    season: int,
    master_df: pd.DataFrame,
    model_dict: dict,
    redistribution_strategy: str = "even",
) -> pd.DataFrame:
    """Simulate redistributing workload across a pitching staff to minimize total risk.

    Tests an alternative workload distribution across staff members and
    identifies the arrangement that minimizes sum of predicted injury probabilities.

    Args:
        team: Team abbreviation (matched against a 'team' column if present).
        season: Season year.
        master_df: Feature matrix.
        model_dict: Must contain 'xgb_pipeline' and 'feature_cols'.
        redistribution_strategy: 'even' = equalize pitch counts,
            'risk_weighted' = give more load to low-risk pitchers.

    Returns:
        DataFrame with pitcher, original_mean_prob, simulated_mean_prob, risk_delta.
    """
    season_df = master_df[master_df["season"] == season]
    if "team" in season_df.columns:
        season_df = season_df[season_df["team"] == team]

    if season_df.empty:
        return pd.DataFrame(columns=["pitcher", "original_mean_prob", "simulated_mean_prob", "risk_delta"])

    pitcher_profiles = (
        season_df.groupby("pitcher")
        .apply(lambda g: g.iloc[len(g) // 2])
        .reset_index(drop=True)
    )

    orig_probs = pitcher_profiles.apply(
        lambda row: _predict_prob(row, model_dict), axis=1
    )

    if redistribution_strategy == "even":
        # Equalize pitch counts to the mean
        mean_count = float(pitcher_profiles["pitch_count"].mean()) if "pitch_count" in pitcher_profiles.columns else 75.0
        mod_profiles = pitcher_profiles.copy()
        if "pitch_count" in mod_profiles.columns:
            mod_profiles["pitch_count"] = mean_count
    elif redistribution_strategy == "risk_weighted":
        # High-risk pitchers get fewer pitches
        orig_arr = orig_probs.values
        rank = orig_arr.argsort().argsort()  # 0 = lowest risk
        n = len(rank)
        adjustments = 0.8 + 0.4 * (n - 1 - rank) / max(n - 1, 1)  # 0.8–1.2
        mod_profiles = pitcher_profiles.copy()
        if "pitch_count" in mod_profiles.columns:
            base = float(mod_profiles["pitch_count"].mean())
            mod_profiles["pitch_count"] = base * adjustments
    else:
        mod_profiles = pitcher_profiles.copy()

    sim_probs = mod_profiles.apply(
        lambda row: _predict_prob(row, model_dict), axis=1
    )

    return pd.DataFrame({
        "pitcher": pitcher_profiles["pitcher"].values,
        "original_mean_prob": orig_probs.values,
        "simulated_mean_prob": sim_probs.values,
        "risk_delta": (sim_probs - orig_probs).values,
    })


def compare_starter_vs_hybrid(
    master_df: pd.DataFrame,
    model_dict: dict,
    control_for: list[str] | None = None,
) -> pd.DataFrame:
    """Compare injury rates and predicted risk between starters and relievers.

    Uses the 'archetype' column (if present) to split groups and computes
    observed injury rate and mean predicted injury probability per group.

    Args:
        master_df: Feature matrix (must contain 'archetype' and 'injured_next_30d').
        model_dict: Must contain 'xgb_pipeline' and 'feature_cols'.
        control_for: Unused; reserved for future covariate adjustment.

    Returns:
        Summary DataFrame with archetype, n_observations, observed_injury_rate,
        mean_predicted_prob.
    """
    if "archetype" not in master_df.columns:
        return pd.DataFrame(columns=["archetype", "n_observations", "observed_injury_rate", "mean_predicted_prob"])

    rows = []
    for arch, grp in master_df.groupby("archetype"):
        sample = grp.sample(n=min(500, len(grp)), random_state=42)
        preds = sample.apply(lambda row: _predict_prob(row, model_dict), axis=1)
        observed_rate = float(grp["injured_next_30d"].mean()) if "injured_next_30d" in grp.columns else float("nan")
        rows.append({
            "archetype": arch,
            "n_observations": len(grp),
            "observed_injury_rate": observed_rate,
            "mean_predicted_prob": float(preds.mean()),
        })

    return pd.DataFrame(rows).sort_values("mean_predicted_prob", ascending=False).reset_index(drop=True)


def run_archetype_optimization(
    pitcher_id: int,
    season: int,
    master_df: pd.DataFrame,
    model_dict: dict,
) -> dict:
    """Find the usage strategy that minimizes injury risk for a specific pitcher.

    Integrates workload and pitch-mix simulations to find the optimal strategy.

    Args:
        pitcher_id: Pitcher to optimize.
        season: Season to target.
        master_df: Feature matrix.
        model_dict: Must contain 'xgb_pipeline' and 'feature_cols'.

    Returns:
        Dictionary with original_prob, best_pitch_count, best_rest_days, min_prob.
    """
    from src.simulation.workload_simulator import find_optimal_pitch_count, simulate_rest_schedule

    pitcher_rows = master_df[
        (master_df["pitcher"] == pitcher_id) & (master_df["season"] == season)
    ]
    if pitcher_rows.empty:
        pitcher_rows = master_df[master_df["pitcher"] == pitcher_id]
    if pitcher_rows.empty:
        return {"original_prob": float("nan"), "best_pitch_count": None, "best_rest_days": None, "min_prob": float("nan")}

    profile = pitcher_rows.iloc[len(pitcher_rows) // 2]
    pitch_opt = find_optimal_pitch_count(profile, model_dict)

    best_prob = pitch_opt["min_prob"]
    opt_profile = profile.copy()
    opt_profile["pitch_count"] = pitch_opt["optimal_pitch_count"]

    best_rest_days = int(profile.get("days_rest", 4.0) or 4.0)
    for rest in range(1, 11):
        result = simulate_rest_schedule(opt_profile, rest, model_dict)
        if result["counterfactual_prob"] < best_prob:
            best_prob = result["counterfactual_prob"]
            best_rest_days = rest

    return {
        "original_prob": pitch_opt["original_prob"],
        "best_pitch_count": pitch_opt["optimal_pitch_count"],
        "best_rest_days": best_rest_days,
        "min_prob": best_prob,
    }
