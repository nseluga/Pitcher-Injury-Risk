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

import numpy as np
import pandas as pd

DEFAULT_WEIGHTS = {
    "injury_prob_30d": 0.50,
    "expected_days_lost": 0.30,
    "hazard_rate": 0.20,
}


def _archetype_for(master_df: pd.DataFrame) -> pd.Series:
    """Rule-based starter/reliever split (see baseline_models._rule_based_archetype).

    `encode_pitcher_archetype` (src/features/risk_factor_features.py) is still
    a stub, so we fall back to the same pitch-count heuristic used elsewhere
    in the pipeline — pitchers averaging >= 50 pitches per appearance in a
    season are treated as starters.
    """
    season_avg = master_df.groupby(["pitcher", "season"])["pitch_count"].transform("mean")
    return pd.Series(np.where(season_avg >= 50, "starter", "reliever"), index=master_df.index)


def _normalize_component(s: pd.Series, clip_percentile: float = 1.0) -> pd.Series:
    """Percentile-clip then min-max normalize a component to [0, 1].

    Clips at [clip_percentile, 100-clip_percentile] quantiles before
    normalizing. Without clipping, a single TJ-surgery case (365+ days lost)
    would dominate the expected_days_lost range and compress all other pitchers
    toward zero. clip_percentile=1.0 retains 98% of the observed distribution.
    Reference: COINr composite indicator guide; recommended for composite
    scores with right-skewed components.
    """
    s = s.astype(float)
    lo_clip = float(s.quantile(clip_percentile / 100.0))
    hi_clip = float(s.quantile(1.0 - clip_percentile / 100.0))
    if np.isfinite(lo_clip) and np.isfinite(hi_clip) and hi_clip > lo_clip:
        s = s.clip(lo_clip, hi_clip)
    lo, hi = s.min(), s.max()
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def compute_raw_risk_score(
    predictions_df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Combine multi-task model predictions into a single raw risk index.

    Blends:
    - Injury probability (30-day) — primary component
    - Expected days lost (severity weight)
    - Survival hazard rate (time-to-event signal)

    Each component is min-max normalized to [0, 1] before blending so that
    weights are comparable regardless of each component's native scale.
    Blend weights default to docs/injury_risk_plus_design.md's values, or
    may be supplied directly (e.g. from score_calibration.optimize_blend_weights).

    Args:
        predictions_df: DataFrame with columns: injury_prob_30d,
            expected_days_lost, hazard_rate.
        weights: Optional override for blend weights (defaults to
            DEFAULT_WEIGHTS / docs/injury_risk_plus_design.md).

    Returns:
        predictions_df with new column raw_risk_score.
    """
    weights = weights or DEFAULT_WEIGHTS
    out = predictions_df.copy()

    raw = pd.Series(np.zeros(len(out)), index=out.index)
    total_weight = 0.0
    for component, w in weights.items():
        if component not in out.columns or w == 0:
            continue
        raw = raw + w * _normalize_component(out[component])
        total_weight += w

    if total_weight > 0:
        raw = raw / total_weight

    out["raw_risk_score"] = raw
    return out


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
        reference_df: Historical reference DataFrame (output of
            score_calibration.build_normalization_reference, with columns
            season, archetype, mean_raw_score) used to compute the
            normalization denominator. If None, the mean of raw_scores itself
            is used (in-sample normalization).

    Returns:
        Series of Injury Risk+ scores (mean ≈ 100 within the reference group).
    """
    denom = None
    if reference_df is not None:
        ref = reference_df[reference_df["season"] == season]
        if archetype is not None:
            ref = ref[ref["archetype"] == archetype]
        if len(ref) > 0:
            denom = float(ref["mean_raw_score"].mean())

    if denom is None or denom == 0:
        denom = float(raw_scores.mean())
    if denom == 0 or not np.isfinite(denom):
        denom = 1.0

    return (raw_scores.astype(float) / denom) * 100.0


def compute_seasonal_risk_plus(
    master_df: pd.DataFrame,
    model_dict: dict[str, object],
    weights: dict[str, float] | None = None,
    survival_model: object | None = None,
) -> pd.DataFrame:
    """Compute Injury Risk+ for all pitchers for each season in the dataset.

    Runs model inference, computes raw scores, and normalizes within each
    season-archetype group.

    Args:
        master_df: Master modeling DataFrame.
        model_dict: Dictionary of fitted task models from multitask_models
            (output of train_chained_multitask_model / train_shared_representation_model,
            or a dict already keyed by task name).
        weights: Optional blend-weight override (see compute_raw_risk_score).
        survival_model: Optional fitted survival model (from survival_models)
            used to derive the hazard-rate component. If None, the hazard
            component is omitted from the blend.

    Returns:
        DataFrame with columns: pitcher_id, season, injury_risk_plus,
        raw_risk_score, archetype, plus all component scores.
    """
    from src.models.multitask_models import predict_all_tasks
    from src.models.baseline_models import _infer_feature_cols
    from src.scoring.score_calibration import build_normalization_reference

    feature_cols = _infer_feature_cols(master_df)
    X = master_df[feature_cols]

    task_preds = predict_all_tasks(model_dict, X)

    components = pd.DataFrame(index=master_df.index)
    components["injury_prob_30d"] = task_preds.get("injured_next_30d_pred", np.nan)
    components["expected_days_lost"] = task_preds.get("next_injury_days_lost_pred", np.nan)

    if survival_model is not None:
        from src.models.survival_models import predict_survival_function
        surv = predict_survival_function(survival_model, X, time_points=[30])
        components["hazard_rate"] = 1.0 - surv["S(30)"].values

    # Aggregate per-appearance predictions to pitcher-season level before
    # blending and normalizing. Normalization must happen at pitcher-season
    # grain so that mean=100 holds when scores are reported at that level.
    app_df = master_df[["pitcher", "season"]].copy().rename(columns={"pitcher": "pitcher_id"})
    app_df["archetype"] = _archetype_for(master_df).values
    for col in components.columns:
        app_df[col] = components[col].values

    component_cols = [c for c in ["injury_prob_30d", "expected_days_lost", "hazard_rate"]
                      if c in app_df.columns]
    risk_df = (
        app_df.groupby(["pitcher_id", "season", "archetype"], observed=True)[component_cols]
        .mean()
        .reset_index()
    )

    risk_df = compute_raw_risk_score(risk_df, weights=weights)

    reference_df = build_normalization_reference(risk_df)

    irp = pd.Series(index=risk_df.index, dtype=float)
    for (season, archetype), group in risk_df.groupby(["season", "archetype"], observed=True):
        irp.loc[group.index] = normalize_to_injury_risk_plus(
            group["raw_risk_score"], season=season, archetype=archetype, reference_df=reference_df,
        )
    risk_df["injury_risk_plus"] = irp

    return risk_df


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
    df = risk_df[risk_df["season"] == season]
    if archetype is not None:
        df = df[df["archetype"] == archetype]
    return df.sort_values("injury_risk_plus", ascending=False).head(n).reset_index(drop=True)


def compute_risk_percentile(risk_df: pd.DataFrame) -> pd.DataFrame:
    """Attach a percentile rank (0–100) alongside each Injury Risk+ score.

    Percentile is computed within the same season and archetype group.

    Args:
        risk_df: Output of compute_seasonal_risk_plus.

    Returns:
        risk_df with new column risk_percentile.
    """
    out = risk_df.copy()
    out["risk_percentile"] = (
        out.groupby(["season", "archetype"], observed=True)["injury_risk_plus"]
        .rank(pct=True, method="average") * 100.0
    )
    return out
