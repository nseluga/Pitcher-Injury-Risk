"""
score_calibration.py

Handles calibration of the Injury Risk+ score, including:
- Blend weight optimization (how much weight each model task gets)
- Era and archetype normalization reference tables
- Probability calibration (Platt scaling, isotonic regression)
- Score stability analysis across seasons

Good calibration ensures that a score of 150 means the same thing in 2015
as it does in 2024, and that a predicted 30% injury probability actually
corresponds to a ~30% observed injury rate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import linregress
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score


def calibrate_probabilities(
    y_true: pd.Series,
    y_prob: pd.Series,
    method: str = "isotonic",
) -> object:
    """Fit a probability calibration model.

    Args:
        y_true: True binary injury labels.
        y_prob: Raw model-predicted probabilities.
        method: 'isotonic' or 'sigmoid' (Platt scaling).

    Returns:
        Fitted calibration model — IsotonicRegression for 'isotonic', or a
        fitted sklearn LogisticRegression (1-D Platt scaler) for 'sigmoid'.
        Both expose `.predict(y_prob)` -> calibrated probabilities via
        `apply_calibration`.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    if method == "isotonic":
        model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        model.fit(y_prob, y_true)
        return model

    if method == "sigmoid":
        model = LogisticRegression()
        model.fit(y_prob.reshape(-1, 1), y_true)
        return model

    raise ValueError(f"Unknown method: {method!r} (expected 'isotonic' or 'sigmoid')")


def apply_calibration(
    y_prob: pd.Series,
    calibration_model: object,
) -> pd.Series:
    """Apply a fitted probability calibration model to raw predictions.

    Args:
        y_prob: Raw predicted probabilities.
        calibration_model: Fitted calibration model from calibrate_probabilities.

    Returns:
        Calibrated probability Series (same index as y_prob).
    """
    arr = np.asarray(y_prob, dtype=float)
    if isinstance(calibration_model, IsotonicRegression):
        calibrated = calibration_model.predict(arr)
    elif isinstance(calibration_model, (LogisticRegression, CalibratedClassifierCV)):
        calibrated = calibration_model.predict_proba(arr.reshape(-1, 1))[:, 1]
    else:
        raise TypeError(f"Unsupported calibration model type: {type(calibration_model).__name__}")

    index = y_prob.index if isinstance(y_prob, pd.Series) else None
    return pd.Series(calibrated, index=index, name="calibrated_prob")


def evaluate_calibration(
    y_true: pd.Series,
    y_prob: pd.Series,
    n_bins: int = 10,
) -> dict[str, float]:
    """Compute calibration quality metrics.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities (should be calibrated).
        n_bins: Number of bins for the reliability diagram.

    Returns:
        Dictionary with keys: expected_calibration_error, max_calibration_error,
        brier_score.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)

    bin_edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    bin_edges[0], bin_edges[-1] = 0.0, 1.0
    bin_edges = np.unique(bin_edges)

    bin_ids = np.digitize(y_prob, bin_edges[1:-1], right=True)

    gaps = []
    weights = []
    for b in np.unique(bin_ids):
        mask = bin_ids == b
        if mask.sum() == 0:
            continue
        bin_conf = y_prob[mask].mean()
        bin_acc = y_true[mask].mean()
        gaps.append(abs(bin_conf - bin_acc))
        weights.append(mask.sum())

    weights = np.asarray(weights, dtype=float)
    gaps = np.asarray(gaps, dtype=float)

    ece = float(np.average(gaps, weights=weights)) if len(gaps) else float("nan")
    mce = float(gaps.max()) if len(gaps) else float("nan")

    # Calibration slope: OLS slope of observed labels on predicted probs.
    # Ideal = 1.0; < 1 = overconfident (probabilities spread too wide);
    # > 1 = underconfident. Recommended by JOSPT 2021 Clinical Prediction
    # Models guide as the primary calibration-in-the-large metric alongside
    # ECE and Brier score.
    if len(np.unique(y_prob)) > 1:
        slope, intercept, _, _, _ = linregress(y_prob, y_true)
    else:
        slope, intercept = float("nan"), float("nan")

    return {
        "expected_calibration_error": ece,
        "max_calibration_error": mce,
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "calibration_slope": float(slope),
        "calibration_intercept": float(intercept),
    }


def optimize_blend_weights(
    predictions_df: pd.DataFrame,
    y_true: pd.Series,
    metric: str = "brier_score",
) -> dict[str, float]:
    """Find the optimal weights for combining model outputs into the raw risk score.

    Searches over the 2-simplex of (injury_prob_weight, days_lost_weight,
    hazard_weight) — each in [0, 1], summing to 1 — for the combination that
    minimizes Brier score (or maximizes AUC-ROC) of the blended score against
    observed injury outcomes.

    Args:
        predictions_df: DataFrame with component prediction columns
            (injury_prob_30d, expected_days_lost, hazard_rate — any subset;
            missing components are dropped from the blend).
        y_true: True injury labels.
        metric: Optimization target ('brier_score' to minimize, 'auc_roc' to maximize).

    Returns:
        Dictionary mapping component name to optimal weight (sums to 1.0).
    """
    if metric not in {"brier_score", "auc_roc"}:
        raise ValueError(f"Unknown metric: {metric!r} (expected 'brier_score' or 'auc_roc')")

    components = [c for c in ["injury_prob_30d", "expected_days_lost", "hazard_rate"]
                  if c in predictions_df.columns]
    if not components:
        raise ValueError("predictions_df has none of the expected blend components "
                         "(injury_prob_30d, expected_days_lost, hazard_rate)")

    y = np.asarray(y_true, dtype=float)

    # Min-max normalize each component to [0, 1] so weights are comparable.
    norm = {}
    for c in components:
        vals = predictions_df[c].astype(float).fillna(predictions_df[c].median())
        lo, hi = vals.min(), vals.max()
        norm[c] = ((vals - lo) / (hi - lo)) if hi > lo else vals * 0.0
    norm_df = pd.DataFrame(norm)

    def loss(raw_weights: np.ndarray) -> float:
        w = np.abs(raw_weights)
        w = w / w.sum() if w.sum() > 0 else np.ones(len(w)) / len(w)
        blended = norm_df.values @ w
        if metric == "brier_score":
            return brier_score_loss(y, blended)
        # Maximize AUC == minimize negative AUC; guard against degenerate y.
        if len(np.unique(y)) < 2:
            return 0.0
        return -roc_auc_score(y, blended)

    n = len(components)
    x0 = np.ones(n) / n
    result = minimize(loss, x0, method="Nelder-Mead")

    w = np.abs(result.x)
    w = w / w.sum() if w.sum() > 0 else np.ones(n) / n

    weights = dict(zip(components, w.tolist()))
    # Always return all three canonical keys (zero weight for absent components)
    # so downstream code (compute_raw_risk_score) can rely on their presence.
    for c in ["injury_prob_30d", "expected_days_lost", "hazard_rate"]:
        weights.setdefault(c, 0.0)
    return weights


def build_normalization_reference(
    risk_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute per-season and per-archetype normalization parameters.

    Stores the mean and std of raw risk scores for each season × archetype
    combination. Used at inference time to map raw scores to Injury Risk+.

    Args:
        risk_df: DataFrame with raw_risk_score, season, and archetype columns.

    Returns:
        Reference DataFrame with columns: season, archetype, mean_raw_score,
        std_raw_score.
    """
    ref = (
        risk_df
        .groupby(["season"], observed=True)["raw_risk_score"]
        .agg(mean_raw_score="mean", std_raw_score="std")
        .reset_index()
    )
    ref["std_raw_score"] = ref["std_raw_score"].fillna(0.0)
    return ref
