"""
evaluation.py

Unified model evaluation utilities used across baseline, survival, and
multitask models. Provides calibration checks, threshold optimization,
and visualization-ready output for use in notebooks.

Metrics tracked:
- Classification: AUC-ROC, PR-AUC, Brier score, calibration curve
- Survival: C-index, Brier score at time horizons, IBS (integrated Brier score)
- Regression: MAE, RMSE, MAPE, R²
- All: confusion matrix, lift curves, feature importance
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

matplotlib.use("Agg")


def evaluate_classifier(
    y_true: pd.Series,
    y_prob: pd.Series,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute a full suite of classification metrics.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        threshold: Decision threshold for converting probabilities to labels.

    Returns:
        Dictionary with keys: auc_roc, pr_auc, brier_score, accuracy,
        precision, recall, f1, mcc.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    # AUC-based metrics require both classes present in y_true.
    if len(np.unique(y_true)) > 1:
        auc_roc = roc_auc_score(y_true, y_prob)
        pr_auc = average_precision_score(y_true, y_prob)
    else:
        auc_roc = float("nan")
        pr_auc = float("nan")

    return {
        "auc_roc": float(auc_roc),
        "pr_auc": float(pr_auc),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
    }


def evaluate_regression(
    y_true: pd.Series,
    y_pred: pd.Series,
) -> dict[str, float]:
    """Compute regression evaluation metrics.

    Args:
        y_true: True continuous values.
        y_pred: Predicted values.

    Returns:
        Dictionary with keys: mae, rmse, mape, r2.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))

    # MAPE blows up near zero targets (days-lost can be 0); guard against that.
    nonzero = y_true != 0
    mape = float(mean_absolute_percentage_error(y_true[nonzero], y_pred[nonzero])) if nonzero.any() else float("nan")

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "mape": mape,
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_survival_model(
    model: object,
    X_test: pd.DataFrame,
    T_test: pd.Series,
    E_test: pd.Series,
    time_points: list[int] | None = None,
) -> dict[str, float]:
    """Compute survival model evaluation metrics.

    Args:
        model: Fitted survival model.
        X_test: Test feature matrix.
        T_test: Test duration series.
        E_test: Test event indicator.
        time_points: Horizons for time-specific Brier scores.

    Returns:
        Dictionary with keys: c_index, ibs, brier_{t} for each time point.
    """
    if time_points is None:
        time_points = [30, 60, 90]

    from src.models.survival_models import compute_concordance_index, predict_survival_function

    metrics: dict[str, float] = {
        "c_index": compute_concordance_index(model, X_test, T_test, E_test),
    }

    surv_df = predict_survival_function(model, X_test, time_points=time_points)

    # Brier score at each horizon: compare predicted survival probability
    # against the observed event-by-time-t indicator, restricted to rows
    # whose follow-up is long enough to be informative at that horizon
    # (i.e. not censored before t).
    brier_scores = []
    for t in time_points:
        observed_event_by_t = ((T_test <= t) & (E_test == 1)).astype(float)
        informative = (T_test >= t) | (E_test == 1)
        if informative.sum() == 0:
            metrics[f"brier_{t}"] = float("nan")
            continue
        pred_event_by_t = 1.0 - surv_df[f"S({t})"].values
        bs = brier_score_loss(
            observed_event_by_t[informative],
            pred_event_by_t[informative.values],
        )
        metrics[f"brier_{t}"] = float(bs)
        brier_scores.append(bs)

    metrics["ibs"] = float(np.mean(brier_scores)) if brier_scores else float("nan")
    return metrics


def plot_calibration_curve(
    y_true: pd.Series,
    y_prob: pd.Series,
    n_bins: int = 10,
) -> object:
    """Generate a calibration curve (reliability diagram).

    Returns a matplotlib figure object for inclusion in reports.

    Args:
        y_true: True binary labels.
        y_prob: Predicted probabilities.
        n_bins: Number of probability bins.

    Returns:
        matplotlib Figure object.
    """
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="quantile")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.plot(mean_pred, frac_pos, marker="o", label="Model")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed injury rate")
    ax.set_title("Calibration curve")
    ax.legend()
    fig.tight_layout()
    return fig


def compute_feature_importance(
    model: object,
    feature_names: list[str],
    method: str = "native",
) -> pd.DataFrame:
    """Extract feature importances from a fitted model.

    Args:
        model: Fitted model with feature importance support. May be a bare
            estimator or an sklearn Pipeline ending in one.
        feature_names: List of feature names in model order.
        method: 'native' for tree/coefficient importances, 'permutation' for
            permutation importance (requires X, y — see notes below).

    Returns:
        DataFrame with columns: feature, importance, sorted descending.
    """
    estimator = model
    if hasattr(model, "steps"):
        estimator = model.steps[-1][1]

    if method != "native":
        raise ValueError(f"Unsupported method: {method!r} (only 'native' is implemented here; "
                         "use sklearn.inspection.permutation_importance directly for 'permutation')")

    if hasattr(estimator, "feature_importances_"):
        importances = np.asarray(estimator.feature_importances_, dtype=float)
    elif hasattr(estimator, "coef_"):
        importances = np.abs(np.asarray(estimator.coef_, dtype=float)).ravel()
    else:
        raise ValueError(f"Model {type(estimator).__name__} exposes neither "
                         "feature_importances_ nor coef_")

    df = pd.DataFrame({"feature": feature_names, "importance": importances})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def temporal_cross_validate(
    model_fn: callable,
    master_df: pd.DataFrame,
    n_splits: int = 5,
    target_col: str = "injured_next_30d",
    feature_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Run time-series cross-validation (walk-forward) for any model.

    Avoids leakage by always training on earlier seasons and testing on
    later ones. Returns per-fold metrics for stability analysis.

    Args:
        model_fn: Callable that accepts (X_train, y_train) and returns a
            fitted model exposing predict_proba.
        master_df: Master modeling DataFrame.
        n_splits: Number of temporal folds.
        target_col: Binary injury label column to predict.
        feature_cols: Feature columns to use. If None, inferred from the
            baseline_models identifier/label conventions.

    Returns:
        DataFrame with one row per fold and metric columns.
    """
    from src.models.baseline_models import _infer_feature_cols

    if feature_cols is None:
        feature_cols = _infer_feature_cols(master_df)

    df = master_df.sort_values("game_date").reset_index(drop=True)
    seasons = sorted(df["season"].unique())

    if len(seasons) > 1:
        # Walk forward season-by-season: each fold trains on all seasons up
        # to k and tests on season k+1.
        folds = list(zip(seasons[:-1], seasons[1:]))[-n_splits:]
        records = []
        for fold_id, (_, test_season) in enumerate(folds):
            train_df = df[df["season"] < test_season]
            test_df = df[df["season"] == test_season]
            if train_df.empty or test_df.empty:
                continue
            X_train, y_train = train_df[feature_cols], train_df[target_col]
            X_test, y_test = test_df[feature_cols], test_df[target_col]
            if y_train.nunique() < 2:
                continue
            model = model_fn(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
            metrics = evaluate_classifier(y_test, y_prob)
            metrics.update({"fold": fold_id, "test_season": test_season, "n_test": len(test_df)})
            records.append(metrics)
        return pd.DataFrame(records)

    # Single-season fallback: chronological expanding-window splits within
    # the season (no cross-season structure available).
    n = len(df)
    fold_edges = np.linspace(int(n * 0.5), n, n_splits + 1, dtype=int)
    records = []
    for fold_id in range(n_splits):
        train_end, test_end = fold_edges[fold_id], fold_edges[fold_id + 1]
        if train_end == test_end:
            continue
        train_df = df.iloc[:train_end]
        test_df = df.iloc[train_end:test_end]
        X_train, y_train = train_df[feature_cols], train_df[target_col]
        X_test, y_test = test_df[feature_cols], test_df[target_col]
        if y_train.nunique() < 2 or len(test_df) == 0:
            continue
        model = model_fn(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]
        metrics = evaluate_classifier(y_test, y_prob)
        metrics.update({"fold": fold_id, "test_season": int(df["season"].iloc[0]), "n_test": len(test_df)})
        records.append(metrics)
    return pd.DataFrame(records)
