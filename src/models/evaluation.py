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

import pandas as pd


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


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
    raise NotImplementedError


def compute_feature_importance(
    model: object,
    feature_names: list[str],
    method: str = "native",
) -> pd.DataFrame:
    """Extract feature importances from a fitted model.

    Args:
        model: Fitted model with feature importance support.
        feature_names: List of feature names in model order.
        method: 'native' for tree importances, 'shap' for SHAP values.

    Returns:
        DataFrame with columns: feature, importance, sorted descending.
    """
    raise NotImplementedError


def temporal_cross_validate(
    model_fn: callable,
    master_df: pd.DataFrame,
    n_splits: int = 5,
) -> pd.DataFrame:
    """Run time-series cross-validation (walk-forward) for any model.

    Avoids leakage by always training on earlier seasons and testing on
    later ones. Returns per-fold metrics for stability analysis.

    Args:
        model_fn: Callable that accepts (X_train, y_train) and returns a
            fitted model.
        master_df: Master modeling DataFrame.
        n_splits: Number of temporal folds.

    Returns:
        DataFrame with one row per fold and metric columns.
    """
    raise NotImplementedError
