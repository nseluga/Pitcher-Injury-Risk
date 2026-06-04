"""
multitask_models.py

Multi-task learning models that predict several injury-related targets
simultaneously. By sharing representations across related tasks, multi-task
models can outperform independently trained single-task models — especially
for rare outcomes like severe injuries.

Prediction targets (tasks):
1. Binary: injured within 30 / 60 / 90 days
2. Regression: days until next injury (time-to-event)
3. Regression: expected days lost if injured
4. Multi-class: injury type (elbow, shoulder, forearm, other)

Approaches:
- Hard parameter sharing (shared trunk, task-specific heads)
- Soft parameter sharing (separate trunks with cross-task regularization)
- Future: neural multi-task models (PyTorch)
"""

from __future__ import annotations

import pandas as pd


def prepare_multitask_dataset(
    master_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    test_seasons: list[int] | None = None,
) -> tuple:
    """Prepare the dataset for multi-task learning.

    Returns a feature matrix and a dictionary of target arrays, one per task.

    Args:
        master_df: Master modeling DataFrame with all label columns present.
        feature_cols: Feature columns to include.
        test_seasons: Seasons to hold out for evaluation.

    Returns:
        Tuple of (X_train, X_test, y_train_dict, y_test_dict) where each
        y_dict maps task name to its target array.
    """
    raise NotImplementedError


def train_chained_multitask_model(
    X_train: pd.DataFrame,
    y_train_dict: dict[str, pd.Series],
) -> dict[str, object]:
    """Train a set of models that pass predictions between tasks as features.

    Injury probability is predicted first, then fed as a feature into the
    injury-type and days-lost regressors. This explicit chain reflects the
    causal structure of the prediction problem.

    Args:
        X_train: Training feature matrix.
        y_train_dict: Dictionary mapping task name to training target.

    Returns:
        Dictionary mapping task name to fitted model object.
    """
    raise NotImplementedError


def train_shared_representation_model(
    X_train: pd.DataFrame,
    y_train_dict: dict[str, pd.Series],
    shared_model_type: str = "gradient_boosting",
) -> object:
    """Train a multi-output model with a shared underlying representation.

    Uses sklearn's MultiOutputClassifier / MultiOutputRegressor or a custom
    gradient boosting approach where the shared tree structure handles all tasks.

    Args:
        X_train: Training feature matrix.
        y_train_dict: Dictionary mapping task name to training target.
        shared_model_type: Base learner type ('gradient_boosting', 'random_forest').

    Returns:
        Fitted multi-output model object.
    """
    raise NotImplementedError


def predict_all_tasks(
    models: dict[str, object] | object,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Run inference for all tasks and return a unified predictions DataFrame.

    Args:
        models: Either a dict of task-name → model (chained) or a single
            multi-output model (shared representation).
        X: Feature matrix.

    Returns:
        DataFrame with one prediction column per task.
    """
    raise NotImplementedError


def compute_multitask_metrics(
    y_true_dict: dict[str, pd.Series],
    y_pred_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute evaluation metrics for all tasks.

    For classification tasks: AUC-ROC, PR-AUC, Brier score.
    For regression tasks: MAE, RMSE, R².

    Args:
        y_true_dict: True labels per task.
        y_pred_df: Predictions DataFrame from predict_all_tasks.

    Returns:
        DataFrame with one row per task and one column per metric.
    """
    raise NotImplementedError
