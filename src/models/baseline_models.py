"""
baseline_models.py

Implements baseline injury prediction models that provide a performance
floor for evaluating more sophisticated approaches.

Baseline approaches:
1. Logistic Regression (injury within N days, binary classification)
2. Random Forest Classifier (injury probability)
3. Gradient Boosting Classifier (XGBoost / LightGBM)
4. Ridge Regression (days lost prediction, continuous target)
5. Naive baselines: always-predict-mean, historical-rate-by-archetype

These models operate on the tabular feature set from src/features/ and
serve as benchmarks for the survival and multitask models.
"""

from __future__ import annotations

import pandas as pd


def prepare_classification_dataset(
    master_df: pd.DataFrame,
    target_col: str = "injured_within_30d",
    feature_cols: list[str] | None = None,
    test_seasons: list[int] | None = None,
) -> tuple:
    """Split the master dataset into train/test sets for classification.

    Uses temporal splitting (train on earlier seasons, test on later) to
    avoid data leakage. Returns X_train, X_test, y_train, y_test.

    Args:
        master_df: Master modeling DataFrame from build_master_dataset.
        target_col: Binary injury label column to predict.
        feature_cols: List of feature columns to include. If None, uses all
            non-label, non-identifier columns.
        test_seasons: List of seasons to hold out for testing. Defaults to
            the two most recent seasons.

    Returns:
        Tuple of (X_train, X_test, y_train, y_test).
    """
    raise NotImplementedError


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> object:
    """Train a regularized logistic regression model.

    Args:
        X_train: Training feature matrix.
        y_train: Binary injury labels.

    Returns:
        Fitted sklearn LogisticRegression object.
    """
    raise NotImplementedError


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> object:
    """Train a Random Forest classifier with cross-validated hyperparameters.

    Args:
        X_train: Training feature matrix.
        y_train: Binary injury labels.

    Returns:
        Fitted sklearn RandomForestClassifier object.
    """
    raise NotImplementedError


def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    framework: str = "xgboost",
) -> object:
    """Train a gradient boosting classifier (XGBoost or LightGBM).

    Args:
        X_train: Training feature matrix.
        y_train: Binary injury labels.
        framework: 'xgboost' or 'lightgbm'.

    Returns:
        Fitted gradient boosting model object.
    """
    raise NotImplementedError


def train_naive_baseline(
    master_df: pd.DataFrame,
    strategy: str = "historical_rate",
) -> object:
    """Train a naive baseline model for benchmarking.

    Strategies:
    - 'mean': Always predict the population mean injury rate.
    - 'historical_rate': Predict injury rate by pitcher archetype and age band.

    Args:
        master_df: Master modeling DataFrame.
        strategy: Which naive strategy to use.

    Returns:
        Fitted baseline model (simple callable or sklearn DummyClassifier).
    """
    raise NotImplementedError


def save_model(model: object, path: str) -> None:
    """Serialize a trained model to disk.

    Args:
        model: Fitted model object.
        path: Destination path (joblib format).
    """
    raise NotImplementedError


def load_model(path: str) -> object:
    """Load a serialized model from disk.

    Args:
        path: Path to the joblib-serialized model.

    Returns:
        Loaded model object.
    """
    raise NotImplementedError
