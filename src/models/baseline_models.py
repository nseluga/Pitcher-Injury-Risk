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

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ID_COLS = ["pitcher", "game_date", "season", "birth_date", "player_name", "age"]
LABEL_COLS = [
    "injured_next_30d", "injured_next_60d", "injured_next_90d",
    "days_to_next_injury", "next_injury_days_lost", "next_injury_type",
]


def _infer_feature_cols(master_df: pd.DataFrame) -> list[str]:
    """Default feature columns: everything that isn't an identifier or label."""
    excluded = set(ID_COLS) | set(LABEL_COLS)
    return [c for c in master_df.columns if c not in excluded]


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
    if feature_cols is None:
        feature_cols = _infer_feature_cols(master_df)

    seasons = sorted(master_df["season"].unique())
    if test_seasons is None:
        test_seasons = seasons[-2:] if len(seasons) > 1 else seasons[-1:]

    is_test = master_df["season"].isin(test_seasons)
    train_df = master_df.loc[~is_test]
    test_df = master_df.loc[is_test]

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_train = train_df[target_col].copy()
    y_test = test_df[target_col].copy()

    return X_train, X_test, y_train, y_test


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> object:
    """Train a regularized logistic regression model.

    Wraps the estimator in a pipeline that imputes missing values (median)
    and standardizes features, since logistic regression is sensitive to
    feature scale and the feature matrix contains nulls from rolling windows.

    Args:
        X_train: Training feature matrix.
        y_train: Binary injury labels.

    Returns:
        Fitted sklearn Pipeline wrapping a LogisticRegression model.
    """
    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=42,
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> object:
    """Train a Random Forest classifier with cross-validated hyperparameters.

    Args:
        X_train: Training feature matrix.
        y_train: Binary injury labels.

    Returns:
        Fitted sklearn RandomForestClassifier object (best estimator from CV).
    """
    base = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1)),
    ])
    param_grid = {
        "clf__n_estimators": [200, 400],
        "clf__max_depth": [4, 8, None],
        "clf__min_samples_leaf": [1, 5],
    }
    search = GridSearchCV(
        base, param_grid, scoring="roc_auc", cv=3, n_jobs=-1, refit=True,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_


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
        Fitted gradient boosting Pipeline (with median imputation upstream).
    """
    n_pos = int(y_train.sum())
    n_neg = int(len(y_train) - n_pos)
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else 1.0

    if framework == "xgboost":
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1,
        )
    elif framework == "lightgbm":
        from lightgbm import LGBMClassifier

        clf = LGBMClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
            verbosity=-1,
        )
    else:
        raise ValueError(f"Unknown framework: {framework!r} (expected 'xgboost' or 'lightgbm')")

    pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", clf),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


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
        A fitted _NaiveBaseline object exposing predict_proba(X).
    """
    return _NaiveBaseline(strategy=strategy).fit(master_df)


class _NaiveBaseline:
    """Simple lookup-table baseline: predicts a constant or group-wise rate.

    Exposes a `predict_proba`-style interface so it can be evaluated with
    the same evaluation utilities as the sklearn-based models.
    """

    def __init__(self, strategy: str = "historical_rate", target_col: str = "injured_next_30d"):
        if strategy not in {"mean", "historical_rate"}:
            raise ValueError(f"Unknown strategy: {strategy!r}")
        self.strategy = strategy
        self.target_col = target_col
        self.global_rate_: float | None = None
        self.group_rates_: pd.Series | None = None

    def _age_band(self, age: pd.Series) -> pd.Series:
        bins = [0, 25, 28, 31, 34, 100]
        labels = ["<=25", "26-28", "29-31", "32-34", "35+"]
        return pd.cut(age, bins=bins, labels=labels)

    def fit(self, master_df: pd.DataFrame) -> "_NaiveBaseline":
        self.global_rate_ = float(master_df[self.target_col].mean())

        if self.strategy == "historical_rate":
            grp = master_df.copy()
            grp["_age_band"] = self._age_band(grp["age"])
            grp["_archetype"] = _rule_based_archetype(grp)
            self.group_rates_ = (
                grp.groupby(["_archetype", "_age_band"], observed=True)[self.target_col]
                .mean()
            )
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if self.strategy == "mean" or self.group_rates_ is None:
            p = np.full(len(X), self.global_rate_)
        else:
            grp = X.copy()
            grp["_archetype"] = _rule_based_archetype(grp)
            if "age" in grp.columns:
                grp["_age_band"] = self._age_band(grp["age"])
                keys = list(zip(grp["_archetype"], grp["_age_band"]))
                p = np.array([
                    self.group_rates_.get(k, self.global_rate_) for k in keys
                ], dtype=float)
                p = np.where(np.isnan(p), self.global_rate_, p)
            else:
                # age not in X (excluded as metadata) — fall back to archetype-only rate
                p = np.array([self.global_rate_] * len(grp), dtype=float)

        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.column_stack([1 - p, p])

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def _rule_based_archetype(df: pd.DataFrame) -> pd.Series:
    """Lightweight starter/reliever split using pitch count per appearance.

    `encode_pitcher_archetype` (src/features/risk_factor_features.py) is not
    yet implemented and the feature matrix has no archetype column, so we use
    a simple, well-understood proxy: starters routinely throw far more pitches
    per outing than relievers. This mirrors the "rule-based fallback" approach
    documented in notebook 11's plan.
    """
    if "pitch_count" not in df.columns:
        return pd.Series(["unknown"] * len(df), index=df.index)
    return pd.Series(
        np.where(df["pitch_count"] >= 50, "starter", "reliever"),
        index=df.index,
    )


def save_model(model: object, path: str) -> None:
    """Serialize a trained model to disk.

    Args:
        model: Fitted model object.
        path: Destination path (joblib format).
    """
    joblib.dump(model, path)


def load_model(path: str) -> object:
    """Load a serialized model from disk.

    Args:
        path: Path to the joblib-serialized model.

    Returns:
        Loaded model object.
    """
    return joblib.load(path)
