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

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from src.models.baseline_models import _infer_feature_cols

CLASSIFICATION_TASKS = ["injured_next_30d", "injured_next_60d", "injured_next_90d"]
REGRESSION_TASKS = ["days_to_next_injury", "next_injury_days_lost"]
MULTICLASS_TASKS = ["next_injury_type"]
ALL_TASKS = CLASSIFICATION_TASKS + REGRESSION_TASKS + MULTICLASS_TASKS


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
    if feature_cols is None:
        feature_cols = _infer_feature_cols(master_df)

    seasons = sorted(master_df["season"].unique())
    if test_seasons is None:
        test_seasons = seasons[-2:] if len(seasons) > 1 else seasons[-1:]
    is_test = master_df["season"].isin(test_seasons)

    train_df, test_df = master_df.loc[~is_test], master_df.loc[is_test]

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    y_train_dict = {task: train_df[task].copy() for task in ALL_TASKS}
    y_test_dict = {task: test_df[task].copy() for task in ALL_TASKS}

    return X_train, X_test, y_train_dict, y_test_dict


def _regression_subset(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.Series]:
    """Severity/time-to-event targets are only observed for injured pitchers."""
    mask = y.notna()
    return X.loc[mask], y.loc[mask]


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
        Dictionary mapping task name to fitted model object. Each value is an
        sklearn Pipeline (imputer + estimator); regression/multiclass heads
        additionally receive `injury_prob_30d` as a chained input feature.
    """
    models: dict[str, object] = {}

    # 1) Probability heads (30/60/90d), trained directly on the raw features.
    for task in CLASSIFICATION_TASKS:
        y = y_train_dict[task]
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=5, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )),
        ])
        pipe.fit(X_train, y)
        models[task] = pipe

    # Chained feature: predicted P(injury within 30d), available for every row.
    chain_feature = models["injured_next_30d"].predict_proba(X_train)[:, 1]
    X_chained = X_train.copy()
    X_chained["injury_prob_30d"] = chain_feature

    # 2) Severity / time-to-event regressors — fit only on rows where the
    #    target is observed (i.e. the pitcher actually was later injured),
    #    using the chained probability as an extra signal.
    for task in REGRESSION_TASKS:
        X_sub, y_sub = _regression_subset(X_chained, y_train_dict[task])
        if len(y_sub) < 10:
            models[task] = None
            continue
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("reg", RandomForestRegressor(
                n_estimators=300, max_depth=5, min_samples_leaf=5,
                random_state=42, n_jobs=-1,
            )),
        ])
        pipe.fit(X_sub, y_sub)
        models[task] = pipe

    # 3) Injury type — multiclass, also only observed for injured pitchers.
    for task in MULTICLASS_TASKS:
        y = y_train_dict[task]
        X_sub, y_sub = _regression_subset(X_chained, y)
        if y_sub.nunique() < 2:
            models[task] = None
            continue
        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=5, min_samples_leaf=5,
                class_weight="balanced", random_state=42, n_jobs=-1,
            )),
        ])
        pipe.fit(X_sub, y_sub)
        models[task] = pipe

    models["_chained"] = True
    return models


def train_shared_representation_model(
    X_train: pd.DataFrame,
    y_train_dict: dict[str, pd.Series],
    shared_model_type: str = "gradient_boosting",
) -> object:
    """Train a multi-output model with a shared underlying representation.

    Uses sklearn's MultiOutputClassifier / MultiOutputRegressor or a custom
    gradient boosting approach where the shared tree structure handles all tasks.

    In practice this builds independently-fit task heads on top of a single
    shared imputation/preprocessing step — the closest practical analogue to a
    shared trunk available with tree ensembles, which can't natively share
    representations across mixed classification/regression/multiclass heads.

    Args:
        X_train: Training feature matrix.
        y_train_dict: Dictionary mapping task name to training target.
        shared_model_type: Base learner type ('gradient_boosting', 'random_forest').

    Returns:
        Fitted _SharedRepresentationModel object exposing predict_all_tasks-
        compatible `.tasks` attribute (dict of task -> fitted estimator).
    """
    if shared_model_type not in {"gradient_boosting", "random_forest"}:
        raise ValueError(f"Unknown shared_model_type: {shared_model_type!r}")

    return _SharedRepresentationModel(shared_model_type=shared_model_type).fit(X_train, y_train_dict)


class _SharedRepresentationModel:
    """Shared imputer/scaler trunk with independent per-task heads.

    Tree ensembles cannot literally share weights across heterogeneous task
    types (binary / regression / multiclass), so "sharing" here means a single
    fitted preprocessing trunk feeds every task-specific estimator — the
    standard practical compromise for hard-parameter-sharing with tree models.
    """

    def __init__(self, shared_model_type: str = "gradient_boosting"):
        self.shared_model_type = shared_model_type
        self.imputer = SimpleImputer(strategy="median")
        self.tasks: dict[str, object] = {}

    def _make_classifier(self):
        if self.shared_model_type == "gradient_boosting":
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=250, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, eval_metric="logloss",
                random_state=42, n_jobs=-1,
            )
        return RandomForestClassifier(
            n_estimators=300, max_depth=5, min_samples_leaf=5,
            class_weight="balanced", random_state=42, n_jobs=-1,
        )

    def _make_regressor(self):
        if self.shared_model_type == "gradient_boosting":
            from xgboost import XGBRegressor
            return XGBRegressor(
                n_estimators=250, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
            )
        return RandomForestRegressor(
            n_estimators=300, max_depth=5, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        )

    def fit(self, X_train: pd.DataFrame, y_train_dict: dict[str, pd.Series]) -> "_SharedRepresentationModel":
        X_imp = pd.DataFrame(
            self.imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index
        )

        for task in CLASSIFICATION_TASKS:
            est = self._make_classifier()
            est.fit(X_imp, y_train_dict[task])
            self.tasks[task] = est

        for task in REGRESSION_TASKS:
            X_sub, y_sub = _regression_subset(X_imp, y_train_dict[task])
            if len(y_sub) < 10:
                self.tasks[task] = None
                continue
            est = self._make_regressor()
            est.fit(X_sub, y_sub)
            self.tasks[task] = est

        for task in MULTICLASS_TASKS:
            X_sub, y_sub = _regression_subset(X_imp, y_train_dict[task])
            if y_sub.nunique() < 2:
                self.tasks[task] = None
                continue
            est = self._make_classifier()
            est.fit(X_sub, y_sub)
            self.tasks[task] = est

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame(self.imputer.transform(X), columns=X.columns, index=X.index)


def predict_all_tasks(
    models: dict[str, object] | object,
    X: pd.DataFrame,
) -> pd.DataFrame:
    """Run inference for all tasks and return a unified predictions DataFrame.

    Args:
        models: Either a dict of task-name → model (chained) or a single
            _SharedRepresentationModel (shared representation).
        X: Feature matrix.

    Returns:
        DataFrame with one prediction column per task, indexed like X. Binary
        tasks yield probabilities (`<task>_pred`), regression tasks yield
        point estimates, and the multiclass task yields the predicted label.
    """
    preds = pd.DataFrame(index=X.index)

    if isinstance(models, _SharedRepresentationModel):
        X_in = models.transform(X)
        task_models = models.tasks
        chained = False
    else:
        task_models = models
        chained = bool(task_models.get("_chained", False))
        X_in = X

    # Classification heads first (chained model needs the 30d head's output).
    for task in CLASSIFICATION_TASKS:
        est = task_models.get(task)
        if est is None:
            continue
        proba = est.predict_proba(X_in)[:, 1]
        preds[f"{task}_pred"] = proba

    if chained and "injured_next_30d_pred" in preds.columns:
        X_in = X_in.copy()
        X_in["injury_prob_30d"] = preds["injured_next_30d_pred"].values

    for task in REGRESSION_TASKS:
        est = task_models.get(task)
        if est is None:
            preds[f"{task}_pred"] = np.nan
            continue
        preds[f"{task}_pred"] = est.predict(X_in)

    for task in MULTICLASS_TASKS:
        est = task_models.get(task)
        if est is None:
            preds[f"{task}_pred"] = None
            continue
        preds[f"{task}_pred"] = est.predict(X_in)

    return preds


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
    from sklearn.metrics import accuracy_score, f1_score

    from src.models.evaluation import evaluate_classifier, evaluate_regression

    rows = []
    for task in CLASSIFICATION_TASKS:
        col = f"{task}_pred"
        if col not in y_pred_df.columns:
            continue
        m = evaluate_classifier(y_true_dict[task], y_pred_df[col])
        m["task"] = task
        m["task_type"] = "classification"
        rows.append(m)

    for task in REGRESSION_TASKS:
        col = f"{task}_pred"
        if col not in y_pred_df.columns:
            continue
        mask = y_true_dict[task].notna() & y_pred_df[col].notna()
        if mask.sum() < 5:
            continue
        m = evaluate_regression(y_true_dict[task][mask], y_pred_df[col][mask])
        m["task"] = task
        m["task_type"] = "regression"
        rows.append(m)

    for task in MULTICLASS_TASKS:
        col = f"{task}_pred"
        if col not in y_pred_df.columns:
            continue
        mask = y_true_dict[task].notna() & y_pred_df[col].notna()
        if mask.sum() < 5:
            continue
        rows.append({
            "task": task,
            "task_type": "multiclass",
            "accuracy": float(accuracy_score(y_true_dict[task][mask], y_pred_df[col][mask])),
            "f1_macro": float(f1_score(y_true_dict[task][mask], y_pred_df[col][mask], average="macro", zero_division=0)),
        })

    return pd.DataFrame(rows).set_index("task")
