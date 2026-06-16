"""
survival_models.py

Time-to-event models for predicting when a pitcher will next be injured.
Survival analysis is the statistically correct framework for this problem
because it handles censored observations — pitchers who have not yet been
injured within the observation window.

Approaches to implement:
1. Cox Proportional Hazards (interpretable baseline survival model)
2. Accelerated Failure Time (AFT) models (Weibull, log-normal)
3. Random Survival Forests (non-parametric, captures interactions)
4. DeepHit / DeepSurv (neural survival models, future phase)

Key concepts:
- Event: IL placement
- Time: days from last appearance (or season start) until IL placement
- Censoring: pitcher reaches end of observation window without an IL stint
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from src.models.baseline_models import _infer_feature_cols

DEFAULT_HORIZON_DAYS = 90
_MAX_COX_ROWS = 12_000   # Cox PH log-partial-likelihood is O(n*D*p); cap to keep fit <5 min on a laptop
_MAX_RSF_ROWS = 12_000   # RSF with n_jobs=-1 is fast at 12K; larger sizes caused 2-hr timeouts
_MAX_COX_FEATURES = 20   # feature pre-selection cap for Cox PH (reduces Hessian dimension)


def _subsample(
    X: pd.DataFrame,
    T: pd.Series,
    E: pd.Series,
    max_rows: int,
    random_state: int = 42,
) -> tuple:
    """Stratified subsample of (X, T, E) keeping events + random censored rows.

    Cox PH and AFT models are consistent estimators on subsamples — no bias
    introduced, just reduced wall time from hours to minutes on a laptop.
    When events alone exceed max_rows, events are subsampled too.
    """
    if len(X) <= max_rows:
        return X, T, E
    rng = np.random.default_rng(random_state)
    idx_event = np.where(E.values == 1)[0]
    idx_censor = np.where(E.values == 0)[0]
    if len(idx_event) >= max_rows:
        # More events than the cap — subsample events only (unusual but handled)
        idx_keep = rng.permutation(rng.choice(idx_event, size=max_rows, replace=False))
        n_evt, n_cen = max_rows, 0
    else:
        n_cen = min(max_rows - len(idx_event), len(idx_censor))
        idx_censor_s = rng.choice(idx_censor, size=n_cen, replace=False)
        idx_keep = rng.permutation(np.concatenate([idx_event, idx_censor_s]))
        n_evt = len(idx_event)
    print(
        f"  [subsample] {len(X):,} → {len(idx_keep):,} rows "
        f"({n_evt:,} events + {n_cen:,} censored)"
    )
    return X.iloc[idx_keep], T.iloc[idx_keep], E.iloc[idx_keep]


def _select_top_features(
    X: pd.DataFrame,
    E: pd.Series,
    n: int = _MAX_COX_FEATURES,
) -> list[str]:
    """Select top-n features by absolute correlation with the event indicator.

    Fast univariate filter that shrinks the Cox PH design matrix before fitting.
    Fewer features mean a smaller Hessian and dramatically faster Newton-Raphson.
    Cox PH is still meaningful on 20 well-selected features.
    """
    corr = X.corrwith(E.astype(float)).abs().sort_values(ascending=False)
    selected = corr.dropna().head(n).index.tolist()
    if not selected:
        return list(X.columns[:n])
    return selected


def prepare_survival_dataset(
    master_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    test_seasons: list[int] | None = None,
) -> tuple:
    """Prepare the dataset for survival analysis.

    Formats the master dataset for lifelines / scikit-survival, constructing
    the (duration, event_observed) pair for each pitcher-game row.

    `days_to_next_injury` is the duration to the next IL placement. Rows for
    which no future injury is observed within the dataset's horizon are
    right-censored: duration is capped at DEFAULT_HORIZON_DAYS and the event
    indicator is 0.

    Args:
        master_df: Master modeling DataFrame with injury label columns.
        feature_cols: Feature columns to include.
        test_seasons: Seasons to hold out for evaluation.

    Returns:
        Tuple of (X_train, X_test, T_train, T_test, E_train, E_test) where
        T is duration and E is the event indicator.
    """
    if feature_cols is None:
        feature_cols = _infer_feature_cols(master_df)

    df = master_df.copy()

    has_event = df["days_to_next_injury"].notna() & (df["days_to_next_injury"] <= DEFAULT_HORIZON_DAYS)
    duration = df["days_to_next_injury"].where(has_event, DEFAULT_HORIZON_DAYS)
    duration = duration.clip(lower=1)  # lifelines/scikit-survival require strictly positive durations

    df["_duration"] = duration.astype(float)
    df["_event"] = has_event.astype(int)

    seasons = sorted(df["season"].unique())
    if test_seasons is None:
        test_seasons = seasons[-2:] if len(seasons) > 1 else seasons[-1:]
    is_test = df["season"].isin(test_seasons)

    train_df, test_df = df.loc[~is_test], df.loc[is_test]

    X_train = train_df[feature_cols].copy()
    X_test = test_df[feature_cols].copy()
    T_train, T_test = train_df["_duration"].copy(), test_df["_duration"].copy()
    E_train, E_test = train_df["_event"].copy(), test_df["_event"].copy()

    return X_train, X_test, T_train, T_test, E_train, E_test


def _impute(X_train: pd.DataFrame, X_test: pd.DataFrame | None = None):
    """Median-impute features (lifelines/scikit-survival cannot handle NaNs).

    `keep_empty_features=True` so all-NaN columns (e.g. `intragame_velo_drop`,
    which is currently all-NaN across the full feature matrix) are kept as a
    constant 0 column rather than silently dropped — preserving the column
    count expected when reattaching `X_train.columns`. `_drop_zero_variance`
    removes the resulting constant columns afterwards.
    """
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    X_train_imp = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    if X_test is None:
        return X_train_imp, imputer
    X_test_imp = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns, index=X_test.index)
    return X_train_imp, X_test_imp, imputer


def _drop_zero_variance(X: pd.DataFrame) -> pd.DataFrame:
    """Drop constant columns — they break Cox/AFT design-matrix inversion."""
    keep = X.columns[X.nunique(dropna=False) > 1]
    return X[keep]


def train_cox_ph(
    X_train: pd.DataFrame,
    T_train: pd.Series,
    E_train: pd.Series,
    max_train_rows: int = _MAX_COX_ROWS,
    n_features: int = _MAX_COX_FEATURES,
) -> object:
    """Train a Cox Proportional Hazards model using lifelines.

    Args:
        X_train: Training feature matrix.
        T_train: Duration (days until event or censoring).
        E_train: Event indicator (1 = injured, 0 = censored).
        max_train_rows: Cap training rows via stratified subsampling (Cox PH
            scales poorly — O(n*D) partial likelihood per iteration).
        n_features: Pre-select top-n features by event correlation before
            fitting. Reduces Hessian dimension and speeds Newton-Raphson
            from hours to seconds on a laptop.

    Returns:
        Fitted lifelines CoxPHFitter object. The imputer used to fill missing
        values is stashed on `model._imputer_` and the trained column order on
        `model._feature_cols_` so downstream prediction can replicate preprocessing.
    """
    from lifelines import CoxPHFitter

    X_train, T_train, E_train = _subsample(X_train, T_train, E_train, max_train_rows)

    X_imp, imputer = _impute(X_train)
    X_imp = _drop_zero_variance(X_imp)

    if n_features and len(X_imp.columns) > n_features:
        selected = _select_top_features(X_imp, E_train, n=n_features)
        X_imp = X_imp[selected]
        print(f"  [feature select] {len(X_train.columns)} → {len(selected)} features for Cox PH")

    fit_df = X_imp.copy()
    fit_df["_duration"] = T_train.values
    fit_df["_event"] = E_train.values

    model = CoxPHFitter(penalizer=0.1)
    model.fit(fit_df, duration_col="_duration", event_col="_event")
    model._imputer_ = imputer
    model._feature_cols_ = list(X_imp.columns)
    return model


def train_aft_model(
    X_train: pd.DataFrame,
    T_train: pd.Series,
    E_train: pd.Series,
    distribution: str = "weibull",
    max_train_rows: int = _MAX_COX_ROWS,
) -> object:
    """Train an Accelerated Failure Time model.

    Args:
        X_train: Training feature matrix.
        T_train: Duration series.
        E_train: Event indicator series.
        distribution: Parametric distribution to use ('weibull', 'lognormal',
            'loglogistic').
        max_train_rows: Subsample cap for laptop-safe training.

    Returns:
        Fitted lifelines AFT model object, with `_imputer_`/`_feature_cols_`
        attached as in train_cox_ph.
    """
    from lifelines import LogLogisticAFTFitter, LogNormalAFTFitter, WeibullAFTFitter

    fitters = {
        "weibull": WeibullAFTFitter,
        "lognormal": LogNormalAFTFitter,
        "loglogistic": LogLogisticAFTFitter,
    }
    if distribution not in fitters:
        raise ValueError(f"Unknown distribution: {distribution!r} (expected one of {list(fitters)})")

    X_train, T_train, E_train = _subsample(X_train, T_train, E_train, max_train_rows)

    X_imp, imputer = _impute(X_train)
    X_imp = _drop_zero_variance(X_imp)

    fit_df = X_imp.copy()
    fit_df["_duration"] = T_train.values
    fit_df["_event"] = E_train.values

    model = fitters[distribution](penalizer=0.1)
    model.fit(fit_df, duration_col="_duration", event_col="_event")
    model._imputer_ = imputer
    model._feature_cols_ = list(X_imp.columns)
    return model


def train_random_survival_forest(
    X_train: pd.DataFrame,
    T_train: pd.Series,
    E_train: pd.Series,
    max_train_rows: int = _MAX_RSF_ROWS,
) -> object:
    """Train a Random Survival Forest using scikit-survival.

    Captures non-linear feature interactions and does not require the
    proportional hazards assumption. Slower to train but often more accurate.

    Args:
        X_train: Training feature matrix.
        T_train: Duration series.
        E_train: Event indicator series.
        max_train_rows: Subsample cap for laptop-safe training.

    Returns:
        Fitted scikit-survival RandomSurvivalForest object, with
        `_imputer_`/`_feature_cols_` attached.
    """
    from sksurv.ensemble import RandomSurvivalForest

    X_train, T_train, E_train = _subsample(X_train, T_train, E_train, max_train_rows)

    X_imp, imputer = _impute(X_train)
    X_imp = _drop_zero_variance(X_imp)

    y = np.array(
        [(bool(e), t) for e, t in zip(E_train.values, T_train.values)],
        dtype=[("event", bool), ("time", float)],
    )

    model = RandomSurvivalForest(
        n_estimators=100,
        max_depth=6,
        min_samples_leaf=20,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_imp, y)
    model._imputer_ = imputer
    model._feature_cols_ = list(X_imp.columns)
    return model


def _prepare_X_for_predict(model: object, X: pd.DataFrame) -> pd.DataFrame:
    """Replay the imputation/column-selection used at training time."""
    imputer = getattr(model, "_imputer_", None)
    feature_cols = getattr(model, "_feature_cols_", list(X.columns))
    if imputer is not None:
        X_imp = pd.DataFrame(imputer.transform(X), columns=X.columns, index=X.index)
    else:
        X_imp = X
    return X_imp[feature_cols]


def predict_survival_function(
    model: object,
    X: pd.DataFrame,
    time_points: list[int] | None = None,
) -> pd.DataFrame:
    """Compute survival probability at specified time horizons for each row.

    Args:
        model: A fitted survival model (Cox, AFT, or RSF).
        X: Feature matrix for which to generate predictions.
        time_points: Days-ahead horizons to evaluate. Defaults to [30, 60, 90].

    Returns:
        DataFrame with one row per observation, columns S(t) for each time_point.
    """
    if time_points is None:
        time_points = [30, 60, 90]

    X_imp = _prepare_X_for_predict(model, X)

    type_name = type(model).__name__
    if type_name == "RandomSurvivalForest":
        surv_funcs = model.predict_survival_function(X_imp)
        data = {f"S({t})": [float(fn(t)) for fn in surv_funcs] for t in time_points}
    else:
        # lifelines Cox/AFT models share `predict_survival_function`, returning
        # a (times x n_samples) DataFrame.
        surv_df = model.predict_survival_function(X_imp, times=time_points)
        data = {f"S({t})": surv_df.loc[t].values for t in time_points}

    return pd.DataFrame(data, index=X.index)


def compute_concordance_index(
    model: object,
    X_test: pd.DataFrame,
    T_test: pd.Series,
    E_test: pd.Series,
) -> float:
    """Compute the concordance index (C-index) for a survival model.

    The C-index is the primary evaluation metric for survival models.
    0.5 = random, 1.0 = perfect.

    Args:
        model: Fitted survival model.
        X_test: Test feature matrix.
        T_test: Test duration series.
        E_test: Test event indicator series.

    Returns:
        C-index float.
    """
    from lifelines.utils import concordance_index

    X_imp = _prepare_X_for_predict(model, X_test)
    type_name = type(model).__name__

    if type_name == "RandomSurvivalForest":
        risk_scores = model.predict(X_imp)  # higher = higher risk
        return float(concordance_index(T_test, -risk_scores, E_test))

    # lifelines models expose partial_hazard / median survival time as a
    # risk proxy; predict_expectation gives expected survival time (lower
    # expected survival time == higher risk), which concordance_index expects
    # as the "predicted_scores" argument with the sign convention below.
    predicted_time = model.predict_expectation(X_imp)
    return float(concordance_index(T_test, predicted_time, E_test))
