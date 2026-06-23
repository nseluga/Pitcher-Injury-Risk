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
_MAX_COX_ROWS = 45_000   # Must exceed event count (~18K) so _subsample keeps censored rows alongside events.
                         # At 45K: ~18K events + ~27K censored (40% event rate). Cox with 20 features scales OK.
_MAX_RSF_ROWS = 35_000   # Same reasoning: previously 12K < 18K events → 0 censored in training (broken).
_MAX_COX_FEATURES = 30   # feature pre-selection cap for Cox PH; 30 captures both injury-history
                         # (prior_il_total is strongest predictor) and key workload features.


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


ARM_INJURY_TYPES = ["elbow", "shoulder", "forearm"]


def prepare_survival_dataset(
    master_df: pd.DataFrame,
    feature_cols: list[str] | None = None,
    test_seasons: list[int] | None = None,
    event_injury_types: list[str] | None = None,
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
        event_injury_types: If provided, only count injuries of these types as events.
            Non-matching injuries within the horizon are treated as censored (cause-specific
            survival analysis). Pass ARM_INJURY_TYPES to focus on arm/shoulder/elbow injuries
            and remove noise from oblique strains, hamstrings, etc.

    Returns:
        Tuple of (X_train, X_test, T_train, T_test, E_train, E_test) where
        T is duration and E is the event indicator.
    """
    if feature_cols is None:
        feature_cols = _infer_feature_cols(master_df)

    df = master_df.copy()

    has_event = df["days_to_next_injury"].notna() & (df["days_to_next_injury"] <= DEFAULT_HORIZON_DAYS)

    if event_injury_types is not None and "next_injury_type" in df.columns:
        arm_mask = df["next_injury_type"].isin(event_injury_types)
        has_event = has_event & arm_mask

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
    strata: list[str] | None = None,
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
        strata: Column(s) to stratify the baseline hazard on. Each unique
            combination of strata values gets its own baseline hazard function,
            relaxing the proportional hazards assumption for those columns.
            Strata columns are included in the fit DataFrame but not estimated
            as regression coefficients.

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
        strata_set = set(strata or [])
        selected = _select_top_features(X_imp, E_train, n=n_features)
        # Always include strata columns even if they fall outside the top-n
        for col in strata_set:
            if col not in selected and col in X_imp.columns:
                selected.append(col)
        X_imp = X_imp[selected]
        print(f"  [feature select] {len(X_train.columns)} → {len(X_imp.columns)} features for Cox PH"
              + (f" (strata={strata})" if strata else ""))

    fit_df = X_imp.copy()
    fit_df["_duration"] = T_train.values
    fit_df["_event"] = E_train.values

    model = CoxPHFitter(penalizer=0.1)
    model.fit(fit_df, duration_col="_duration", event_col="_event", strata=strata)
    model._imputer_ = imputer
    model._feature_cols_ = list(X_imp.columns)
    model._strata_ = strata
    # Stash a copy of the fit DataFrame so check_ph_assumption() can run
    # Schoenfeld residual tests without re-running preprocessing.
    model._fit_df_ = fit_df.copy()
    return model


def check_ph_assumption(cox_model: object, p_value_threshold: float = 0.05) -> None:
    """Test the proportional hazards assumption using Schoenfeld residuals.

    Prints a summary of features whose hazard ratio appears to vary over time
    (i.e. where the PH assumption is violated). A flat Schoenfeld residual plot
    confirms the assumption holds. A trend indicates a time-varying hazard — if
    many features violate PH, a Weibull AFT model (which lacks this assumption)
    should be preferred.

    Args:
        cox_model: A CoxPHFitter fitted by train_cox_ph (must have _fit_df_).
        p_value_threshold: Features with p < threshold are flagged as violations.
    """
    if not hasattr(cox_model, "_fit_df_"):
        print("PH assumption check skipped — _fit_df_ not found. Re-fit with train_cox_ph.")
        return
    try:
        print(f"Proportional Hazards Assumption Test (Schoenfeld residuals, α={p_value_threshold}):")
        cox_model.check_assumptions(cox_model._fit_df_, p_value_threshold=p_value_threshold, show_plots=False)
    except Exception as exc:
        print(f"PH assumption test failed: {exc}")


def train_aft_model(
    X_train: pd.DataFrame,
    T_train: pd.Series,
    E_train: pd.Series,
    distribution: str = "weibull",
    max_train_rows: int = _MAX_COX_ROWS,
    n_features: int = _MAX_COX_FEATURES,
) -> object:
    """Train an Accelerated Failure Time model.

    Does not require the proportional hazards assumption — a key advantage over
    Cox PH when injury hazard rates change over time (e.g. early-season vs.
    late-season risk profiles differ). Uses the same feature pre-selection and
    subsampling caps as train_cox_ph for laptop safety.

    Args:
        X_train: Training feature matrix.
        T_train: Duration series.
        E_train: Event indicator series.
        distribution: Parametric distribution to use ('weibull', 'lognormal',
            'loglogistic').
        max_train_rows: Subsample cap for laptop-safe training.
        n_features: Pre-select top-n features by event correlation (same filter
            as Cox PH). Reduces fitting time on high-dimensional inputs.

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

    orig_n = len(X_imp.columns)
    if n_features and orig_n > n_features:
        selected = _select_top_features(X_imp, E_train, n=n_features)
        X_imp = X_imp[selected]
        print(f"  [feature select] {orig_n} → {len(selected)} features for AFT ({distribution})")

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


def train_extra_survival_trees(
    X_train: pd.DataFrame,
    T_train: pd.Series,
    E_train: pd.Series,
    max_train_rows: int = _MAX_RSF_ROWS,
    n_estimators: int = 100,
    max_depth: int = 6,
    min_samples_leaf: int = 20,
) -> object:
    """Train an Extra Survival Trees ensemble using scikit-survival.

    ExtraSurvivalTrees extends RSF by using random (rather than optimal) split
    thresholds for each feature, plus the log-rank criterion for split quality.
    The additional randomization produces trees that are more decorrelated from
    GBSA (which also uses optimal splits) — increasing ensemble diversity beyond
    what RSF already provides. Same interface as RandomSurvivalForest.

    Args:
        X_train: Training feature matrix.
        T_train: Duration series.
        E_train: Event indicator series.
        max_train_rows: Subsample cap (same default as RSF).
        n_estimators: Number of trees.
        max_depth: Maximum tree depth.
        min_samples_leaf: Minimum samples required at a leaf node.

    Returns:
        Fitted ExtraSurvivalTrees object with `_imputer_` and `_feature_cols_` attached.
    """
    from sksurv.ensemble import ExtraSurvivalTrees

    X_train, T_train, E_train = _subsample(X_train, T_train, E_train, max_train_rows)

    X_imp, imputer = _impute(X_train)
    X_imp = _drop_zero_variance(X_imp)

    y = np.array(
        [(bool(e), t) for e, t in zip(E_train.values, T_train.values)],
        dtype=[("event", bool), ("time", float)],
    )

    model = ExtraSurvivalTrees(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features="sqrt",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_imp, y)
    model._imputer_ = imputer
    model._feature_cols_ = list(X_imp.columns)
    return model


def train_gradient_boosted_survival(
    X_train: pd.DataFrame,
    T_train: pd.Series,
    E_train: pd.Series,
    max_train_rows: int = _MAX_RSF_ROWS,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 2,
    min_samples_leaf: int = 20,
    subsample: float = 0.8,
    max_features: str | float | int | None = None,
    loss: str = "coxph",
) -> object:
    """Train a Gradient Boosted Survival Analysis model using scikit-survival.

    GradientBoostingSurvivalAnalysis uses componentwise gradient boosting with
    Cox partial likelihood as the loss. Unlike Cox PH, it captures nonlinear
    relationships and feature interactions without requiring the proportional
    hazards assumption at prediction time. Uses all available features — no
    pre-selection needed since boosting handles high-dimensional inputs naturally.

    Args:
        X_train: Training feature matrix.
        T_train: Duration series.
        E_train: Event indicator series.
        max_train_rows: Subsample cap (same default as RSF for comparable compute).
        n_estimators: Number of boosting stages.
        learning_rate: Shrinkage factor applied to each tree.
        max_depth: Maximum depth of individual regression estimators.
        min_samples_leaf: Minimum samples required at a leaf node.
        subsample: Fraction of samples used per base learner (< 1.0 enables
            Stochastic Gradient Boosting — reduces variance, improves
            generalization). Friedman (2002) recommends 0.5–0.8 for tabular data.
        max_features: Number of features considered when looking for the best split
            per tree ('sqrt', 'log2', float fraction, int count, or None for all).
            Column subsampling adds feature diversity on top of row subsampling
            (subsample), creating "double stochastic" boosting. Particularly
            effective when features are correlated or redundant.
        loss: Loss function to optimize. 'coxph' (default) uses Cox partial
            likelihood — predict() returns log-hazard (higher = more risk).
            'ipcwls' uses Inverse Probability of Censoring Weighted Least Squares —
            an AFT-style objective that doesn't require the PH assumption and
            corrects for censoring bias; predict() returns log-time (higher =
            longer survival = lower risk). Sign convention differs — see
            compute_concordance_index for handling. 'squared' uses squared
            regression loss on log-time without censoring correction.

    Returns:
        Fitted GradientBoostingSurvivalAnalysis object with `_imputer_`,
        `_feature_cols_`, and `_loss_` attached for prediction replay.
    """
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis

    X_train, T_train, E_train = _subsample(X_train, T_train, E_train, max_train_rows)

    X_imp, imputer = _impute(X_train)
    X_imp = _drop_zero_variance(X_imp)

    y = np.array(
        [(bool(e), t) for e, t in zip(E_train.values, T_train.values)],
        dtype=[("event", bool), ("time", float)],
    )

    model = GradientBoostingSurvivalAnalysis(
        loss=loss,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        subsample=subsample,
        max_features=max_features,
        random_state=42,
    )
    model.fit(X_imp, y)
    model._imputer_ = imputer
    model._feature_cols_ = list(X_imp.columns)
    model._loss_ = loss  # needed by compute_concordance_index for sign convention
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
    if type_name in ("RandomSurvivalForest", "GradientBoostingSurvivalAnalysis"):
        # sksurv step-functions: each element is callable at any time point.
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

    # sksurv tree-based models: predict() sign depends on loss function.
    # coxph / rsf: predict() returns cumulative hazard or log-hazard — higher = more risk.
    #   concordance_index expects higher score = longer survival → negate.
    # ipcwls: predict() returns log-time — higher = longer survival (lower risk).
    #   concordance_index expects higher score = longer survival → no negation.
    if type_name in ("RandomSurvivalForest", "GradientBoostingSurvivalAnalysis", "ExtraSurvivalTrees"):
        raw = model.predict(X_imp)
        loss = getattr(model, "_loss_", "coxph")
        survival_proxy = raw if loss == "ipcwls" else -raw
        return float(concordance_index(T_test, survival_proxy, E_test))

    # lifelines models expose partial_hazard / median survival time as a
    # risk proxy; predict_expectation gives expected survival time (lower
    # expected survival time == higher risk), which concordance_index expects
    # as the "predicted_scores" argument with the sign convention below.
    predicted_time = model.predict_expectation(X_imp)
    return float(concordance_index(T_test, predicted_time, E_test))
