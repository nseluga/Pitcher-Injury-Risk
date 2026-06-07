"""
pitch_mix_features.py

Computes features describing a pitcher's pitch mix composition and how
it changes over time. Pitch mix shifts can reflect strategic changes,
fatigue, mechanical adjustments, or attempts to hide injury.

Feature categories:
- Usage rate by pitch type (per game and rolling)
- Rolling change in pitch mix (delta vs. 30-day baseline)
- Pitch mix diversity index (Shannon entropy)
- Slider-heavy usage flag (associated with UCL stress)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

PITCH_GROUPS = {
    "fb_pct":       {"FF", "SI", "FC"},
    "sl_pct":       {"SL", "ST"},
    "cu_pct":       {"CU", "KC", "SV"},
    "ch_pct":       {"CH", "FS"},
    "breaking_pct": {"SL", "ST", "CU", "KC", "SV"},
}


def compute_game_pitch_mix(pitch_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-game pitch type usage rates for each pitcher.

    For each pitcher-game, calculates the proportion of pitches thrown for
    each pitch group: fastball, slider/sweeper, curveball/knuckle-curve,
    changeup/splitter, and all breaking balls combined.

    Args:
        pitch_df: Pitch-level DataFrame with columns pitcher (int), game_date
            (datetime), pitch_type (str).

    Returns:
        Game-level DataFrame with pitch count and usage rate columns.
    """
    df = pitch_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    # Total pitch count per game
    totals = (
        df.groupby(["pitcher", "game_date"])["pitch_type"]
        .count()
        .reset_index(name="pitch_count")
    )

    # Usage rates per pitch group
    for col, types in PITCH_GROUPS.items():
        counts = (
            df[df["pitch_type"].isin(types)]
            .groupby(["pitcher", "game_date"])["pitch_type"]
            .count()
            .reset_index(name=f"_n_{col}")
        )
        totals = totals.merge(counts, on=["pitcher", "game_date"], how="left")
        totals[f"_n_{col}"] = totals[f"_n_{col}"].fillna(0)
        totals[col] = (totals[f"_n_{col}"] / totals["pitch_count"].clip(lower=1)).round(4)
        totals.drop(columns=[f"_n_{col}"], inplace=True)

    # Shannon entropy of pitch mix (diversity)
    # Use the 5 raw pitch type proportions from the full distribution
    pitch_type_counts = (
        df.groupby(["pitcher", "game_date", "pitch_type"])["pitch_type"]
        .count()
        .unstack(fill_value=0)
        .reset_index()
    )
    all_types = [c for c in pitch_type_counts.columns
                 if c not in ["pitcher", "game_date"]]
    pt_matrix = pitch_type_counts[all_types].values.astype(float)
    row_sums = pt_matrix.sum(axis=1, keepdims=True).clip(min=1)
    probs = pt_matrix / row_sums
    # H = -sum(p * log(p)), masking zeros
    with np.errstate(divide="ignore", invalid="ignore"):
        log_probs = np.where(probs > 0, np.log(probs), 0)
    entropy = -(probs * log_probs).sum(axis=1)
    pitch_type_counts["pitch_mix_entropy"] = entropy

    totals = totals.merge(
        pitch_type_counts[["pitcher", "game_date", "pitch_mix_entropy"]],
        on=["pitcher", "game_date"], how="left"
    )

    return totals


def compute_rolling_pitch_mix(
    game_df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling average pitch usage rates over multiple lookback windows.

    Args:
        game_df: Game-level pitch mix DataFrame from compute_game_pitch_mix.
        windows: Lookback windows in days. Defaults to [30].

    Returns:
        game_df with rolling mix columns appended (e.g. sl_pct_30d_avg).
    """
    if windows is None:
        windows = [30]

    mix_cols = [c for c in PITCH_GROUPS if c in game_df.columns]
    df = game_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df_s = df.sort_values(["pitcher", "game_date"]).set_index("game_date")

    for w in windows:
        for col in mix_cols:
            result = (
                df_s.groupby("pitcher")[col]
                .rolling(f"{w}D", min_periods=1, closed="left")
                .mean()
                .reset_index()
            )
            result.columns = ["pitcher", "game_date", f"{col}_{w}d_avg"]
            df = df.merge(result, on=["pitcher", "game_date"], how="left")

    return df


def compute_pitch_mix_delta(game_df: pd.DataFrame) -> pd.DataFrame:
    """Compute deviation of current pitch mix from the 30-day rolling baseline.

    Large deviations can signal mechanical adjustments or attempts to
    protect an injured body part.

    Args:
        game_df: Game-level pitch mix DataFrame with 30d rolling columns.

    Returns:
        game_df with delta columns appended per pitch group.
    """
    df = game_df.copy()
    for col in PITCH_GROUPS:
        rolling_col = f"{col}_30d_avg"
        if col in df.columns and rolling_col in df.columns:
            df[f"{col}_delta_30d"] = df[col] - df[rolling_col]
    return df


def build_pitch_mix_features(pitch_df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrate all pitch mix feature computations.

    Args:
        pitch_df: Raw pitch-level Statcast DataFrame with pitcher, game_date,
            pitch_type columns.

    Returns:
        Game-level DataFrame with all pitch mix feature columns.
    """
    game_mix = compute_game_pitch_mix(pitch_df)
    game_mix = compute_rolling_pitch_mix(game_mix, windows=[30])
    game_mix = compute_pitch_mix_delta(game_mix)
    return game_mix
