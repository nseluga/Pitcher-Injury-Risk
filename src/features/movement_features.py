"""
movement_features.py

Computes pitch movement and mechanics-related features from Statcast data.
Changes in movement profiles can signal mechanical adjustments or arm
fatigue that precede injury.

Feature categories:
- Horizontal and vertical break (mean per game)
- Spin rate by pitch type (mean, rolling, delta vs. season baseline)
- Release point consistency (within-game std of release_pos_x, release_pos_z)
- Extension trends
- Movement deltas vs. 30-day rolling baseline
"""

from __future__ import annotations

import numpy as np
import pandas as pd

FASTBALL_TYPES = {"FF", "SI", "FC"}


def compute_game_movement_stats(pitch_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate movement and mechanics metrics to the game level.

    Computes mean horizontal break (pfx_x), induced vertical break (pfx_z),
    spin rate, release point position, and release extension per pitcher-game.
    Splits fastball metrics from all-pitch metrics.

    Args:
        pitch_df: Pitch-level DataFrame with Statcast movement columns.

    Returns:
        Game-level DataFrame with movement summary columns.
    """
    df = pitch_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    agg_all = (
        df.groupby(["pitcher", "game_date"])
        .agg(
            pfx_x_mean        = ("pfx_x",             "mean"),
            pfx_z_mean        = ("pfx_z",             "mean"),
            spin_rate_mean    = ("release_spin_rate",  "mean"),
            extension_mean    = ("release_extension",  "mean"),
            rel_x_mean        = ("release_pos_x",      "mean"),
            rel_z_mean        = ("release_pos_z",      "mean"),
        )
        .reset_index()
    )

    # Release point consistency: within-game std (proxy for mechanical stability)
    agg_consistency = (
        df.groupby(["pitcher", "game_date"])
        .agg(
            rel_x_std  = ("release_pos_x", "std"),
            rel_z_std  = ("release_pos_z", "std"),
            spin_std   = ("release_spin_rate", "std"),
        )
        .reset_index()
    )

    # Fastball-specific spin rate
    fb = df[df["pitch_type"].isin(FASTBALL_TYPES)]
    fb_spin = (
        fb.groupby(["pitcher", "game_date"])["release_spin_rate"]
        .mean()
        .reset_index(name="fb_spin_mean")
    )

    out = agg_all.merge(agg_consistency, on=["pitcher", "game_date"], how="left")
    out = out.merge(fb_spin, on=["pitcher", "game_date"], how="left")
    return out


def compute_rolling_movement(
    game_df: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """Compute rolling averages of key movement metrics.

    Tracks the 30-day baseline for spin rate, break, and release point so
    that delta features can detect drift away from the pitcher's norm.

    Args:
        game_df: Game-level movement DataFrame from compute_game_movement_stats.
        windows: Lookback windows in days. Defaults to [30].

    Returns:
        game_df with rolling movement columns appended.
    """
    if windows is None:
        windows = [30]

    roll_cols = ["pfx_x_mean", "pfx_z_mean", "spin_rate_mean",
                 "extension_mean", "rel_x_mean", "rel_z_mean",
                 "fb_spin_mean"]
    roll_cols = [c for c in roll_cols if c in game_df.columns]

    df = game_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])
    df_s = df.sort_values(["pitcher", "game_date"]).set_index("game_date")

    for w in windows:
        for col in roll_cols:
            result = (
                df_s.groupby("pitcher")[col]
                .rolling(f"{w}D", min_periods=2, closed="left")
                .mean()
                .reset_index()
            )
            result.columns = ["pitcher", "game_date", f"{col}_{w}d_avg"]
            df = df.merge(result, on=["pitcher", "game_date"], how="left")

    return df


def compute_movement_deltas(game_df: pd.DataFrame) -> pd.DataFrame:
    """Compute deviation of current game from the 30-day rolling movement baseline.

    Spin rate drop and release point drift are the two most clinically
    meaningful signals here.

    Args:
        game_df: Game-level movement DataFrame with 30d rolling columns.

    Returns:
        game_df with delta columns appended.
    """
    df = game_df.copy()
    delta_pairs = [
        ("spin_rate_mean",  "spin_rate_mean_30d_avg",  "spin_rate_delta_30d"),
        ("fb_spin_mean",    "fb_spin_mean_30d_avg",    "fb_spin_delta_30d"),
        ("pfx_z_mean",      "pfx_z_mean_30d_avg",      "pfx_z_delta_30d"),
        ("pfx_x_mean",      "pfx_x_mean_30d_avg",      "pfx_x_delta_30d"),
        ("extension_mean",  "extension_mean_30d_avg",  "extension_delta_30d"),
        ("rel_x_mean",      "rel_x_mean_30d_avg",      "rel_x_drift_30d"),
        ("rel_z_mean",      "rel_z_mean_30d_avg",      "rel_z_drift_30d"),
    ]
    for cur, base, delta in delta_pairs:
        if cur in df.columns and base in df.columns:
            df[delta] = df[cur] - df[base]

    # Euclidean release point drift (combined X + Z drift magnitude)
    if "rel_x_drift_30d" in df.columns and "rel_z_drift_30d" in df.columns:
        df["release_drift_30d"] = np.sqrt(
            df["rel_x_drift_30d"] ** 2 + df["rel_z_drift_30d"] ** 2
        )

    return df


def build_movement_features(pitch_df: pd.DataFrame) -> pd.DataFrame:
    """Orchestrate all movement feature computations.

    Args:
        pitch_df: Raw pitch-level Statcast DataFrame with pitcher, game_date,
            pfx_x, pfx_z, release_spin_rate, release_pos_x, release_pos_z,
            release_extension columns.

    Returns:
        Game-level DataFrame with all movement feature columns.
    """
    game_mov = compute_game_movement_stats(pitch_df)
    game_mov = compute_rolling_movement(game_mov, windows=[30])
    game_mov = compute_movement_deltas(game_mov)
    return game_mov
