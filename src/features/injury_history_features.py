"""
injury_history_features.py

Derives features from a pitcher's prior injury history. Prior injury is
one of the strongest known predictors of future injury, and the nature,
recency, and frequency of past injuries all carry signal.

Feature categories:
- Career injury count (total and by body part)
- Days since last injury (recency)
- Days lost in previous IL stint (severity)
- Career total days lost
- Injury frequency rate
"""

from __future__ import annotations

import numpy as np
import pandas as pd

INJURY_TYPES = [
    "elbow", "shoulder", "forearm", "back", "oblique",
    "hamstring", "hip", "knee", "finger_hand", "calf_ankle",
    "neck", "illness",
]


def compute_injury_counts(
    game_df: pd.DataFrame,
    injuries_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute cumulative prior IL stints for each pitcher up to each game date.

    For each pitcher-game row, counts all prior IL placements (before that
    game date) — total and split by injury category. This is strictly
    historical: no future injury information leaks into the features.

    Args:
        game_df: Game-level DataFrame with columns pitcher (int), game_date.
        injuries_df: Injury database with columns player_id (int),
            transaction_date (datetime), injury_type (str).

    Returns:
        game_df with new columns:
            prior_il_total, prior_il_{type} for each injury type.
    """
    df = game_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    inj = injuries_df.copy()
    inj["transaction_date"] = pd.to_datetime(inj["transaction_date"])
    inj = inj.rename(columns={"player_id": "pitcher"})

    # Cross-join all (pitcher, game_date) × injury rows for that pitcher
    merged = df[["pitcher", "game_date"]].merge(
        inj[["pitcher", "transaction_date", "injury_type"]],
        on="pitcher", how="left"
    )
    # Keep only prior injuries (strictly before this game)
    merged = merged[merged["transaction_date"] < merged["game_date"]]

    # Total prior IL stints
    total_counts = (
        merged.groupby(["pitcher", "game_date"])["transaction_date"]
        .count()
        .reset_index(name="prior_il_total")
    )
    df = df.merge(total_counts, on=["pitcher", "game_date"], how="left")
    df["prior_il_total"] = df["prior_il_total"].fillna(0).astype(int)

    # Per-type counts
    for inj_type in INJURY_TYPES:
        type_counts = (
            merged[merged["injury_type"] == inj_type]
            .groupby(["pitcher", "game_date"])["transaction_date"]
            .count()
            .reset_index(name=f"prior_il_{inj_type}")
        )
        df = df.merge(type_counts, on=["pitcher", "game_date"], how="left")
        df[f"prior_il_{inj_type}"] = df[f"prior_il_{inj_type}"].fillna(0).astype(int)

    return df


def compute_days_since_last_injury(
    game_df: pd.DataFrame,
    injuries_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the days elapsed since a pitcher's most recent IL placement.

    Returns NaN for pitchers with no prior injury history. A short
    days_since_last_injury can flag a pitcher returning from a recent IL stint.

    Args:
        game_df: Game-level DataFrame with pitcher, game_date.
        injuries_df: Injury database with player_id, transaction_date.

    Returns:
        game_df with new column days_since_last_injury.
    """
    df = game_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    inj = injuries_df.copy()
    inj["transaction_date"] = pd.to_datetime(inj["transaction_date"])
    inj = inj.rename(columns={"player_id": "pitcher"})

    merged = df[["pitcher", "game_date"]].merge(
        inj[["pitcher", "transaction_date", "days_lost"]],
        on="pitcher", how="left"
    )
    merged = merged[merged["transaction_date"] < merged["game_date"]]
    merged["gap"] = (merged["game_date"] - merged["transaction_date"]).dt.days

    # Most recent prior IL placement per pitcher-game
    last_inj = (
        merged.sort_values("gap")
        .groupby(["pitcher", "game_date"])
        .first()
        .reset_index()[["pitcher", "game_date", "gap", "days_lost"]]
    )
    last_inj.columns = ["pitcher", "game_date",
                         "days_since_last_injury", "prior_il_days_lost"]

    df = df.merge(last_inj, on=["pitcher", "game_date"], how="left")
    return df


def build_injury_history_features(
    game_df: pd.DataFrame,
    injuries_df: pd.DataFrame,
) -> pd.DataFrame:
    """Orchestrate all injury history feature computations.

    Args:
        game_df: Game-level pitcher DataFrame with pitcher, game_date.
        injuries_df: Injury database from injury_loader.

    Returns:
        game_df with all injury history feature columns appended.
    """
    df = compute_injury_counts(game_df, injuries_df)
    df = compute_days_since_last_injury(df, injuries_df)
    return df
