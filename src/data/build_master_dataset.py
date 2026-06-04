"""
build_master_dataset.py

Orchestrates the full data pipeline: pulls raw data from all sources,
applies feature engineering, and joins everything into a single
analysis-ready master DataFrame. This is the entry point for reproducing
the dataset from scratch.

Pipeline stages:
1. Load Statcast pitch-level data (statcast_loader)
2. Load IL / injury data (injury_loader)
3. Load transaction data (transactions_loader)
4. Load player metadata (player_metadata from pybaseball/MLB Stats API)
5. Aggregate pitch-level data to game and season grain
6. Compute workload, velocity, movement, pitch-mix features (features/)
7. Attach injury history features
8. Attach injury labels (next IL stint, days lost, injury type)
9. Save master dataset to data/processed/
"""

from __future__ import annotations

import pandas as pd


def load_all_raw_data(
    start_year: int,
    end_year: int,
) -> dict[str, pd.DataFrame]:
    """Load all raw data sources and return them as a named dictionary.

    Calls each individual loader and returns a dict with keys:
    'statcast', 'injuries', 'transactions', 'player_metadata'.

    Args:
        start_year: First season to include.
        end_year: Last season to include, inclusive.

    Returns:
        Dictionary mapping source name to its DataFrame.
    """
    raise NotImplementedError


def aggregate_statcast_to_game(statcast_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pitch-level Statcast data to game-level summaries per pitcher.

    Computes per-game statistics including pitch counts by type, mean and max
    velocities, spin rates, movement profiles, and usage rates. This grain
    feeds rolling workload features.

    Args:
        statcast_df: Pitch-level DataFrame from statcast_loader.

    Returns:
        Game-level pitcher DataFrame with one row per (pitcher, game).
    """
    raise NotImplementedError


def aggregate_statcast_to_season(statcast_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate pitch-level Statcast data to season-level summaries per pitcher.

    Provides a complementary view to the game-level aggregation, capturing
    full-season mix, velocity trends, and high-level usage volume.

    Args:
        statcast_df: Pitch-level DataFrame from statcast_loader.

    Returns:
        Season-level pitcher DataFrame with one row per (pitcher, season).
    """
    raise NotImplementedError


def attach_injury_labels(
    game_df: pd.DataFrame,
    injuries_df: pd.DataFrame,
) -> pd.DataFrame:
    """Attach forward-looking injury labels to each pitcher-game row.

    For each row in game_df, looks ahead in time to find the pitcher's next
    IL stint. Creates label columns used as model targets:
    - days_until_next_injury (time-to-event)
    - next_injury_type
    - next_days_lost
    - injured_within_30d, injured_within_60d, injured_within_90d (binary flags)

    Args:
        game_df: Game-level pitcher DataFrame.
        injuries_df: Injury database from injury_loader.

    Returns:
        game_df augmented with injury label columns.
    """
    raise NotImplementedError


def build_master_dataset(
    start_year: int,
    end_year: int,
    save_path: str = "data/processed/master_dataset.parquet",
) -> pd.DataFrame:
    """Run the full pipeline and produce the master modeling dataset.

    Orchestrates all loading, aggregation, feature engineering, and label
    attachment steps. Saves the result to disk and returns it.

    Args:
        start_year: First season to include.
        end_year: Last season to include, inclusive.
        save_path: Destination path for the processed parquet file.

    Returns:
        Master DataFrame ready for modeling.
    """
    raise NotImplementedError


def load_master_dataset(path: str) -> pd.DataFrame:
    """Load the pre-built master dataset from disk.

    Args:
        path: Path to parquet file produced by build_master_dataset.

    Returns:
        Master modeling DataFrame.
    """
    raise NotImplementedError
