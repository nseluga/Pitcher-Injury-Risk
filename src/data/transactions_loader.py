"""
transactions_loader.py

Fetches and parses MLB transaction data beyond IL placements — including
trades, DFA, option assignments, and free-agent signings. This broader
transaction context can be useful for understanding workload changes,
role transitions (starter ↔ reliever), and team-level usage patterns.

Data sources:
- Baseball Reference transactions
- MLB Stats API transactions endpoint
- pybaseball transactions utilities
"""

from __future__ import annotations

import pandas as pd


def fetch_all_transactions(
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Fetch all MLB transactions for a given range of seasons.

    Pulls the full transaction log (trades, DFA, options, signings, IL) so
    downstream code can filter to the transaction types it needs.

    Args:
        start_year: First season to include.
        end_year: Last season to include, inclusive.

    Returns:
        DataFrame with columns: player_id, player_name, team_from, team_to,
        transaction_type, transaction_date, notes.
    """
    raise NotImplementedError


def filter_pitcher_transactions(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Restrict a transactions DataFrame to pitchers only.

    Joins against the player metadata to identify pitchers. Retains all
    transaction types so callers can further filter as needed.

    Args:
        transactions_df: Full transactions DataFrame from fetch_all_transactions.

    Returns:
        Subset of transactions_df limited to pitchers.
    """
    raise NotImplementedError


def identify_role_changes(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Detect starter-to-reliever and reliever-to-starter transitions.

    Analyzes consecutive transactions and game-log data to flag when a pitcher
    meaningfully changed roles within or across seasons. Role labels are
    determined by usage patterns rather than roster designations.

    Args:
        transactions_df: Pitcher transactions DataFrame from
            filter_pitcher_transactions.

    Returns:
        DataFrame with columns: player_id, season, role_before, role_after,
        transition_date.
    """
    raise NotImplementedError


def flag_rehab_assignments(transactions_df: pd.DataFrame) -> pd.DataFrame:
    """Mark minor-league rehab assignment transactions.

    Rehab assignments indicate a pitcher returning from injury. These can be
    paired with IL data to create a more complete picture of injury recovery
    timelines.

    Args:
        transactions_df: Pitcher transactions DataFrame.

    Returns:
        transactions_df with an additional boolean column: is_rehab.
    """
    raise NotImplementedError


def save_transactions(df: pd.DataFrame, path: str) -> None:
    """Persist a transactions DataFrame to parquet.

    Args:
        df: Transactions DataFrame to save.
        path: Destination file path (should end in .parquet).
    """
    raise NotImplementedError


def load_transactions(path: str) -> pd.DataFrame:
    """Load a saved transactions DataFrame from parquet.

    Args:
        path: Path to parquet file produced by save_transactions.

    Returns:
        Transactions DataFrame.
    """
    raise NotImplementedError
