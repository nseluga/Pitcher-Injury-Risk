"""
injury_loader.py

Handles ingestion of pitcher injury data from external sources including
the MLB IL (Injured List) transaction records, Baseball Reference, and
any supplementary injury databases integrated in the future.

Data sources:
- MLB Transactions (pybaseball or direct scraping)
- Baseball Reference injury history
- Spotrac (contract/injury history, future integration)
"""

from __future__ import annotations

import pandas as pd


def fetch_il_transactions(
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """Fetch pitcher IL placement and activation transactions from MLB.

    Uses pybaseball or MLB Stats API to pull all IL-related transactions
    for a given date range. Filters to pitchers only.

    Args:
        start_year: First season to include (e.g. 2015).
        end_year: Last season to include, inclusive.

    Returns:
        DataFrame with columns: player_id, player_name, team, transaction_type,
        transaction_date, injury_type, notes.
    """
    raise NotImplementedError


def parse_injury_type(notes: str) -> str:
    """Extract a normalized injury category from raw transaction notes.

    Maps free-text notes (e.g. 'right elbow inflammation') to a canonical
    injury category (e.g. 'elbow'). Categories are defined in the data
    dictionary.

    Args:
        notes: Raw transaction note string from the MLB transactions feed.

    Returns:
        Normalized injury category string.
    """
    raise NotImplementedError


def compute_days_lost(il_df: pd.DataFrame) -> pd.DataFrame:
    """Pair each IL placement with its corresponding activation to derive days lost.

    Matches each IL-placement row to the next IL-activation row for the same
    player and computes the number of days between them. Handles edge cases
    such as season-ending injuries with no matching activation.

    Args:
        il_df: DataFrame returned by fetch_il_transactions.

    Returns:
        il_df augmented with columns: activation_date, days_lost, season_ending.
    """
    raise NotImplementedError


def build_injury_database(
    start_year: int,
    end_year: int,
    save_path: str | None = None,
) -> pd.DataFrame:
    """Build the master injury database for all pitchers across a date range.

    Orchestrates fetch_il_transactions → parse_injury_type → compute_days_lost
    and returns a clean, analysis-ready DataFrame. Optionally serializes the
    result to disk.

    Args:
        start_year: First season to include.
        end_year: Last season to include, inclusive.
        save_path: If provided, saves the resulting DataFrame as a parquet file
            at this path.

    Returns:
        Master injury DataFrame with one row per IL stint.
    """
    raise NotImplementedError


def load_injury_database(path: str) -> pd.DataFrame:
    """Load a previously built injury database from disk.

    Args:
        path: Path to the parquet file produced by build_injury_database.

    Returns:
        Injury DataFrame.
    """
    raise NotImplementedError
