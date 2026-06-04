"""
Statcast data ingestion via PyBaseball and Baseball Savant.

Handles pitch-level data pulls for individual pitchers, date ranges, and full
season sweeps. Outputs are written to data/raw/statcast/ as Parquet files.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

RAW_STATCAST_DIR = Path("data/raw/statcast")


def fetch_statcast_by_pitcher(
    pitcher_id: int,
    start_date: str,
    end_date: str,
    save: bool = True,
) -> pd.DataFrame:
    """
    Pull pitch-level Statcast data for a single pitcher over a date range.

    Uses pybaseball.statcast_pitcher under the hood. Applies rate-limiting
    to avoid hammering Baseball Savant's servers.

    Parameters
    ----------
    pitcher_id : int
        MLB MLBAM player ID (e.g., 592789 for Gerrit Cole).
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    save : bool
        If True, writes result to data/raw/statcast/<pitcher_id>.parquet.

    Returns
    -------
    pd.DataFrame
        Pitch-level Statcast data with columns including pitch_type, release_speed,
        pfx_x, pfx_z, spin_rate, plate_x, plate_z, events, description, etc.
    """
    raise NotImplementedError


def fetch_statcast_season(
    season: int,
    pitcher_ids: Optional[list[int]] = None,
    save: bool = True,
) -> pd.DataFrame:
    """
    Pull Statcast data for an entire season, optionally filtered to a pitcher list.

    Pulls data in monthly chunks to avoid timeouts. Filters to pitchers only.
    If pitcher_ids is None, pulls all pitchers.

    Parameters
    ----------
    season : int
        MLB season year (e.g., 2023).
    pitcher_ids : list[int] or None
        Restrict pull to these MLBAM IDs. If None, pulls all pitchers.
    save : bool
        If True, writes to data/raw/statcast/season_{season}.parquet.

    Returns
    -------
    pd.DataFrame
        Full season pitch-level data for all (or specified) pitchers.
    """
    raise NotImplementedError


def fetch_statcast_date_range(
    start_date: str,
    end_date: str,
    save: bool = True,
) -> pd.DataFrame:
    """
    Pull all pitcher Statcast data for an arbitrary date range.

    Splits requests into weekly chunks to stay within API limits.

    Parameters
    ----------
    start_date : str
        Start of window in 'YYYY-MM-DD' format.
    end_date : str
        End of window in 'YYYY-MM-DD' format.
    save : bool
        If True, appends results to data/raw/statcast/.

    Returns
    -------
    pd.DataFrame
        Pitch-level Statcast data for the requested window.
    """
    raise NotImplementedError


def load_statcast_from_disk(
    season: Optional[int] = None,
    pitcher_id: Optional[int] = None,
) -> pd.DataFrame:
    """
    Load previously saved Statcast Parquet files from data/raw/statcast/.

    Parameters
    ----------
    season : int or None
        Load a full-season file if provided.
    pitcher_id : int or None
        Load a pitcher-specific file if provided.

    Returns
    -------
    pd.DataFrame
        Combined Statcast data from matching files.
    """
    raise NotImplementedError


def get_pitcher_ids_from_savant(
    season: int,
    min_pitches: int = 100,
) -> list[int]:
    """
    Retrieve MLBAM IDs for all pitchers who threw a minimum pitch count in a season.

    Scrapes the Baseball Savant leaderboard for the given season and filters
    to pitchers meeting the minimum pitch threshold.

    Parameters
    ----------
    season : int
        Season year.
    min_pitches : int
        Minimum pitches thrown to be included.

    Returns
    -------
    list[int]
        List of MLBAM pitcher IDs.
    """
    raise NotImplementedError
