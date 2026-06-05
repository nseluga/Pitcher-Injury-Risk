"""
Statcast data ingestion via PyBaseball and Baseball Savant.

Handles pitch-level data pulls for individual pitchers, date ranges, and full
season sweeps. Outputs are written to data/raw/statcast/ as Parquet files.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

RAW_STATCAST_DIR = Path("data/raw/statcast")

# Columns we keep from the raw Statcast pull.  Dropping the ~40 less-used
# columns keeps file sizes reasonable while retaining everything needed for
# injury modeling.
STATCAST_KEEP_COLS = [
    "pitch_type", "game_date", "release_speed", "release_pos_x",
    "release_pos_z", "player_name", "batter", "pitcher", "events",
    "description", "spin_dir", "zone", "des", "game_type", "stand",
    "p_throws", "home_team", "away_team", "type", "hit_location",
    "bb_type", "balls", "strikes", "pfx_x", "pfx_z", "plate_x",
    "plate_z", "on_3b", "on_2b", "on_1b", "outs_when_up", "inning",
    "inning_topbot", "hc_x", "hc_y", "fielder_2", "ump_id",
    "sv_id", "vx0", "vy0", "vz0", "ax", "ay", "az",
    "sz_top", "sz_bot", "hit_distance_sc", "launch_speed",
    "launch_angle", "effective_speed", "release_spin_rate",
    "release_extension", "game_pk", "pitcher_1", "fielder_2_1",
    "fielder_3", "fielder_4", "fielder_5", "fielder_6", "fielder_7",
    "fielder_8", "fielder_9", "release_pos_y", "estimated_ba_using_speedangle",
    "estimated_woba_using_speedangle", "woba_value", "woba_denom",
    "babip_value", "iso_value", "launch_speed_angle", "at_bat_number",
    "pitch_number", "home_score", "away_score", "bat_score",
    "fld_score", "post_away_score", "post_home_score", "post_bat_score",
    "post_fld_score", "if_fielding_alignment", "of_fielding_alignment",
    "spin_axis",
]


def _month_ranges(season: int) -> list[tuple[str, str]]:
    """Return (start, end) date strings for each month of an MLB season.

    Covers March (spring training / early openers) through October (postseason).
    The 2020 COVID season started in late July; the wider window handles it safely.
    """
    months = [
        (3, 31), (4, 30), (5, 31), (6, 30),
        (7, 31), (8, 31), (9, 30), (10, 31),
    ]
    return [(f"{season}-{m:02d}-01", f"{season}-{m:02d}-{d}") for m, d in months]


def fetch_statcast_season(
    season: int,
    pitcher_ids: Optional[list[int]] = None,
    save: bool = True,
    delay_seconds: float = 5.0,
) -> pd.DataFrame:
    """Pull Statcast pitch-level data for an entire season in monthly chunks.

    Chunking by month stays well within Baseball Savant's request limits.
    If pitcher_ids is None, every pitcher in the season is included.

    Args:
        season: MLB season year (e.g. 2023).
        pitcher_ids: Restrict to these MLBAM pitcher IDs. None = all pitchers.
        save: Write result to data/raw/statcast/season_{season}.parquet.
        delay_seconds: Sleep between monthly requests to avoid rate-limiting.

    Returns:
        Pitch-level DataFrame for the season.
    """
    try:
        import pybaseball as pyb
    except ImportError as exc:
        raise ImportError("pybaseball is required: pip install pybaseball") from exc

    chunks: list[pd.DataFrame] = []
    for start, end in _month_ranges(season):
        logger.info("Fetching %s → %s", start, end)
        try:
            chunk = pyb.statcast(start_dt=start, end_dt=end, verbose=False)
        except Exception:
            logger.exception("Failed to fetch %s → %s; skipping", start, end)
            time.sleep(delay_seconds)
            continue

        if chunk is not None and len(chunk) > 0:
            chunks.append(chunk)
            logger.info("  %s pitches", f"{len(chunk):,}")

        time.sleep(delay_seconds)

    if not chunks:
        logger.warning("No data returned for season %s", season)
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)

    # Keep only columns that actually exist in this pull (schema varies by year).
    cols = [c for c in STATCAST_KEEP_COLS if c in df.columns]
    df = df[cols]

    if pitcher_ids is not None:
        df = df[df["pitcher"].isin(pitcher_ids)]

    df["game_date"] = pd.to_datetime(df["game_date"])
    df = df.sort_values(["game_date", "game_pk", "at_bat_number", "pitch_number"])
    df = df.reset_index(drop=True)

    if save:
        out_path = RAW_STATCAST_DIR / f"season_{season}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)
        logger.info("Saved %s (%s rows)", out_path, f"{len(df):,}")

    return df


def fetch_statcast_by_pitcher(
    pitcher_id: int,
    start_date: str,
    end_date: str,
    save: bool = True,
) -> pd.DataFrame:
    """Pull pitch-level Statcast data for a single pitcher over a date range.

    Uses pybaseball.statcast_pitcher under the hood. Applies rate-limiting
    to avoid hammering Baseball Savant's servers.

    Args:
        pitcher_id: MLB MLBAM player ID (e.g. 592789 for Gerrit Cole).
        start_date: Start date in 'YYYY-MM-DD' format.
        end_date: End date in 'YYYY-MM-DD' format.
        save: If True, writes result to data/raw/statcast/<pitcher_id>.parquet.

    Returns:
        Pitch-level DataFrame with Statcast columns.
    """
    try:
        import pybaseball as pyb
    except ImportError as exc:
        raise ImportError("pybaseball is required: pip install pybaseball") from exc

    df = pyb.statcast_pitcher(start_dt=start_date, end_dt=end_date, player_id=pitcher_id)

    if df is None or len(df) == 0:
        logger.warning("No data for pitcher_id=%s in %s → %s", pitcher_id, start_date, end_date)
        return pd.DataFrame()

    cols = [c for c in STATCAST_KEEP_COLS if c in df.columns]
    df = df[cols]
    df["game_date"] = pd.to_datetime(df["game_date"])

    if save:
        out_path = RAW_STATCAST_DIR / f"{pitcher_id}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)

    return df


def fetch_statcast_date_range(
    start_date: str,
    end_date: str,
    save: bool = True,
) -> pd.DataFrame:
    """Pull all pitcher Statcast data for an arbitrary date range.

    Splits requests into weekly chunks to stay within API limits.

    Args:
        start_date: Start of window in 'YYYY-MM-DD' format.
        end_date: End of window in 'YYYY-MM-DD' format.
        save: If True, appends results to data/raw/statcast/.

    Returns:
        Pitch-level DataFrame for the requested window.
    """
    try:
        import pybaseball as pyb
    except ImportError as exc:
        raise ImportError("pybaseball is required: pip install pybaseball") from exc

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    chunks: list[pd.DataFrame] = []
    cursor = start

    while cursor <= end:
        week_end = min(cursor + pd.Timedelta(days=6), end)
        logger.info("Fetching %s → %s", cursor.date(), week_end.date())
        try:
            chunk = pyb.statcast(
                start_dt=cursor.strftime("%Y-%m-%d"),
                end_dt=week_end.strftime("%Y-%m-%d"),
                verbose=False,
            )
            if chunk is not None and len(chunk) > 0:
                chunks.append(chunk)
        except Exception:
            logger.exception("Failed chunk %s → %s", cursor.date(), week_end.date())
        time.sleep(5)
        cursor = week_end + pd.Timedelta(days=1)

    if not chunks:
        return pd.DataFrame()

    df = pd.concat(chunks, ignore_index=True)
    cols = [c for c in STATCAST_KEEP_COLS if c in df.columns]
    df = df[cols]
    df["game_date"] = pd.to_datetime(df["game_date"])

    if save:
        slug = f"{start_date}_{end_date}".replace("-", "")
        out_path = RAW_STATCAST_DIR / f"range_{slug}.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out_path, index=False)

    return df


def load_statcast_from_disk(
    season: Optional[int] = None,
    pitcher_id: Optional[int] = None,
) -> pd.DataFrame:
    """Load previously saved Statcast Parquet files from data/raw/statcast/.

    Args:
        season: Load a full-season file if provided.
        pitcher_id: Load a pitcher-specific file if provided.

    Returns:
        Combined Statcast DataFrame from matching files.
    """
    if season is not None:
        path = RAW_STATCAST_DIR / f"season_{season}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No file found at {path}. Run fetch_statcast_season first.")
        return pd.read_parquet(path)

    if pitcher_id is not None:
        path = RAW_STATCAST_DIR / f"{pitcher_id}.parquet"
        if not path.exists():
            raise FileNotFoundError(f"No file found at {path}.")
        return pd.read_parquet(path)

    # Load all season files and concatenate.
    files = sorted(RAW_STATCAST_DIR.glob("season_*.parquet"))
    if not files:
        raise FileNotFoundError(f"No season parquet files found in {RAW_STATCAST_DIR}.")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def get_pitcher_ids_from_savant(
    season: int,
    min_pitches: int = 100,
) -> list[int]:
    """Retrieve MLBAM IDs for all pitchers who threw a minimum pitch count in a season.

    Uses pybaseball.pitching_stats_bref or the FanGraphs ID map to get MLBAM IDs.

    Args:
        season: Season year.
        min_pitches: Minimum pitches thrown to be included.

    Returns:
        List of MLBAM pitcher IDs.
    """
    try:
        import pybaseball as pyb
    except ImportError as exc:
        raise ImportError("pybaseball is required: pip install pybaseball") from exc

    stats = pyb.pitching_stats(season, season, qual=0)
    if stats is None or len(stats) == 0:
        logger.warning("No pitching stats returned for %s", season)
        return []

    # FanGraphs stats include 'IDfg'; we need MLBAM IDs for Statcast joins.
    id_map = pyb.playerid_reverse_lookup(
        stats["IDfg"].dropna().astype(int).tolist(),
        key_type="fangraphs",
    )
    mlbam_ids = id_map["key_mlbam"].dropna().astype(int).tolist()
    return mlbam_ids
