"""
injury_loader.py

Handles ingestion of pitcher injury data from the MLB Stats API.
Builds a structured IL database with one row per IL stint, including
normalized injury category, placement date, activation date, and days lost.

Data source:
    MLB Stats API  —  https://statsapi.mlb.com/api/v1/transactions
    No authentication required; public endpoint.
"""

from __future__ import annotations

import calendar
import logging
import re
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

RAW_INJURIES_DIR = Path("data/raw/injuries")

# ── IL transaction classification ───────────────────────────────────────────
# The MLB Stats API returns all IL transactions as typeCode='SC' (Status Change).
# The actual event type lives in the description text, so we derive it from there.
_RE_PLACEMENT  = re.compile(r"\bplaced\b.{0,60}\binjured list\b", re.I)
_RE_ACTIVATION = re.compile(r"\bactivated\b.{0,60}\binjured list\b", re.I)
_RE_TRANSFER   = re.compile(r"\btransferred\b.{0,60}\binjured list\b", re.I)
_RE_ANY_IL     = re.compile(r"\binjured list\b", re.I)

_PLACEMENT_CODES  = frozenset({"IL"})
_ACTIVATION_CODES = frozenset({"IA"})
_TRANSFER_CODES   = frozenset({"ITD"})

# ── Injury-type keyword patterns (applied in order — first match wins) ───────
# Order matters: more specific body parts precede broader ones.
_INJURY_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Typos "elobw" and "eblow" appear in real MLB API data.
    ("elbow",       re.compile(r"elbow|elobw|eblow|ucl|ulnar collateral|flexor mass|"
                               r"medial epicondyle|olecranon|valgus", re.I)),
    ("shoulder",    re.compile(r"shoulder|rotator cuff|labrum|biceps tendon|"
                               r"biceps tendinit|biceps inflam|biceps strain|"
                               r"glenohumeral|supraspinatus|infraspinatus|"
                               r"ac joint|acromioclavicular|teres major|"
                               r"\bdeltoid\b|triceps", re.I)),
    # "flexor strain" / "flexor muscle" without a qualifier maps to forearm.
    ("forearm",     re.compile(r"forearm|pronator|flexor.pronator|"
                               r"flexor (muscle |tendon )?(strain|tear|inflammation|sprain)", re.I)),
    ("back",        re.compile(r"\bback\b|lumbar|thoracic|\bspine\b|"
                               r"vertebr|disc herniation|stress reaction|"
                               r"\blat\b|latissimus|rhomboid|rib.cage|"
                               r"pectoral|\bpec\b", re.I)),
    ("oblique",     re.compile(r"oblique|side strain", re.I)),
    ("hamstring",   re.compile(r"hamstring", re.I)),
    ("hip",         re.compile(r"\bhip\b|groin|adductor|hip flexor|"
                               r"femoral|si joint|sacroiliac", re.I)),
    ("knee",        re.compile(r"\bknee\b|patellar|meniscus|\bacl\b|\bmcl\b|"
                               r"pcl\b", re.I)),
    ("finger_hand", re.compile(r"finger|blister|thumb|\bhand\b|wrist|"
                               r"carpal|nail|metacarpal|knuckle", re.I)),
    ("calf_ankle",  re.compile(r"\bcalf\b|achilles|\bankle\b|plantar|"
                               r"\bshin\b|\bfoot\b|\btoe\b|metatarsal", re.I)),
    ("neck",        re.compile(r"\bneck\b|cervical", re.I)),
    ("illness",     re.compile(r"illness|sick|virus|\bflu\b|infection|"
                               r"covid|personal|non.baseball|bereavement|"
                               r"paternity|concussion|mental health|"
                               r"undisclosed|anxiety|cancer|facial fracture", re.I)),
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _month_end_day(year: int, month: int) -> int:
    return calendar.monthrange(year, month)[1]


def _get(url: str, params: dict, retries: int = 3) -> dict:
    """GET with simple retry on transient errors."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning("Request failed (%s), retrying in %ss…", exc, wait)
            time.sleep(wait)
    return {}


# ── Public API ───────────────────────────────────────────────────────────────

def fetch_il_transactions(
    start_year: int,
    end_year: int,
    delay_seconds: float = 0.5,
) -> pd.DataFrame:
    """Fetch all IL-related transactions from the MLB Stats API.

    Queries one month at a time to keep response sizes manageable. Returns
    the raw transaction log — both placements (IL) and activations (IA) —
    before any pairing or label attachment.

    Args:
        start_year: First season to include (e.g. 2015).
        end_year: Last season to include, inclusive.
        delay_seconds: Sleep between monthly requests.

    Returns:
        DataFrame with columns:
            player_id, player_name, team, transaction_type, type_desc,
            transaction_date, description, season.
    """
    BASE = "https://statsapi.mlb.com/api/v1/transactions"
    records: list[dict] = []

    for year in range(start_year, end_year + 1):
        logger.info("Fetching IL transactions for %s…", year)
        year_count = 0

        for month in range(1, 13):
            start = f"{year}-{month:02d}-01"
            end   = f"{year}-{month:02d}-{_month_end_day(year, month)}"

            data = _get(BASE, {"startDate": start, "endDate": end, "sportId": 1})
            transactions = data.get("transactions", [])

            for t in transactions:
                desc = t.get("description") or ""

                # The API returns all IL events as typeCode='SC' (Status Change).
                # Skip anything that doesn't mention the injured list.
                if not _RE_ANY_IL.search(desc):
                    continue

                # Classify placement / activation / transfer from description text.
                if _RE_PLACEMENT.search(desc):
                    tx_type = "IL"
                elif _RE_ACTIVATION.search(desc):
                    tx_type = "IA"
                elif _RE_TRANSFER.search(desc):
                    tx_type = "ITD"
                else:
                    tx_type = "IL_OTHER"

                person = t.get("person") or {}
                team   = (t.get("toTeam") or t.get("fromTeam") or {}).get("name")

                records.append({
                    "player_id":        person.get("id"),
                    "player_name":      person.get("fullName"),
                    "team":             team,
                    "transaction_type": tx_type,
                    "type_desc":        t.get("typeDesc", ""),
                    "transaction_date": t.get("date") or t.get("effectiveDate"),
                    "description":      desc,
                    "season":           year,
                })
                year_count += 1

            time.sleep(delay_seconds)

        logger.info("  %s IL transactions found in %s", year_count, year)

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
    df = df.dropna(subset=["player_id", "transaction_date"])
    df["player_id"] = df["player_id"].astype(int)
    df = df.sort_values(["player_id", "transaction_date"]).reset_index(drop=True)
    return df


def parse_injury_type(description: str) -> str:
    """Map a raw transaction description to a normalized injury category.

    Applies keyword patterns in priority order — elbow before shoulder before
    forearm, etc. — so more specific categories are preferred.

    Args:
        description: Raw description string from the MLB Stats API transaction.

    Returns:
        One of: elbow, shoulder, forearm, back, oblique, hamstring, hip,
        knee, finger_hand, calf_ankle, neck, illness, or 'other'.
    """
    if not isinstance(description, str) or not description.strip():
        return "other"

    for category, pattern in _INJURY_PATTERNS:
        if pattern.search(description):
            return category

    return "other"


def compute_days_lost(il_df: pd.DataFrame) -> pd.DataFrame:
    """Pair each IL placement with its corresponding activation.

    For each placement (typeCode == 'IL'), finds the earliest activation
    (typeCode == 'IA') for the same player on or after the placement date.
    IL transfers (ITD) are ignored for pairing — they extend an existing
    stint rather than starting a new one, so the days_lost naturally spans
    the full duration from the original placement to the final activation.

    Edge cases handled:
    - Season-ending injuries with no activation: activation_date = NaT,
      season_ending = True.
    - Overlapping stints (re-placement before activation): only the first
      available activation is consumed.

    Args:
        il_df: DataFrame from fetch_il_transactions.

    Returns:
        DataFrame of IL placements only, augmented with:
            activation_date, days_lost (int), season_ending (bool).
    """
    placements  = il_df[il_df["transaction_type"].isin(_PLACEMENT_CODES)].copy()
    activations = il_df[il_df["transaction_type"].isin(_ACTIVATION_CODES)].copy()

    result_rows: list[dict] = []
    # Track which activation rows have already been matched so we don't
    # double-count them when a player has multiple stints in one season.
    used_activation_indices: set[int] = set()

    for _, p in placements.iterrows():
        pid            = p["player_id"]
        placement_date = p["transaction_date"]

        candidates = activations[
            (activations["player_id"] == pid)
            & (activations["transaction_date"] >= placement_date)
            & (~activations.index.isin(used_activation_indices))
        ].sort_values("transaction_date")

        row = p.to_dict()

        if len(candidates) > 0:
            act = candidates.iloc[0]
            used_activation_indices.add(act.name)
            row["activation_date"] = act["transaction_date"]
            row["days_lost"]       = int((act["transaction_date"] - placement_date).days)
            row["season_ending"]   = False
        else:
            row["activation_date"] = pd.NaT
            row["days_lost"]       = None
            row["season_ending"]   = True

        result_rows.append(row)

    out = pd.DataFrame(result_rows)
    if len(out) == 0:
        return out

    # Attach normalized injury type from the description field.
    out["injury_type"] = out["description"].apply(parse_injury_type)

    col_order = [
        "player_id", "player_name", "team", "season",
        "transaction_date", "activation_date",
        "days_lost", "season_ending",
        "injury_type", "type_desc", "description",
    ]
    out = out[[c for c in col_order if c in out.columns]]
    out = out.sort_values(["player_id", "transaction_date"]).reset_index(drop=True)
    return out


def build_injury_database(
    start_year: int,
    end_year: int,
    pitcher_ids: set[int] | None = None,
    save_path: str | None = None,
) -> pd.DataFrame:
    """Build the master pitcher injury database.

    Runs the full pipeline: fetch → parse → pair. Optionally filters to a
    known set of pitcher MLBAM IDs so the output contains only pitchers.

    Args:
        start_year: First season to include.
        end_year: Last season to include, inclusive.
        pitcher_ids: If provided, restricts output to these MLBAM IDs.
        save_path: If provided, saves the result as parquet at this path.

    Returns:
        One row per IL stint with placement/activation dates, days lost,
        injury type, and season_ending flag.
    """
    il_raw = fetch_il_transactions(start_year, end_year)

    if pitcher_ids is not None:
        il_raw = il_raw[il_raw["player_id"].isin(pitcher_ids)]
        logger.info("Filtered to %s pitcher IL records", len(il_raw))

    db = compute_days_lost(il_raw)

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        db.to_parquet(path, index=False)
        logger.info("Saved injury database → %s (%s stints)", path, len(db))

    return db


def load_injury_database(path: str) -> pd.DataFrame:
    """Load a previously built injury database from parquet.

    Args:
        path: Path to the parquet file produced by build_injury_database.

    Returns:
        Injury DataFrame.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No injury database at {p}. Run build_injury_database first.")
    return pd.read_parquet(p)
