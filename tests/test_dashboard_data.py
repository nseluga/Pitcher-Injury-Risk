"""Guards for the dashboard's pitcher-season frame.

Run directly (`python tests/test_dashboard_data.py`) or under pytest.

These cover three bugs that shipped together:
  * ~23% of pitchers displayed as a bare numeric id instead of a name;
  * the 2022 and 2023 "Lowest Risk" panels rendered almost no bars, because
    those numeric ids flipped plotly's y-axis from category to linear;
  * the archetype filter offered only starter/reliever.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dashboard.data_loader import ARCHETYPES, load_pitcher_season  # noqa: E402

NUMERIC = re.compile(r"^\d+$")


def _ps():
    return load_pitcher_season()


def test_every_pitcher_has_a_real_name():
    ps = _ps()
    bad = ps.loc[ps["player_name"].astype(str).str.match(NUMERIC), "pitcher_id"].unique()
    assert len(bad) == 0, f"{len(bad)} pitchers still fall back to their id: {bad[:5]}"


def test_names_identify_exactly_one_pitcher():
    """The profile and comparison panels select rows by name, so a name shared
    by two pitchers would silently merge their careers into one series."""
    ps = _ps()
    per_name = ps.groupby("player_name")["pitcher_id"].nunique()
    clashes = per_name[per_name > 1]
    assert clashes.empty, f"names mapping to >1 pitcher: {clashes.to_dict()}"


def test_leaderboard_y_axis_stays_categorical():
    """plotly.js infers a linear axis unless strings outnumber numeric-looking
    values 2:1, which is what broke 2022 and 2023. Assert the margin directly
    on the exact slices the leaderboard plots."""
    ps = _ps()
    for season in sorted(ps["season"].unique()):
        sub = ps[ps["season"] == season].sort_values("ir_plus", ascending=False)
        n = min(10, len(sub) // 2) or len(sub)
        for label, names in (
            ("highest", sub.head(n)["player_name"]),
            ("lowest", sub.tail(n)["player_name"]),
        ):
            numeric = int(names.astype(str).str.match(NUMERIC).sum())
            categorical = len(names) - numeric
            assert categorical > 2 * numeric, (
                f"{season} {label}: {numeric} numeric-looking of {len(names)} labels — "
                "plotly would render this axis as linear and drop the named bars"
            )


def test_all_archetypes_present_and_assigned():
    ps = _ps()
    assert set(ps["archetype"]) <= set(ARCHETYPES), "unexpected archetype label"
    missing = set(ARCHETYPES) - set(ps["archetype"])
    assert not missing, f"archetypes never assigned: {missing}"
    assert ps["archetype"].notna().all()
    # the coarse role is still available — the leaderboard caption counts on it
    assert set(ps["role"].dropna()) == {"starter", "reliever"}


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
    print("\nall checks passed")
