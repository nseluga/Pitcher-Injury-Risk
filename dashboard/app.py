"""Pitcher Injury Risk+ — Unified Analysis Dashboard."""

import sys
from pathlib import Path

# Ensure project root is on the path so imports work when launched from any cwd.
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

st.set_page_config(
    page_title="Pitcher Injury Risk+ Dashboard",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="expanded",
)

PANELS = {
    "Season Leaderboard": "leaderboard",
    "Pitcher Profile": "pitcher_profile",
    "Multi-Pitcher Comparison": "trend_comparison",
    "Archetype Analysis": "archetype_panel",
}

st.sidebar.title("⚾ Injury Risk+")
st.sidebar.markdown("Pitcher injury risk analysis · 2015–2024")
st.sidebar.markdown("---")

panel_label = st.sidebar.radio("Panel", list(PANELS.keys()))
panel_key = PANELS[panel_label]

st.sidebar.markdown("---")
st.sidebar.caption(
    "IR+ = 100 is league average. Higher values indicate elevated injury risk. "
    "Components: injury probability (50%), expected days lost (30%), hazard rate (20%)."
)

st.title(panel_label)

if panel_key == "leaderboard":
    from dashboard.components import leaderboard
    leaderboard.render()

elif panel_key == "pitcher_profile":
    from dashboard.components import pitcher_profile
    pitcher_profile.render()

elif panel_key == "trend_comparison":
    from dashboard.components import trend_comparison
    trend_comparison.render()

elif panel_key == "archetype_panel":
    from dashboard.components import archetype_panel
    archetype_panel.render()
