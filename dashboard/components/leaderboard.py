import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.data_loader import ARCHETYPES, load_pitcher_season


def render() -> None:
    ps = load_pitcher_season()
    seasons = sorted(ps["season"].unique(), reverse=True)
    present = set(ps["archetype"].dropna())
    archetypes = ["All"] + [a for a in ARCHETYPES if a in present]

    col1, col2 = st.columns([1, 2])
    with col1:
        season = st.selectbox("Season", seasons, index=0)
    with col2:
        arch_choice = st.selectbox("Archetype", archetypes, index=0)

    sub = ps[ps["season"] == season].copy()
    if arch_choice != "All":
        sub = sub[sub["archetype"] == arch_choice]

    if sub.empty:
        st.warning("No pitchers match that season and archetype.")
        return

    sub = sub.sort_values("ir_plus", ascending=False).reset_index(drop=True)
    # A narrow archetype can hold fewer than 20 pitchers in a season, and then
    # head(10) and tail(10) overlap — the same pitcher shows up as both highest
    # and lowest risk. Cap each panel at half the group.
    n = min(10, len(sub) // 2) or len(sub)
    # plotly draws a horizontal bar chart's first row at the *bottom*, so each
    # slice is reversed to put the leading pitcher at the top: highest risk
    # descends most-to-least, lowest risk ascends least-to-most.
    top = sub.head(n).sort_values("ir_plus")
    bot = sub.tail(n).sort_values("ir_plus", ascending=False)

    suffix = f" ({arch_choice})" if arch_choice != "All" else ""
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Top {n} Highest Risk — {season}{suffix}",
            f"Top {n} Lowest Risk — {season}{suffix}",
        ),
        horizontal_spacing=0.14,
    )
    fig.add_trace(
        go.Bar(
            x=top["ir_plus"],
            y=top["player_name"],
            orientation="h",
            marker_color="crimson",
            text=top["ir_plus"].map("{:.1f}".format),
            textposition="outside",
            name="Highest Risk",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=bot["ir_plus"],
            y=bot["player_name"],
            orientation="h",
            marker_color="steelblue",
            text=bot["ir_plus"].map("{:.1f}".format),
            textposition="outside",
            name="Lowest Risk",
        ),
        row=1,
        col=2,
    )
    fig.add_vline(x=100, line_dash="dash", line_color="gray", row=1, col=1)
    fig.add_vline(x=100, line_dash="dash", line_color="gray", row=1, col=2)
    fig.update_layout(
        title_text=f"Injury Risk+ Leaderboard — {season}",
        height=720,
        showlegend=False,
        font=dict(size=11),
        margin=dict(l=10, r=10),
    )
    fig.update_xaxes(title_text="Injury Risk+")
    # Pin the axis type. Player names are the categories, and plotly infers a
    # linear axis when enough of them parse as numbers — which silently dropped
    # most of the 2022/2023 bars back when unnamed pitchers fell back to their id.
    fig.update_yaxes(type="category")
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"{len(sub):,} pitchers shown · mean IR+ = {sub['ir_plus'].mean():.1f} · "
        f"starters = {(sub['role']=='starter').sum()} · "
        f"relievers = {(sub['role']=='reliever').sum()}"
    )

    with st.expander("Full table"):
        show = sub[["player_name", "archetype", "role", "ir_plus", "percentile",
                     "injury_prob_30d", "expected_days_lost", "hazard_rate"]].copy()
        show.columns = ["Name", "Archetype", "Role", "IR+", "Percentile",
                        "Inj Prob 30d", "Exp Days Lost", "Hazard Rate"]
        show = show.reset_index(drop=True)
        show.index += 1
        st.dataframe(show, use_container_width=True)
