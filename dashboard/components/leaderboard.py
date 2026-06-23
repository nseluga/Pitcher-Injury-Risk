import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.data_loader import load_pitcher_season


def render() -> None:
    ps = load_pitcher_season()
    seasons = sorted(ps["season"].unique(), reverse=True)
    archetypes = ["All"] + sorted(ps["archetype"].dropna().unique().tolist())

    col1, col2 = st.columns([1, 2])
    with col1:
        season = st.selectbox("Season", seasons, index=0)
    with col2:
        arch_choice = st.selectbox("Archetype", archetypes, index=0)

    sub = ps[ps["season"] == season].copy()
    if arch_choice != "All":
        sub = sub[sub["archetype"] == arch_choice]

    sub = sub.sort_values("ir_plus", ascending=False).reset_index(drop=True)
    top25 = sub.head(25)
    bot25 = sub.tail(25).sort_values("ir_plus")

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Top 25 Highest Risk — {season}"
            + (f" ({arch_choice})" if arch_choice != "All" else ""),
            f"Top 25 Lowest Risk — {season}"
            + (f" ({arch_choice})" if arch_choice != "All" else ""),
        ),
        horizontal_spacing=0.14,
    )
    fig.add_trace(
        go.Bar(
            x=top25["ir_plus"],
            y=top25["player_name"],
            orientation="h",
            marker_color="crimson",
            text=top25["ir_plus"].map("{:.1f}".format),
            textposition="outside",
            name="Highest Risk",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Bar(
            x=bot25["ir_plus"],
            y=bot25["player_name"],
            orientation="h",
            marker_color="steelblue",
            text=bot25["ir_plus"].map("{:.1f}".format),
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
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        f"{len(sub):,} pitchers shown · mean IR+ = {sub['ir_plus'].mean():.1f} · "
        f"starters = {(sub['archetype']=='starter').sum()} · "
        f"relievers = {(sub['archetype']=='reliever').sum()}"
    )

    with st.expander("Full table"):
        show = sub[["player_name", "archetype", "ir_plus", "percentile",
                     "injury_prob_30d", "expected_days_lost", "hazard_rate"]].copy()
        show.columns = ["Name", "Archetype", "IR+", "Percentile",
                        "Inj Prob 30d", "Exp Days Lost", "Hazard Rate"]
        show = show.reset_index(drop=True)
        show.index += 1
        st.dataframe(show, use_container_width=True)
