import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.data_loader import load_pitcher_season


def render() -> None:
    ps = load_pitcher_season()
    all_names = sorted(ps["player_name"].dropna().unique().tolist())

    st.markdown("Select 2–5 pitchers to overlay their Injury Risk+ trends.")

    selected_names = st.multiselect(
        "Pitchers",
        options=all_names,
        default=all_names[:3] if len(all_names) >= 3 else all_names,
        max_selections=5,
    )

    if len(selected_names) < 2:
        st.info("Select at least 2 pitchers to compare.")
        return

    colors = px.colors.qualitative.Plotly
    fig = go.Figure()

    for i, name in enumerate(selected_names):
        rows = ps[ps["player_name"] == name].sort_values("season")
        if rows.empty:
            continue
        color = colors[i % len(colors)]
        fig.add_trace(
            go.Scatter(
                x=rows["season"],
                y=rows["ir_plus"],
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=8),
                name=name,
                hovertemplate=f"<b>{name}</b><br>Season %{{x}}<br>IR+ %{{y:.1f}}<extra></extra>",
            )
        )

    fig.add_hline(
        y=100,
        line_dash="dash",
        line_color="gray",
        annotation_text="League avg (100)",
        annotation_position="bottom right",
    )
    fig.update_layout(
        title="Injury Risk+ Career Trends — Comparison",
        xaxis_title="Season",
        yaxis_title="Injury Risk+",
        height=520,
        legend=dict(x=0.01, y=0.99),
        margin=dict(t=50),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary stats table
    rows_list = []
    for name in selected_names:
        r = ps[ps["player_name"] == name]
        if r.empty:
            continue
        rows_list.append({
            "Name": name,
            "Archetype": r["archetype"].mode().iloc[0],
            "Seasons": r["season"].nunique(),
            "Avg IR+": round(r["ir_plus"].mean(), 1),
            "Peak IR+": round(r["ir_plus"].max(), 1),
            "Latest IR+": round(r.sort_values("season")["ir_plus"].iloc[-1], 1),
            "Latest Season": int(r["season"].max()),
        })
    if rows_list:
        import pandas as pd
        st.dataframe(pd.DataFrame(rows_list).set_index("Name"), use_container_width=True)
