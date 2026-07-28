import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.data_loader import load_pitcher_season, load_workload_by_season


def _fuzzy_search(names: list[str], query: str) -> list[str]:
    q = query.strip().lower()
    if not q:
        return names
    starts = [n for n in names if n.lower().startswith(q)]
    contains = [n for n in names if q in n.lower() and n not in starts]
    return starts + contains


def render() -> None:
    ps = load_pitcher_season()
    wl = load_workload_by_season()

    all_names = sorted(ps["player_name"].dropna().unique().tolist())

    search = st.text_input("Search pitcher name", placeholder="e.g. Verlander")
    filtered = _fuzzy_search(all_names, search) if search else all_names

    if not filtered:
        st.warning("No pitchers match that search.")
        return

    selected_name = st.selectbox("Select pitcher", filtered)
    pitcher_rows = ps[ps["player_name"] == selected_name].sort_values("season")

    if pitcher_rows.empty:
        st.error("No data found.")
        return

    pitcher_id = int(pitcher_rows["pitcher_id"].iloc[0])
    arch = pitcher_rows["archetype"].mode().iloc[0]

    # --- IR+ trend vs archetype average ---
    arch_stats = (
        ps[ps["archetype"] == arch]
        .groupby("season")["ir_plus"]
        .agg(["mean", "std"])
        .reset_index()
    )
    arch_stats.columns = ["season", "arch_mean", "arch_std"]
    arch_stats["arch_lo"] = arch_stats["arch_mean"] - arch_stats["arch_std"]
    arch_stats["arch_hi"] = arch_stats["arch_mean"] + arch_stats["arch_std"]

    fig_trend = go.Figure()
    fig_trend.add_trace(
        go.Scatter(
            x=arch_stats["season"].tolist() + arch_stats["season"].tolist()[::-1],
            y=arch_stats["arch_hi"].tolist() + arch_stats["arch_lo"].tolist()[::-1],
            fill="toself",
            fillcolor="rgba(100,100,200,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name=f"{arch} ±1 SD",
        )
    )
    fig_trend.add_trace(
        go.Scatter(
            x=arch_stats["season"],
            y=arch_stats["arch_mean"],
            mode="lines",
            line=dict(color="rgba(100,100,200,0.5)", dash="dot"),
            name=f"{arch} avg",
        )
    )
    fig_trend.add_trace(
        go.Scatter(
            x=pitcher_rows["season"],
            y=pitcher_rows["ir_plus"],
            mode="lines+markers",
            line=dict(color="crimson", width=2.5),
            marker=dict(size=9),
            name=selected_name,
            text=pitcher_rows["ir_plus"].map("{:.1f}".format),
            textposition="top center",
            hovertemplate="Season %{x}<br>IR+ %{y:.1f}<extra></extra>",
        )
    )
    fig_trend.add_hline(y=100, line_dash="dash", line_color="gray",
                        annotation_text="League avg (100)")
    fig_trend.update_layout(
        title=f"{selected_name} — Injury Risk+ by Season",
        xaxis_title="Season",
        yaxis_title="Injury Risk+",
        height=400,
        legend=dict(x=0.01, y=0.99),
        margin=dict(t=50),
    )
    st.plotly_chart(fig_trend, use_container_width=True)

    # --- Season selector for component breakdown ---
    seasons_available = sorted(pitcher_rows["season"].tolist(), reverse=True)
    sel_season = st.selectbox("Component breakdown — season", seasons_available)
    row = pitcher_rows[pitcher_rows["season"] == sel_season].iloc[0]
    league_row = ps[ps["season"] == sel_season].mean(numeric_only=True)

    components = ["injury_prob_30d", "expected_days_lost", "hazard_rate"]
    labels = ["Injury Prob (30d)", "Expected Days Lost", "Hazard Rate"]
    weights = [0.50, 0.30, 0.20]
    pitcher_vals = [float(row[c]) if c in row.index else 0.0 for c in components]
    league_vals = [float(league_row[c]) if c in league_row.index else 0.0 for c in components]

    fig_comp = go.Figure()
    fig_comp.add_trace(
        go.Bar(
            name=selected_name,
            x=labels,
            y=pitcher_vals,
            marker_color="crimson",
            opacity=0.85,
            text=[f"{v:.3f}" for v in pitcher_vals],
            textposition="outside",
        )
    )
    fig_comp.add_trace(
        go.Bar(
            name=f"League avg ({sel_season})",
            x=labels,
            y=league_vals,
            marker_color="steelblue",
            opacity=0.6,
            text=[f"{v:.3f}" for v in league_vals],
            textposition="outside",
        )
    )
    for label, w in zip(labels, weights):
        fig_comp.add_annotation(
            x=label,
            y=0,
            text=f"wt={w:.0%}",
            showarrow=False,
            yshift=-22,
            font=dict(size=10, color="gray"),
        )
    fig_comp.update_layout(
        barmode="group",
        title=f"Component Breakdown: {selected_name} — {sel_season} (IR+ = {row['ir_plus']:.1f})",
        yaxis_title="Component Value",
        height=420,
        legend=dict(x=0.01, y=0.99),
        margin=dict(t=50),
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    # --- Workload history ---
    pitcher_wl = wl[wl["pitcher_id"] == pitcher_id].sort_values("season")
    if not pitcher_wl.empty:
        fig_wl = go.Figure()
        for col, label, color in [
            ("pitches_7d", "Avg Pitches (7d)", "royalblue"),
            ("pitches_28d", "Avg Pitches (28d)", "darkorange"),
            ("pitches_90d", "Avg Pitches (90d)", "green"),
        ]:
            fig_wl.add_trace(
                go.Scatter(
                    x=pitcher_wl["season"],
                    y=pitcher_wl[col],
                    mode="lines+markers",
                    name=label,
                    line=dict(color=color, width=2),
                    marker=dict(size=6),
                )
            )
        fig_wl.update_layout(
            title=f"{selected_name} — Workload History",
            xaxis_title="Season",
            yaxis_title="Avg Pitches in Window",
            height=360,
            legend=dict(x=0.01, y=0.99),
            margin=dict(t=50),
        )
        st.plotly_chart(fig_wl, use_container_width=True)
