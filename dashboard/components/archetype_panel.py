import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.data_loader import load_pitcher_season


def render() -> None:
    ps = load_pitcher_season()

    arch_agg = (
        ps.groupby(["archetype", "season"])
        .agg(
            mean_ir_plus=("ir_plus", "mean"),
            mean_prob=("injury_prob_30d", "mean"),
            mean_days=("expected_days_lost", "mean"),
            mean_hazard=("hazard_rate", "mean"),
            n=("pitcher_id", "count"),
        )
        .reset_index()
    )

    colors_by_arch = {"starter": "steelblue", "reliever": "darkorange"}
    archetypes = sorted(arch_agg["archetype"].unique().tolist())

    # --- Row 1: IR+ and injury probability ---
    fig1 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Mean Injury Risk+ by Season", "Mean 30-Day Injury Probability (%)"],
        horizontal_spacing=0.10,
    )
    for arch in archetypes:
        grp = arch_agg[arch_agg["archetype"] == arch].sort_values("season")
        color = colors_by_arch.get(arch, "gray")
        fig1.add_trace(
            go.Scatter(
                x=grp["season"],
                y=grp["mean_ir_plus"],
                mode="lines+markers",
                name=arch.capitalize(),
                line=dict(color=color, width=2),
                marker=dict(size=7),
                legendgroup=arch,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig1.add_trace(
            go.Scatter(
                x=grp["season"],
                y=grp["mean_prob"] * 100,
                mode="lines+markers",
                name=arch.capitalize(),
                line=dict(color=color, width=2),
                marker=dict(size=7),
                legendgroup=arch,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig1.add_hline(y=100, line_dash="dash", line_color="gray", row=1, col=1)
    fig1.update_xaxes(title_text="Season")
    fig1.update_yaxes(title_text="Mean IR+", row=1, col=1)
    fig1.update_yaxes(title_text="Mean 30d Injury Prob (%)", row=1, col=2)
    fig1.update_layout(
        title="Archetype Comparison: Starters vs. Relievers",
        height=420,
        legend=dict(x=0.01, y=0.99),
        margin=dict(t=60),
    )
    st.plotly_chart(fig1, use_container_width=True)

    # --- Row 2: Expected days lost and hazard rate ---
    fig2 = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Mean Expected Days Lost", "Mean Hazard Rate"],
        horizontal_spacing=0.10,
    )
    for arch in archetypes:
        grp = arch_agg[arch_agg["archetype"] == arch].sort_values("season")
        color = colors_by_arch.get(arch, "gray")
        fig2.add_trace(
            go.Scatter(
                x=grp["season"],
                y=grp["mean_days"],
                mode="lines+markers",
                name=arch.capitalize(),
                line=dict(color=color, width=2),
                marker=dict(size=7),
                legendgroup=arch,
                showlegend=True,
            ),
            row=1,
            col=1,
        )
        fig2.add_trace(
            go.Scatter(
                x=grp["season"],
                y=grp["mean_hazard"],
                mode="lines+markers",
                name=arch.capitalize(),
                line=dict(color=color, width=2),
                marker=dict(size=7),
                legendgroup=arch,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig2.update_xaxes(title_text="Season")
    fig2.update_yaxes(title_text="Mean Exp. Days Lost", row=1, col=1)
    fig2.update_yaxes(title_text="Mean Hazard Rate", row=1, col=2)
    fig2.update_layout(
        height=400,
        legend=dict(x=0.01, y=0.99),
        margin=dict(t=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # --- Latest-season snapshot table ---
    latest = int(ps["season"].max())
    st.subheader(f"Latest Season Snapshot ({latest})")
    snap = arch_agg[arch_agg["season"] == latest][
        ["archetype", "mean_ir_plus", "mean_prob", "mean_days", "mean_hazard", "n"]
    ].copy()
    snap.columns = ["Archetype", "Mean IR+", "Mean Inj Prob 30d", "Mean Exp Days Lost",
                    "Mean Hazard Rate", "# Pitchers"]
    snap["Mean IR+"] = snap["Mean IR+"].round(1)
    snap["Mean Inj Prob 30d"] = (snap["Mean Inj Prob 30d"] * 100).round(2)
    snap["Mean Exp Days Lost"] = snap["Mean Exp Days Lost"].round(1)
    snap["Mean Hazard Rate"] = snap["Mean Hazard Rate"].round(4)
    snap = snap.set_index("Archetype")
    st.dataframe(snap, use_container_width=True)
