import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from dashboard.data_loader import ARCHETYPES, load_pitcher_season

# Categorical slots stepped for a dark surface, kept in this order because the
# ordering itself is what holds adjacent series apart under colour-vision
# deficiency. Validated against Streamlit's #0e1117: worst adjacent CVD ΔE 8.4,
# normal-vision ΔE 19.3, all ≥3:1 contrast.
#
# "Standard" is the residual bucket, so it takes a recessive neutral rather than
# a seventh hue — which also removes the worst normal-vision pair (a violet
# seventh slot sat ΔE 9.8 from the blue first slot, under the 15 floor).
#
# Six hues on one plot cannot clear the all-pairs gate no matter how they are
# ordered, so the dash pattern below carries the starter/reliever split as a
# second channel: any two series that read alike still differ in line style.
ARCHETYPE_COLORS = dict(
    zip(
        ARCHETYPES,
        ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#8b8b86"],
    )
)
ARCHETYPE_DASH = {a: ("solid" if a.endswith("Starter") else "dash") for a in ARCHETYPES}
ARCHETYPE_DASH["Standard"] = "dot"


def render() -> None:
    ps = load_pitcher_season()

    all_seasons = sorted(ps["season"].unique())
    available = [a for a in ARCHETYPES if a in set(ps["archetype"])]

    col1, col2 = st.columns([1, 2])
    with col1:
        first, last = st.select_slider(
            "Seasons",
            options=all_seasons,
            value=(all_seasons[0], all_seasons[-1]),
        )
    with col2:
        picked = st.multiselect("Archetypes", options=available, default=available)

    if not picked:
        st.info("Select at least one archetype.")
        return

    ps = ps[
        ps["season"].between(first, last) & ps["archetype"].isin(picked)
    ]
    if ps.empty:
        st.warning("No pitchers match that season range and archetype selection.")
        return

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

    colors_by_arch = ARCHETYPE_COLORS
    present = set(arch_agg["archetype"])
    # keep taxonomy order, and keep each archetype's colour fixed regardless of
    # which others are selected
    archetypes = [a for a in ARCHETYPES if a in present]

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
        dash = ARCHETYPE_DASH.get(arch, "solid")
        fig1.add_trace(
            go.Scatter(
                x=grp["season"],
                y=grp["mean_ir_plus"],
                mode="lines+markers",
                name=arch,
                line=dict(color=color, width=2, dash=dash),
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
                name=arch,
                line=dict(color=color, width=2, dash=dash),
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
        title="Archetype Comparison",
        height=460,
        legend=dict(orientation="h", y=-0.18, x=0),
        margin=dict(t=60, b=90),
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
        dash = ARCHETYPE_DASH.get(arch, "solid")
        fig2.add_trace(
            go.Scatter(
                x=grp["season"],
                y=grp["mean_days"],
                mode="lines+markers",
                name=arch,
                line=dict(color=color, width=2, dash=dash),
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
                name=arch,
                line=dict(color=color, width=2, dash=dash),
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
        height=440,
        legend=dict(orientation="h", y=-0.18, x=0),
        margin=dict(t=40, b=90),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # --- Snapshot table for the last season in range ---
    latest = int(ps["season"].max())
    st.subheader(f"Season Snapshot ({latest})")
    snap = arch_agg[arch_agg["season"] == latest][
        ["archetype", "mean_ir_plus", "mean_prob", "mean_days", "mean_hazard", "n"]
    ].copy()
    snap["archetype"] = pd.Categorical(snap["archetype"], ARCHETYPES, ordered=True)
    snap = snap.sort_values("archetype")
    snap.columns = ["Archetype", "Mean IR+", "Mean Inj Prob 30d", "Mean Exp Days Lost",
                    "Mean Hazard Rate", "# Pitchers"]
    snap["Mean IR+"] = snap["Mean IR+"].round(1)
    snap["Mean Inj Prob 30d"] = (snap["Mean Inj Prob 30d"] * 100).round(2)
    snap["Mean Exp Days Lost"] = snap["Mean Exp Days Lost"].round(1)
    snap["Mean Hazard Rate"] = snap["Mean Hazard Rate"].round(4)
    snap = snap.set_index("Archetype")
    st.dataframe(snap, use_container_width=True)
