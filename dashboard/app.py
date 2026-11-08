"""Clippers lead-collapse dashboard.

Run with ``leadcollapse dashboard`` (or ``streamlit run dashboard/app.py``).
Reads ``data/processed/`` and ``reports/analysis.json`` only: no API calls.
"""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from leadcollapse.config import get_settings
from leadcollapse.viz import dashboard_data as D

H1, H2 = "#2a78d6", "#eb6834"
HELPED, HURT, NEUTRAL = "#2a78d6", "#e34948", "#898781"
MUTED = "#898781"

st.set_page_config(page_title="Clippers lead collapse", layout="wide")
settings = get_settings()


@st.cache_data
def load() -> D.DashboardData:
    return D.load(settings.processed_dir, settings.reports_dir / "analysis.json")


def style(fig: go.Figure, height: int = 360, **layout) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=30, b=10), hovermode="closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0), **layout)
    fig.update_xaxes(showgrid=False)
    return fig


try:
    data = load()
except FileNotFoundError as e:
    st.error(f"Processed data not found ({e.filename}). Run `leadcollapse build` first.")
    st.stop()

st.title("Why do the Clippers lose leads after halftime?")
meta = data.analysis.get("meta", {})
if meta:
    st.caption(f"{meta['season']}: {meta['games']} games, compared with {meta['comparison_season']}. "
               f"Analyzed {meta['analyzed_on']}. Source: stats.nba.com via nba_api.")

season = st.sidebar.selectbox("Season", data.seasons, index=0)
st.sidebar.caption("The verdicts on the Overview tab are for the current season only.")
overview, games_tab, players_tab, league_tab = st.tabs(["Overview", "Games", "Players", "League ranks"])

# ------------------------------------------------------------------ overview
with overview:
    results = data.analysis.get("results", [])
    if not results:
        st.info("No analysis yet. Run `leadcollapse analyze`.")
    for r in results:
        with st.expander(f"**{r['id']}**: {r['title']}", expanded=r["id"] in ("H3", "ATTR")):
            st.write(r["verdict"])
    fig_path = settings.figures_dir / "attribution.png"
    if fig_path.exists():
        st.image(str(fig_path), caption="Where the second-half points went (from reports/figures/)")

# --------------------------------------------------------------------- games
with games_tab:
    options = D.game_options(data.game_leads, season)
    choice = st.selectbox("Game", options["label"], index=0)
    game = options[options["label"] == choice].iloc[0]
    gid = game["game_id"]

    m = D.game_margin(data.margins, gid)
    blown = game["lead_class"] == "blown_lead"
    color = H2 if blown else H1
    fig = go.Figure(go.Scatter(
        x=m["minute"], y=m["margin"], mode="lines", line=dict(color=color, width=2, shape="hv"),
        fill="tozeroy", fillcolor="rgba(235,104,52,0.10)" if blown else "rgba(42,120,214,0.10)",
        customdata=m[["team_score", "opp_score", "period"]],
        hovertemplate="Minute %{x:.1f} (Q%{customdata[2]})<br>Margin %{y:+d}<br>"
                      "Score %{customdata[0]}-%{customdata[1]}<extra></extra>",
        name="Clippers margin",
    ))
    fig.add_vline(x=24, line=dict(color=MUTED, width=1), annotation_text="half", annotation_position="bottom right")
    fig.add_hline(y=0, line=dict(color=MUTED, width=1))
    style(fig, xaxis_title="Game minute", yaxis_title="Clippers margin", showlegend=False)
    st.plotly_chart(fig, width="stretch")

    c1, c2, c3 = st.columns(3)
    c1.metric("Biggest lead", f"{int(game['max_lead'])}")
    c2.metric("Halftime margin", f"{int(game['halftime_margin']):+d}")
    c3.metric("Final margin", f"{int(game['final_margin']):+d}")

    left, right = st.columns(2)
    with left:
        st.subheader("1st half vs 2nd half")
        st.dataframe(D.game_half_table(data.game_halves, gid), hide_index=True, width="stretch")
    with right:
        st.subheader("Where the margin change went")
        att = D.game_attribution(data.game_halves, gid)
        colors = [NEUTRAL if f == "residual" else HELPED if v >= 0 else HURT
                  for f, v in zip(att["factor"], att["points"])]
        fig = go.Figure(go.Bar(x=att["points"], y=att["label"], orientation="h", marker_color=colors,
                               hovertemplate="%{y}: %{x:+.1f} pts<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.add_vline(x=0, line=dict(color=MUTED, width=1))
        style(fig, height=340, xaxis_title=f"Points (total change {att['margin_change'].iloc[0]:+.0f})")
        st.plotly_chart(fig, width="stretch")
        st.caption("Blue = better after halftime, red = worse, gray = unattributed. Factors sum to the total.")

    st.subheader("Clippers turnovers")
    st.dataframe(D.game_turnovers(data.turnovers, gid), hide_index=True, width="stretch")

# ------------------------------------------------------------------- players
with players_tab:
    ph = D.player_halves(data.player_stats, season)
    if ph.empty:
        st.info("No players with enough minutes in both halves yet.")
    else:
        w = ph.pivot(index="player_name", columns="split", values="tov_pct").sort_values("H2")
        tov = ph.pivot(index="player_name", columns="split", values="tov").reindex(w.index)
        share = ph.pivot(index="player_name", columns="split", values="tov_share").reindex(w.index)
        fig = go.Figure()
        for name in w.index:
            fig.add_trace(go.Scatter(x=[w.loc[name, "H1"], w.loc[name, "H2"]], y=[name, name], mode="lines",
                                     line=dict(color="#d8d7d0", width=2), hoverinfo="skip", showlegend=False))
        for split, color, label in (("H1", H1, "1st half"), ("H2", H2, "2nd half")):
            fig.add_trace(go.Scatter(
                x=w[split], y=w.index, mode="markers", name=label,
                marker=dict(color=color, size=11, line=dict(color="white", width=1.5)),
                customdata=np.column_stack([tov[split], share[split]]),
                hovertemplate="%{y}<br>TOV% %{x:.1%}<br>%{customdata[0]:.0f} turnovers, "
                              "%{customdata[1]:.0%} of team<extra>" + label + "</extra>",
            ))
        style(fig, height=60 + 34 * len(w), xaxis_title="TOV% (turnovers per 100 plays)", xaxis_tickformat=".0%")
        fig.update_yaxes(tickvals=list(w.index), ticktext=[f"<b>{n}</b>" if n == settings.focus_player else n
                                                           for n in w.index])
        st.subheader("Turnover rate by half")
        st.plotly_chart(fig, width="stretch")
        st.caption("Players with 40+ minutes in each half. Hover for turnover counts and share of team turnovers.")

        names = sorted(ph["player_name"].unique())
        default = names.index(settings.focus_player) if settings.focus_player in names else 0
        player = st.selectbox("Player", names, index=default)
        games = D.player_games(data.player_stats, data.game_leads, season, player)
        totals = ph[ph["player_name"] == player].set_index("split")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("1st-half TOV%", f"{totals.loc['H1', 'tov_pct']:.1%}")
        c2.metric("2nd-half TOV%", f"{totals.loc['H2', 'tov_pct']:.1%}",
                  delta=f"{100 * (totals.loc['H2', 'tov_pct'] - totals.loc['H1', 'tov_pct']):+.1f} pts",
                  delta_color="inverse")
        c3.metric("2nd-half share of team TOV", f"{totals.loc['H2', 'tov_share']:.0%}")
        c4.metric("2nd-half live-ball share", f"{totals.loc['H2', 'live_share']:.0%}")
        show = games.rename(columns={
            "game_date": "date", "opponent_abbreviation": "opp", "final_margin": "final", "lead_class": "game type",
            "minutes_h1": "min 1H", "minutes_h2": "min 2H", "tov_h1": "TOV 1H", "tov_h2": "TOV 2H",
            "tov_live_h1": "live 1H", "tov_live_h2": "live 2H", "tov_pct_h1": "TOV% 1H", "tov_pct_h2": "TOV% 2H",
        }).drop(columns=["game_id"])
        st.dataframe(show, hide_index=True, width="stretch", column_config={
            "date": st.column_config.DateColumn(format="MMM D"),
            "min 1H": st.column_config.NumberColumn(format="%.1f"),
            "min 2H": st.column_config.NumberColumn(format="%.1f"),
            "TOV% 1H": st.column_config.NumberColumn(format="percent"),
            "TOV% 2H": st.column_config.NumberColumn(format="percent"),
        })

# -------------------------------------------------------------------- league
with league_tab:
    st.subheader("Clippers league ranks (1 = best)")
    cr = D.clippers_ranks(data.ranks, season)
    st.dataframe(cr, width="stretch", height=38 + 35 * len(cr))

    metrics = [m for m in D.METRIC_LABELS if m in set(data.ranks["metric"])]
    c1, c2 = st.columns(2)
    metric = c1.selectbox("Metric", metrics, index=metrics.index("net_rtg"), format_func=D.METRIC_LABELS.get)
    split = c2.radio("Split", ["H1", "H2", "change"], index=2, horizontal=True,
                     format_func={"H1": "1st half", "H2": "2nd half", "change": "Change (1H to 2H)"}.get)
    lv = D.league_view(data.ranks, season, metric, split)
    fig = go.Figure(go.Bar(
        x=lv["value"], y=lv["team_name"], orientation="h",
        marker_color=[H2 if t else "#c3c2b7" for t in lv["is_team"]],
        customdata=lv["rank"], hovertemplate="%{y}<br>%{x:.3f}<br>rank %{customdata}<extra></extra>",
    ))
    fig.add_vline(x=lv["league_mean"].iloc[0], line=dict(color=MUTED, width=1),
                  annotation_text="league average", annotation_position="top")
    fig.update_yaxes(autorange="reversed")
    style(fig, height=760, xaxis_title=f"{D.METRIC_LABELS[metric]} ({D.better_direction(metric)}; best at top)")
    st.plotly_chart(fig, width="stretch")
