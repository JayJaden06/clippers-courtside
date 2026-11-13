"""Vs. the league: every team's first-half level and second-half change, and the Clippers' ranks.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from courtside.analysis.hypotheses import ordinal
from courtside.viz import dashboard_data as D
from courtside.viz import story as S
from courtside.web.context import current
from courtside.web.theme import BAD, BG, FOCUS, GOOD, GRAY, H1C, H2C, INK_MARK, LIGHT, signed
from courtside.web.ui import answer_cards, next_step, page_top, section, show


def page_league() -> None:
    ctx = current()
    data = ctx.data
    PAGE_WHATNOW = ctx.pages["whatnow"]
    page_top("Vs. the league", "Does every team get worse after halftime? No. Most teams barely change.")
    stat_names = {"net_rtg": "Overall (net rating)", "efg_pct": "Shooting accuracy",
                  "opp_efg_pct": "Opponents' shooting", "tov_pct": "Turnovers",
                  "opp_ft_rate": "Fouling (opponent free throws)", "opp_oreb_pct": "Second chances given up"}
    band = st.container(key="band_navy_league")  # the season picker and the key cards share the navy band
    season_l = band.columns([1, 5])[0].selectbox("Season", data.seasons, index=0, key="league_season")
    cr = D.clippers_ranks(data.ranks, season_l)
    with band:
        answer_cards(S.league_cards(cr))
    section("How every team changes after halftime", "teams",
            "Pick a stat. Each logo is a team: <b>left = worse after halftime, right = better</b>; higher = "
            "better in first halves. Clippers ringed in orange.")
    metric = st.pills("Stat", list(stat_names), default="net_rtg", format_func=stat_names.get, key="league_metric")
    metric = metric or "net_rtg"
    flip = D.better_direction(metric) == "lower is better"
    sign = -1 if flip else 1  # so right and up always mean better
    ch = D.league_view(data.ranks, season_l, metric, "change").set_index("team_name")
    h1 = D.league_view(data.ranks, season_l, metric, "H1").set_index("team_name")
    ids = data.ranks[data.ranks["season"] == season_l].drop_duplicates("team_name").set_index("team_name")["team_id"]
    q = ch.join(h1[["value", "league_mean"]].add_suffix("_h1")).assign(team_id=ids)
    q["x"] = sign * q["value"]
    q["y"] = sign * (q["value_h1"] - q["league_mean_h1"])
    pct = metric not in ("net_rtg", "tov_pct")
    scale = 100 if pct else 1
    fmt = (lambda v: f"{signed(v * 100)} pts") if pct else (lambda v: signed(v))
    q[["x", "y"]] *= scale
    xr = float(q["x"].abs().max()) * 1.18 or 1
    yr = float(q["y"].abs().max()) * 1.22 or 1
    fig = go.Figure()
    corner = dict(showarrow=False, font=dict(size=13, color=INK_MARK), opacity=0.75)
    for x_, y_, xa, ya, text in ((-xr, yr, "left", "top", "Strong starts, fade late"),
                                 (xr, yr, "right", "top", "Strong, and get stronger"),
                                 (-xr, -yr, "left", "bottom", "Slow starts, get worse"),
                                 (xr, -yr, "right", "bottom", "Slow starts, finish strong")):
        fig.add_annotation(x=x_, y=y_, xanchor=xa, yanchor=ya, text=f"<b>{text}</b>", **corner)
    fig.add_shape(type="rect", x0=-xr, x1=0, y0=-yr, y1=yr, fillcolor="rgba(217,89,38,0.06)", line_width=0,
                  layer="below")
    fig.add_shape(type="rect", x0=0, x1=xr, y0=-yr, y1=yr, fillcolor="rgba(57,135,229,0.06)", line_width=0,
                  layer="below")
    fig.add_vline(x=0, line=dict(color=GRAY, width=1))
    fig.add_hline(y=0, line=dict(color=GRAY, width=1, dash="dot"))
    lac = q[q["is_team"]]
    fig.add_trace(go.Scatter(x=lac["x"], y=lac["y"], mode="markers", hoverinfo="skip", showlegend=False,
                             marker=dict(size=46, color="rgba(0,0,0,0)", line=dict(color=FOCUS, width=3))))
    fig.add_trace(go.Scatter(
        x=q["x"], y=q["y"], mode="markers", showlegend=False, marker=dict(size=30, opacity=0),
        customdata=np.column_stack([q.index, [fmt(v) for v in q["value"]], q["rank"],
                                    [fmt(v) for v in q["value_h1"]], [fmt(v) for v in q["league_mean_h1"]]]),
        hovertemplate="<b>%{customdata[0]}</b><br>Change after halftime: %{customdata[1]} "
                      "(%{customdata[2]} of 30)<br>First halves: %{customdata[3]} (league %{customdata[4]})"
                      "<extra></extra>"))
    for name, r_ in q.iterrows():
        fig.add_layout_image(source=D.logo_url(int(r_["team_id"]), dark=False), xref="x", yref="y", x=r_["x"], y=r_["y"],
                             xanchor="center", yanchor="middle", sizex=xr * 0.11, sizey=yr * 0.11,
                             sizing="contain", layer="above")
    fig.update_xaxes(range=[-xr, xr], showticklabels=False, showgrid=False,
                     title="← worse after halftime            better after halftime →")
    fig.update_yaxes(range=[-yr, yr], showticklabels=False, showgrid=False,
                     title="← worse first halves      better first halves →")
    row = q[q["is_team"]].iloc[0]
    left, right = st.columns([2, 1], gap="large")
    with left:
        show(fig, 620)
        st.caption("Center lines: no change at halftime (solid), league-average first half (dotted). Hover a logo "
                   "for the numbers.")
    with right:
        st.metric(f"Clippers: {stat_names[metric]}", f"{ordinal(int(row['rank']))} of 30", border=True,
                  help="Rank of the change from first half to second half. 1st = improves the most.")
        st.metric("Clippers' change", fmt(row["value"]), border=True,
                  help=f"League average change: {fmt(row['league_mean'])}.")
        with st.expander("Every team, ranked"):
            ranked = q.sort_values("rank").reset_index()
            st.dataframe(pd.DataFrame({"logo": [D.logo_url(int(t), dark=False) for t in ranked["team_id"]],
                                       "Team": ranked["team_name"], "Rank": ranked["rank"],
                                       "Change": [fmt(v) for v in ranked["value"]]}),
                         hide_index=True, width="stretch",
                         column_config={"logo": st.column_config.ImageColumn("", width="small")})

    section("Where the Clippers rank on everything", "ranks",
            "Gray dot = their league rank in first halves, orange = second halves. Right = better.")
    stats = [x for x in cr.index if x != "Pace (possessions)"]
    fig = go.Figure()
    for name in stats:
        a, b = cr.loc[name, cr.columns[0]], cr.loc[name, cr.columns[1]]
        fig.add_trace(go.Scatter(x=[a, b], y=[name, name], mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color=BAD if b > a + 2 else GOOD if b < a - 2 else LIGHT, width=3)))
    for col, color, label in ((cr.columns[0], H1C, "1st half"), (cr.columns[1], H2C, "2nd half")):
        # name the two dots on the top row, right above them (the legend stays too)
        fig.add_annotation(x=cr.loc[stats[0], col], y=stats[0], text=label, showarrow=False, yshift=17,
                           font=dict(size=12, color=INK_MARK))
        fig.add_trace(go.Scatter(
            x=cr.loc[stats, col], y=stats, mode="markers", name=label,
            marker=dict(color=color, size=13, line=dict(color=BG, width=1.5)),
            text=[ordinal(int(v)) for v in cr.loc[stats, col]],
            hovertemplate="%{y}: %{text} of 30<extra>" + label + "</extra>"))
    fig.update_xaxes(range=[30.5, 0.5], tickvals=[30, 20, 10, 1], ticktext=["30th", "20th", "10th", "1st"],
                     title="League rank (1st = best)")
    fig.update_yaxes(autorange="reversed")
    show(fig, 90 + 38 * len(stats))
    st.caption("Orange line = they drop 3+ places after halftime; blue = they climb.")
    with st.expander("The ranks as a table"):
        st.dataframe(cr.map(ordinal), width="stretch", height=38 + 35 * len(cr))
    next_step(PAGE_WHATNOW, "So what has to change? What the numbers point to.")
