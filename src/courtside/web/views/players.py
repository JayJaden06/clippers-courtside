"""Players: who loses the ball after halftime, the rotation as cards, and one player game by game.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from courtside.names import short
from courtside.viz import dashboard_data as D
from courtside.viz import story as S
from courtside.web.context import current
from courtside.web.theme import BAD, GOOD, H1C, H2C, INK_MARK, MUTED_BAR
from courtside.web.ui import answer_cards, clicked, face, fill_toc, next_step, page_top, section, show, toc_slot, track


def player_cards(ph: pd.DataFrame, season: str) -> None:
    """The rotation as trading cards: photo, jersey number, and two numbers each, busiest first."""
    ctx = current()
    data = ctx.data
    season_now = ctx.season_now
    h2 = ph[ph["split"] == "H2"].sort_values("play_share", ascending=False)
    roster = (data.roster.set_index("player_id") if len(data.roster) and season == season_now
              else pd.DataFrame(columns=["number", "position"]))
    cards = []
    for r in h2.itertuples():
        info = roster.loc[r.player_id] if r.player_id in roster.index else None
        num = f"#{info['number']}" if info is not None and str(info["number"]).strip() else ""
        pos = info["position"] if info is not None else ""
        every = f"1 in {round(1 / r.tov_pct)}" if r.tov_pct else "none"
        cards.append(
            f'<div class="pcard" title="{r.player_name}"><div class="pc-top"><span class="pc-num">{num}</span>'
            f'<span class="pc-pos">{pos}</span><img src="{D.headshot_url(int(r.player_id))}" alt="" loading="lazy">'
            f'</div><div class="pc-body"><div class="pc-name">{r.player_name}</div><div class="pc-stats">'
            f'<div><b>{r.play_share:.0%}</b><span>of the plays after halftime</span></div>'
            f'<div><b>{every}</b><span>plays ends in a turnover</span></div></div></div></div>')
    st.markdown(f'<div class="pcards">{"".join(cards)}</div>', unsafe_allow_html=True)


def page_players() -> None:
    ctx = current()
    data = ctx.data
    focus = ctx.focus
    season_now = ctx.season_now
    PAGE_LEAGUE = ctx.pages["league"]
    page_top("Players", "Who loses the ball after halftime. Click a card or a player to dig in.")
    nav_ = toc_slot("players")
    band = st.container(key="band_navy_players")  # the season picker and the key cards share the navy band
    season_p = band.columns([1, 5])[0].selectbox("Season", data.seasons, index=0, key="player_season")
    ph = D.player_halves(data.player_stats, season_p)
    if ph.empty:
        band.info("No players with enough minutes in both halves yet.")
    else:
        with band:
            answer_cards(S.player_cards(ph, focus))
            player_cards(ph, season_p)
        h2 = ph[ph["split"] == "H2"].sort_values("tov_share")
        section("Who's losing the ball after halftime?", "who",
                "The further above the line, the more he gives it away for how much he has it.")
        names = sorted(ph["player_name"].unique())
        st.session_state.setdefault("player_pick", focus if focus in names else names[0])
        if st.session_state["player_pick"] not in names:
            st.session_state["player_pick"] = names[0]
        xmax, ymax = float(h2["play_share"].max()) * 1.15, float(h2["tov_share"].max()) * 1.15
        lim = min(xmax, ymax)
        named = set(h2.nlargest(4, "tov_share")["player_name"]) | set(h2.nlargest(2, "play_share")["player_name"])
        fig = go.Figure()
        # above the diagonal = more of the turnovers than of the plays
        fig.add_shape(type="path", path=f"M 0 0 L {lim} {lim} L {lim} {ymax} L 0 {ymax} Z",
                      fillcolor="rgba(217,89,38,0.08)",
                      line_width=0, layer="below")
        fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim, line=dict(color=INK_MARK, width=1.5, dash="dot"))
        fig.add_annotation(x=lim * 0.97, y=lim * 0.97, text="his fair share", showarrow=False, xanchor="right",
                           yanchor="bottom", font=dict(size=12, color=INK_MARK), textangle=-38)
        fig.add_annotation(x=xmax * 0.03, y=ymax * 0.96, text="<b>More than his share of turnovers</b>",
                           showarrow=False, xanchor="left", font=dict(size=13, color=BAD))
        fig.add_trace(go.Scatter(
            x=h2["play_share"], y=h2["tov_share"], mode="markers+text", customdata=h2["player_name"],
            text=[short(n_) if n_ in named else "" for n_ in h2["player_name"]],
            textposition="bottom center",
            textfont=dict(size=12, color=INK_MARK), marker=dict(size=40, opacity=0), showlegend=False,
            hovertemplate="<b>%{customdata}</b><br>%{x:.0%} of the plays after halftime<br>%{y:.0%} of the "
                          "turnovers<extra></extra>"))
        for x_, y_, n_ in zip(h2["play_share"], h2["tov_share"], h2["player_name"]):
            if face(n_):
                fig.add_layout_image(source=face(n_), xref="x", yref="y", x=x_, y=y_, xanchor="center",
                                     yanchor="middle", sizex=xmax * 0.07, sizey=ymax * 0.11, sizing="contain",
                                     layer="above")
        fig.update_xaxes(range=[0, xmax], tickformat=".0%", title="Share of the team's plays after halftime")
        fig.update_yaxes(range=[0, ymax], tickformat=".0%", title="Share of the team's turnovers after halftime")
        event = show(fig, 520, key=f"who_{season_p}", select=True)
        name = clicked(event, "customdata", f"who_click_{season_p}")
        if name in names:
            st.session_state["player_pick"] = name
        st.caption("Each face is a player, placed by how much of the offense he runs (across) and how many of the "
                   "turnovers he makes (up). Above the dotted line = more than his share. Click a face to see his "
                   "games.")

        section("Who gets sloppier after halftime?", "sloppier",
                "How often one of his plays ends in a turnover: before halftime (gray dot) and after it. "
                "Sloppiest after halftime first.")
        wl = ph.pivot(index="player_name", columns="split", values="tov_pct").sort_values("H2", ascending=False)
        every = lambda v: f"1 in {round(1 / v)}" if v and v > 0 else "none"  # noqa: E731
        hi_ = float(wl.to_numpy().max()) * 1.08
        rows_ = []
        for n_, r_ in wl.iterrows():
            d_ = r_["H2"] - r_["H1"]
            color, verdict = ((BAD, "Sloppier") if d_ >= 0.02 else (GOOD, "Safer") if d_ <= -0.02
                              else (MUTED_BAR, "About the same"))
            rows_.append(
                f'<div class="slop-row"><img src="{face(n_) or ""}" alt=""><div class="nm">{short(n_)}'
                f'<span class="chip sm" style="background:{color}">{verdict}</span></div>'
                f'<div class="vals"><span class="b">{every(r_["H1"])}</span> → '
                f'<span style="color:{color}">{every(r_["H2"])}</span></div>'
                + track(r_["H1"], r_["H2"], 0, hi_, color) + "</div>")
        st.markdown(f'<div class="slop">{"".join(rows_)}</div>', unsafe_allow_html=True)
        st.caption("Plays = shots, free-throw trips and turnovers. Sloppier or safer = a change of 2+ percentage "
                   "points in how often his plays end in a turnover. Players with 40+ minutes in each half.")

        section("One player, game by game", "one")
        player = st.selectbox("Player", names, key="player_pick")
        t = ph[ph["player_name"] == player].set_index("split")
        pid_ = int(t["player_id"].iloc[0])
        info = data.roster.set_index("player_id").loc[pid_] if len(data.roster) and pid_ in set(
            data.roster["player_id"]) and season_p == season_now else None
        num = f"#{info['number']}" if info is not None and str(info["number"]).strip() else ""
        d = 100 * (t.loc["H2", "tov_pct"] - t.loc["H1", "tov_pct"])
        st.markdown(
            f'<div class="hero-card"><div class="hc-photo"><span class="pc-num">{num}</span>'
            f'<img src="{D.headshot_url(pid_)}" alt=""></div><div class="hc-body">'
            f'<div class="kicker">{info["position"] + " · " if info is not None else ""}after halftime, {season_p}</div>'
            f'<div class="hc-name">{player}</div><div class="hc-stats">'
            f'<div><b>{t.loc["H1", "tov_pct"]:.0%}</b><span>of his plays end in a turnover before halftime</span></div>'
            f'<div><b class="{"hc-up" if d > 0 else "hc-down"}">{t.loc["H2", "tov_pct"]:.0%}</b><span>after halftime '
            f'({"+" if d > 0 else "−"}{abs(d):.1f} pts)</span></div>'
            f'<div><b>{t.loc["H2", "tov_share"]:.0%}</b><span>of the team\'s turnovers after halftime, on '
            f'{t.loc["H2", "play_share"]:.0%} of its plays</span></div>'
            f'<div><b>{t.loc["H2", "live_share"]:.0%}</b><span>of them steals (live balls that lead to fast breaks)'
            f'</span></div></div></div></div>', unsafe_allow_html=True)
        g = D.player_games(data.player_stats, data.game_leads, season_p, player)
        gx = pd.to_datetime(g["game_date"]).dt.strftime("%b %d").str.replace(" 0", " ") + " " + g[
            "opponent_abbreviation"]
        fig = go.Figure([
            go.Bar(x=gx, y=g["tov_h1"], name="1st half", marker_color=H1C,
                   hovertemplate="%{x}: %{y} turnovers<extra>1st half</extra>"),
            go.Bar(x=gx, y=g["tov_h2"], name="2nd half", marker_color=H2C,
                   hovertemplate="%{x}: %{y} turnovers<extra>2nd half</extra>"),
        ])
        fig.update_layout(barmode="group", bargap=0.3)
        show(fig, 300, yaxis_title="Turnovers")
        with st.expander("Game-by-game table"):
            st.dataframe(g.drop(columns=["game_id"]), hide_index=True, width="stretch")
    fill_toc(nav_)
    next_step(PAGE_LEAGUE, "Is this normal? How every team changes after halftime.")
