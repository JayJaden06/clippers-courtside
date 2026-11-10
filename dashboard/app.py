"""Clippers Courtside dashboard.

Run with ``courtside dashboard`` (or ``streamlit run dashboard/app.py``).
Reads ``data/processed/`` and ``reports/analysis.json`` only: no API calls.
Written for two audiences at once: plain-English headlines and explanations
up front, the basketball numbers one hover or expander away.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from courtside.analysis import gamereport as R
from courtside.analysis import gameplan as GP
from courtside.analysis import preview as P
from courtside.ingest import schedule as sched
from courtside.analysis.hypotheses import CONTRIBUTES, DRIVES, ordinal
from courtside.config import get_settings
from courtside.viz import dashboard_data as D
from courtside.viz import graphics as G
from courtside.viz import story as S
from courtside.viz import whatnow as W

# Dark-surface steps of the report palette (validated for the dark chart surface).
H1C, H2C = "#3987e5", "#d95926"  # 1st half / 2nd half
GOOD, BAD, GRAY = "#3987e5", "#e66767", "#898781"
FOCUS = "#d95926"
INK_MARK = "#e8e6df"
LIGHT = "rgba(137,135,129,0.45)"
MUTED_BAR = "#5f5e5a"
BG = "#141413"  # matches .streamlit/config.toml backgroundColor
PLOTLY_CONFIG = {"displayModeBar": False}
TONE_BADGE = {"yes": ("red", "The answer"), "partly": ("orange", "Part of it"), "no": ("gray", "Not really")}

st.set_page_config(page_title="Clippers Courtside", layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
.kicker {text-transform: uppercase; letter-spacing: .09em; font-size: .78rem; opacity: .65; margin-bottom: .2rem}
.hero {font-size: clamp(1.9rem, 4vw, 3rem); font-weight: 800; line-height: 1.08; margin: 0 0 .6rem 0}
.lede {font-size: 1.15rem; opacity: .85; max-width: 52rem; margin-bottom: 1.2rem}
.story {font-size: 1.12rem; line-height: 1.55}
.big-q {font-size: 1.05rem; font-weight: 700; margin-bottom: .3rem}
.rank {font-size: 2.4rem; font-weight: 800; line-height: 1; text-align: center; opacity: .8}
div[data-testid="stMetricValue"] {font-size: 1.9rem}
.jump {display: flex; flex-wrap: wrap; gap: .45rem; margin: .4rem 0 1.4rem 0}
.jump a {text-decoration: none; font-size: .9rem; padding: .3rem .8rem; border-radius: 999px;
         border: 1px solid rgba(137,135,129,.45); color: inherit}
.jump a:hover {border-color: #3987e5; color: #3987e5}
.tldr {border-left: 4px solid #d95926; padding: .2rem 0 .2rem 1rem; margin: .6rem 0 .4rem 0}
.tldr li {margin-bottom: .35rem; font-size: 1.05rem}
.vcards {display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: .8rem; margin: .4rem 0 1rem 0}
.vcard {display: flex; flex-direction: column; gap: .15rem; text-decoration: none !important; color: inherit !important;
        background: linear-gradient(180deg, #22221f 0%, #1b1b19 100%); border: 1px solid #2c2c29; border-radius: 14px;
        padding: .9rem 1rem .85rem 1rem; transition: transform .12s, border-color .12s}
a.vcard:hover {transform: translateY(-2px); border-color: #4a4a45}
.v-title {font-weight: 650; font-size: .9rem; line-height: 1.25; opacity: .8}
.v-dot {display: inline-block; width: .55rem; height: .55rem; border-radius: 50%; margin-right: .4rem;
        vertical-align: .08rem}
.v-big {font-size: 1.95rem; font-weight: 800; line-height: 1.1; margin: .15rem 0 .1rem 0; letter-spacing: -.01em}
.v-mini svg {width: 100%; max-width: 260px; height: 40px; display: block; margin: .15rem 0 .25rem 0}
.v-small {font-size: .84rem; opacity: .7; line-height: 1.32}
.v-img {width: 74px; height: 54px; object-fit: cover; object-position: top; border-radius: 10px;
         background: #2c2c29; margin-bottom: .35rem}
.vcard:has(.v-img) .v-big {font-size: 1.6rem}
[data-testid="stLayoutWrapper"]:has(> .st-key-gamebar) {position: sticky; top: 3.6rem; z-index: 60}
.st-key-gamebar {background: #141413; padding: .35rem 0 .55rem 0; border-bottom: 1px solid #2c2c29}
.lineup {display: flex; align-items: center; gap: .9rem; padding: .55rem .2rem; border-bottom: 1px solid #262624}
.lineup .faces {display: flex}
.lineup .faces img {width: 52px; height: 52px; border-radius: 50%; object-fit: cover; object-position: top;
                    background: #2c2c29; border: 2px solid #141413; margin-left: -10px}
.lineup .faces img:first-child {margin-left: 0}
.lineup .who {flex: 1; font-size: .92rem; opacity: .85}
.lineup .pm {font-size: 1.35rem; font-weight: 800; min-width: 4.5rem; text-align: right}
.lineup .min {font-size: .82rem; opacity: .6; min-width: 3.5rem}
[class*="st-key-gcard_"] {padding: .55rem .6rem !important; gap: .25rem}
.st-key-gcard_sel {border-color: #d95926 !important; background: #1f1a17}
.gc {padding-bottom: .9rem}
.gc-date {font-size: .78rem; opacity: .7}
.gc-opp {font-weight: 750; font-size: 1.02rem}
.gc-status {font-weight: 700; font-size: .9rem}
.ledger-sum {font-size: 1.25rem; margin: .2rem 0 .8rem 0}
.ledger {background: #1b1b19; border: 1px solid #2c2c29; border-radius: 14px; padding: .8rem 1rem}
.ledger .lhead {font-weight: 750; margin-bottom: .45rem}
.ledger-row {display: grid; grid-template-columns: 1.6fr 1fr auto; gap: .7rem; align-items: center;
             padding: .38rem 0; border-top: 1px solid #262624}
.ledger .lname {font-size: .95rem}
.ledger .lbar {height: 10px; background: #2a2a27; border-radius: 6px; overflow: hidden}
.ledger .lbar span {display: block; height: 100%; border-radius: 6px}
.ledger .lval {font-weight: 800; font-size: 1.1rem; min-width: 3.4rem; text-align: right}
.ledger .lempty {opacity: .6; font-size: .9rem}
.kval {font-weight: 800; font-size: 1.05rem; color: #3987e5; margin-top: .3rem}
.kval span {font-weight: 500; font-size: .82rem; color: inherit; opacity: .75}
.kwhy {font-size: .9rem; opacity: .85; line-height: 1.4}
.lean {display: flex; gap: .55rem; align-items: center; margin: .2rem 0}
.lean img {width: 40px; height: 40px; border-radius: 50%; object-fit: cover; object-position: top;
           background: #2c2c29}
.lean span {font-size: .8rem; opacity: .7}
.odds-note {text-align: center; font-size: 1.1rem; margin: -.2rem 0 .8rem 0}
.takeaway {font-size: 1.18rem; margin: .2rem 0 .8rem 0}
.page-sub {opacity: .75; font-size: 1.05rem; margin-top: -.6rem; margin-bottom: .8rem}
</style>
""", unsafe_allow_html=True)

settings = get_settings()


@st.cache_data
def load() -> D.DashboardData:
    return D.load(settings.processed_dir, settings.reports_dir / "analysis.json")


def show(fig: go.Figure, height: int, key: str | None = None, select: bool = False, **layout):
    """Render a plotly figure; with ``select=True`` clicks on points rerun the app and are returned."""
    settings_ = dict(height=height, margin=dict(l=8, r=8, t=36, b=8), hovermode="closest",
                     legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None, font=dict(size=13)),
                     hoverlabel=dict(align="left", font=dict(size=13)), font=dict(size=13),
                     barcornerradius=4, clickmode="event+select" if select else None)
    fig.update_layout(**(settings_ | layout))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(zeroline=False)
    return st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG, key=key,
                           on_select="rerun" if select else "ignore", selection_mode="points")


def picture(svg: str, max_width: str | None = None) -> None:
    """Render one of the SVG graphics from courtside.viz.graphics."""
    style = f' style="max-width:{max_width};margin-left:auto;margin-right:auto"' if max_width else ""
    st.markdown(svg.replace('<div class="gfx">', f'<div class="gfx"{style}>', 1), unsafe_allow_html=True)


def clicked(event, field: str, memo: str):
    """The clicked point's ``field`` if this click is new since last run (so widgets can still be used)."""
    pts = event.selection.points if event and event.selection else []
    value = pts[0].get(field) if pts else None
    if value is None or st.session_state.get(memo) == value:
        return None
    st.session_state[memo] = value
    return value


def page_top(title: str, subtitle: str) -> None:
    left, right = st.columns([4, 1], vertical_alignment="bottom")
    left.title(title)
    with right:
        glossary_button()
    st.markdown(f'<div class="page-sub">{subtitle}</div>', unsafe_allow_html=True)


def signed(v: float, digits: int = 1) -> str:
    return f"{v:+.{digits}f}".replace("-", "−")


try:
    data = load()
except FileNotFoundError as e:
    st.error(f"Processed data not found ({e.filename}). Run `courtside build` and `courtside analyze` first.")
    st.stop()

meta = data.analysis.get("meta", {})
season_now = meta.get("season", data.seasons[0])
season_prev = meta.get("comparison_season", data.seasons[-1])
focus = settings.focus_player
focus_last = focus.split(" ", 1)[-1]



def glossary_button() -> None:
    with st.popover("What do these words mean?", icon=":material/menu_book:"):
        for term, text in S.GLOSSARY.items():
            st.markdown(f"**{term}**  \n{text}")



# ================================================================ the answer
STATUS_BADGE = {S.NEW: "red", S.SAME: "orange", S.SIMILAR: "gray", S.BETTER: "blue"}
VERDICT_MARK = {"Worse than last season": "▼ Worse than last season", "Better than last season": "▲ Better than last season",
                "About the same": "= About the same"}


def page_answer() -> None:
    hl = S.headline(data.analysis, data.comparison, data.games, season_now) if not data.comparison.empty else None
    att_now, att_prev = data.attribution.get(season_now), data.attribution.get(season_prev)
    res = {r["id"]: r for r in data.analysis.get("results", [])}
    sh = D.season_halves(data.game_halves)
    reasons = (S.ranked_reasons(att_now, att_prev, sh.loc[season_now])
               if att_now is not None and att_prev is not None else [])
    top = reasons[0].factor if reasons else None
    last_game = pd.to_datetime(data.game_leads.loc[data.game_leads["season"] == season_now, "game_date"]).max()
    st.markdown(f'<div class="kicker">LA Clippers · {season_now} season · through {last_game:%b} {last_game.day}'
                "</div>", unsafe_allow_html=True)
    title = ("They're fine for a half. Then the shots stop falling." if top == "shooting"
             else "They're fine for a half. Then it falls apart.")
    st.markdown(f'<div class="hero">{title}</div>', unsafe_allow_html=True)
    if hl:
        garland_main = "H3" in res and res["H3"]["key"]["share_of_swing"] >= DRIVES
        st.markdown(
            f'<div class="lede">{hl["wins"]}-{hl["losses"]}, and {hl["blown"]} of {hl["leads"]} early leads blown. '
            "Is it Darius Garland's turnovers? "
            + ("Mostly, yes." if garland_main else "Partly, but not mainly.")
            + " Click any card to see the chart behind it.</div>", unsafe_allow_html=True)
        picture(G.court_halves(hl["h1_per_game"], hl["h2_per_game"]), max_width="760px")
        # The court above already shows the half-by-half margin, and section 3 lists what's new.
        answer_cards([c_ for c_ in S.answer_page_cards(hl, reasons, res, season_prev)
                      if not c_.title.startswith(("Fine before", "New problems"))])
        glossary_button()

    hurts = [r for r in reasons if r.per_game < 0]
    helps = [r for r in reasons if r.per_game >= 0]

    # ---------------------------------------------------------- the 60-second version
    if hl and reasons:
        with st.expander("Read the 60-second version as text"):
            st.markdown("\n".join(f"- {line}" for line in S.sixty_seconds(reasons, res, hl, season_prev)),
                        unsafe_allow_html=True)
        b1, b2, b3 = st.columns(3)
        worst = S.worst_collapse(data.game_leads, season_now)
        if worst is not None and b1.button(f"Watch the worst collapse ({worst['label']})", icon=":material/play_arrow:",
                                            type="primary", width="stretch"):
            st.session_state["jump_game"] = worst["game_id"]
            st.switch_page(PAGE_REPLAY)
        if b2.button("What needs to happen now →", icon=":material/checklist:", width="stretch"):
            st.switch_page(PAGE_WHATNOW)
        if b3.button("Compare with all 30 teams →", icon=":material/leaderboard:", width="stretch"):
            st.switch_page(PAGE_LEAGUE)
    st.markdown('<div class="jump">'
                '<a href="#reasons">1 · Ranked reasons</a>'
                '<a href="#theories">Theories checked</a><a href="#changes">Since last season</a>'
                '<a href="#whatif">What if?</a><a href="#stats">Every stat</a></div>', unsafe_allow_html=True)

    # ---------------------------------------------------------- 1. ranked reasons (costs and helps)
    total_help = sum(r.per_game for r in helps)
    section("1. Why they blow leads, ranked", "reasons",
            "Each part of the game, after halftime: <b>red = this season</b>, gray = last season. Left of zero "
            "costs them points; right of zero helps." if reasons else "")
    if reasons:
        ordered = hurts + sorted(helps, key=lambda r: -r.per_game)
        names = [f"{i}. {r.name}" for i, r in enumerate(ordered, 1)]
        fig = go.Figure()
        for label, vals, color in ((f"This season ({season_now})", [r.per_game for r in ordered], BAD),
                                   (f"Last season ({season_prev})", [r.prev_per_game for r in ordered], MUTED_BAR)):
            fig.add_trace(go.Bar(
                y=names, x=vals, orientation="h", name=label, marker_color=color,
                text=[f"{signed(v)} a game" for v in vals], textposition="outside", cliponaxis=False,
                customdata=[r.detail for r in ordered],
                hovertemplate="<b>%{y}</b><br>" + label + ": %{x:+.1f} points a game after halftime"
                              "<br>%{customdata}<extra></extra>"))
        lo = min(min(r.per_game, r.prev_per_game) for r in ordered)
        hi = max(max(r.per_game, r.prev_per_game) for r in ordered)
        fig.add_vline(x=0, line=dict(color=INK_MARK, width=1))
        fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Points a game after halftime  (← costs them · helps them →)",
                         range=[min(lo, 0) * 1.45 - 0.8, max(hi, 0) * 1.45 + 0.8], zeroline=False)
        show(fig, 110 + 62 * len(ordered))
        if helps and hl:
            st.caption(f"The ones right of zero hide part of the drop: without them, second halves would be "
                       f"{signed(hl['h2_per_game'] - total_help)} a game instead of {signed(hl['h2_per_game'])}.")
        with st.expander("Details for each reason"):
            for i, r in enumerate(ordered, 1):
                verb = f"costs **{-r.per_game:.1f} points a game**" if r.per_game < 0 else \
                    f"gains **{r.per_game:.1f} points a game**"
                st.markdown(f"**{i}. {r.name}** {verb} after halftime. {r.detail} "
                            f"Last season: {signed(r.prev_per_game)} a game.")
                st.badge(r.status, color=STATUS_BADGE[r.status])
            if att_now is not None:
                residual = att_now.set_index("factor").loc["residual", "per_game"]
                st.markdown(f"A further {signed(residual)} a game can't be placed in any one part.")

    # ---------------------------------------------------------- 3. theories
    section("2. The popular theories, checked", "theories")
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.subheader("Is it Garland?")
        if "H3" in res:
            k3 = res["H3"]["key"]
            share = k3["share_of_swing"]
            fig = go.Figure()
            for x0, x1, label, shade in ((0, CONTRIBUTES, "minor", 0.08), (CONTRIBUTES, DRIVES, "part of it", 0.18),
                                         (DRIVES, 1.0, "the main cause", 0.32)):
                fig.add_shape(type="rect", x0=x0, x1=x1, y0=-0.5, y1=0.5, line_width=0,
                              fillcolor=f"rgba(230,103,103,{shade})", layer="below")
                fig.add_annotation(x=(x0 + x1) / 2, y=0.72, text=label, showarrow=False, font=dict(size=12))
            fig.add_trace(go.Bar(x=[min(share, 1)], y=[0], orientation="h", width=0.34, marker_color=FOCUS,
                                 hovertemplate=f"Garland's extra turnovers explain {share:.0%} of the "
                                               f"second-half drop<extra></extra>"))
            fig.add_annotation(x=min(share, 1), y=-0.62, text=f"<b>{share:.0%}</b>", showarrow=False,
                               font=dict(size=20))
            fig.update_xaxes(range=[0, 1], tickformat=".0%", tickvals=[0, CONTRIBUTES, DRIVES, 1])
            fig.update_yaxes(visible=False, range=[-0.95, 0.95])
            show(fig, 190, showlegend=False)
            st.markdown(f"His extra turnovers explain about **{share:.0%}** of the drop. "
                        + (f"That's real, but short of the **{DRIVES:.0%}** we set in advance for calling something "
                           "*the* cause." if share < DRIVES else "That's enough to call him the main cause."))
    with right:
        cards = S.answer_cards(data.analysis)
        answer_cards(S.theory_cards(cards), link=False, big_size="1.3rem")
        with st.expander("The full answer to each theory"):
            for card in cards:
                st.markdown(f"**{card.question}** {card.answer}. {card.text}")

    # ---------------------------------------------------------- The Wall
    ft = D.opp_ft_by_half(data.game_halves)
    wall = S.wall_check(ft)
    if wall is not None:
        st.subheader("Is it The Wall?", anchor="wall")
        dome = ft[(ft["where"] == "home") & (ft["season"] >= S.INTUIT_DOME_FIRST_SEASON)]
        by = dome.groupby("split")[["ftm", "fta"]].sum()
        st.markdown(f'<div class="takeaway"><b>{wall.big}.</b> {wall.small}</div>', unsafe_allow_html=True)
        picture(G.wall_court(by.loc["H1", "ftm"] / by.loc["H1", "fta"], by.loc["H2", "ftm"] / by.loc["H2", "fta"]),
                max_width="900px")
        st.link_button("See the full Intuit Dome study", settings.wall_study_url, icon=":material/open_in_new:")
        with st.expander("Each season, at home and on the road"):
            order = sorted(ft["season"].unique(), reverse=True)
            x = [f"{s_}<br>{'At Intuit Dome' if w_ == 'home' else 'On the road'}" for s_ in order
                 for w_ in ("home", "road")]
            fig = go.Figure()
            for split, color, label in (("H1", H1C, "Before halftime"), ("H2", H2C, "After halftime")):
                rows = [ft[(ft["season"] == s_) & (ft["where"] == w_) & (ft["split"] == split)]
                        for s_ in order for w_ in ("home", "road")]
                vals = [float(r_["ft_pct"].iloc[0]) if len(r_) else None for r_ in rows]
                att = [int(r_["fta"].iloc[0]) if len(r_) else 0 for r_ in rows]
                fig.add_trace(go.Bar(x=x, y=vals, name=label, marker_color=color, customdata=att,
                                     text=[f"{v:.0%}" if v is not None else "" for v in vals],
                                     textposition="outside",
                                     hovertemplate="%{x}<br>" + label + ": %{y:.1%} on %{customdata} free throws"
                                                   "<extra></extra>"))
            fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.06)
            fig.update_yaxes(tickformat=".0%", range=[0, 1.12], title="Opponents' free throws made")
            show(fig, 320)
        with st.expander("Details"):
            st.markdown(
                "- Teams switch baskets at halftime, so visitors face The Wall for only one half. The box score "
                "doesn't record which end a team shot at, so this checks whether opponents miss more after "
                "halftime at home at all. They don't.\n"
                "- Includes late free throws from intentional fouls.\n"
                "- The home-court study looks at every arena since 2013-14 and finds the same thing: visitors' "
                "free-throw shooting at Intuit Dome isn't consistently worse than at other arenas.")

    # ---------------------------------------------------------- 3. since last season
    section_n = 3
    if not data.comparison.empty and season_prev in data.attribution:
        st.header(f"{section_n}. What changed since last season", divider="gray", anchor="changes")
        comp = data.comparison.set_index("season")
        cur, prev = comp.loc[season_now], comp.loc[season_prev]
        cur_leads = int(cur["blown_leads"] + cur["held_leads"])
        prev_leads = int(prev["blown_leads"] + prev["held_leads"])
        new = [r.name for r in reasons if r.status == S.NEW]
        same = [r.name for r in reasons if r.status == S.SAME]
        better = [r.name for r in reasons if r.status == S.BETTER]
        answer_cards([
            W.Verdict("Early leads blown", f"{int(cur['blown_leads'])} of {cur_leads}",
                      f"Last season: {int(prev['blown_leads'])} of {prev_leads} "
                      f"({prev['blown_leads'] / max(prev_leads, 1):.0%}).", "bad", "changes",
                      ("pair", prev["blown_leads"] / max(prev_leads, 1), cur["blown_leads"] / max(cur_leads, 1),
                       "last year", "now")),
            W.Verdict("Second halves, a game", signed(cur["margin_h2_per_game"]),
                      f"Last season: {signed(prev['margin_h2_per_game'])} a game.", "bad"
                      if cur["margin_h2_per_game"] < prev["margin_h2_per_game"] else "good", "changes",
                      ("diverge", (prev["margin_h2_per_game"], cur["margin_h2_per_game"]), ("last year", "now"))),
        ] + [W.Verdict(title, " · ".join(names), sub, tone, "changes")
             for title, names, sub, tone in (
                 ("New this season", new, "Worse than last season by a point a game or more.", "bad"),
                 ("Same as last season", same, "Hurt then, hurts now.", "warn"),
                 ("Better than last season", better, "Hurts less than it did.", "good")) if names],
            link=False, big_size="1.15rem")
        with st.expander("Every stat, this season vs last (table)"):
            rows = S.scorecard(sh, season_now, season_prev)
            table = pd.DataFrame([{
                "": r.stat, f"{season_prev} 1st half": r.prev_h1, f"{season_prev} 2nd half": r.prev_h2,
                f"{season_now} 1st half": r.cur_h1, f"{season_now} 2nd half": r.cur_h2,
                "This season's 2nd-half drop-off": VERDICT_MARK[r.verdict]} for r in rows])
            st.dataframe(table, hide_index=True, width="stretch", height=38 + 35 * len(table))
            st.caption("The last column compares how much each stat changes from 1st to 2nd half this season with "
                       "the same change last season. Opponent free throws include late fouls by a team already "
                       "behind.")
        section_n += 1

    # ---------------------------------------------------------- 5. what if
    if hl and reasons:
        st.header(f"{section_n}. What if they fixed it?", divider="gray", anchor="whatif")
        st.write("Switch a problem back to last season's level and see what their second halves would look like.")
        fixable = [r for r in hurts if r.prev_per_game - r.per_game >= 0.1]
        left, right = st.columns([1, 1.2], gap="large")
        with left:
            fixed = {r.factor for r in fixable
                     if st.toggle(f"{r.name} back to last season ({signed(r.prev_per_game)} instead of "
                                  f"{signed(r.per_game)} per game)", key=f"fix_{r.factor}")}
            g_pts = res["H3"]["key"]["excess_pts"] / max(hl["games"], 1) if "H3" in res else 0.0
            fix_g = st.toggle(f"Garland's turnovers back to his last-season rate (about {signed(g_pts)} per game)",
                              key="fix_garland", disabled="turnovers" in fixed,
                              help="Already included if you fix all turnovers.")
        new_h2 = S.what_if(hl["h2_per_game"], reasons, fixed, g_pts, fix_g)
        with right:
            st.metric("Second halves, per game", signed(new_h2), border=True,
                      delta=f"{new_h2 - hl['h2_per_game']:+.1f} vs today" if new_h2 != hl["h2_per_game"] else None)
            fig = go.Figure(go.Bar(
                y=["Today", "With your fixes", "1st halves (for reference)"],
                x=[hl["h2_per_game"], new_h2, hl["h1_per_game"]], orientation="h",
                marker_color=[BAD if hl["h2_per_game"] < 0 else GOOD, BAD if new_h2 < 0 else GOOD, GRAY],
                text=[signed(v) for v in (hl["h2_per_game"], new_h2, hl["h1_per_game"])], textposition="outside",
                hovertemplate="%{y}: %{x:+.1f} per game<extra></extra>"))
            fig.add_vline(x=0, line=dict(color=GRAY, width=1))
            lim = max(abs(hl["h2_per_game"]), abs(new_h2), abs(hl["h1_per_game"]), 2) * 1.35
            fig.update_xaxes(range=[-lim, lim], title="Score margin per game")
            fig.update_yaxes(autorange="reversed")
            show(fig, 220, showlegend=False)
        st.caption("Bookkeeping, not a prediction: real games don't change one thing at a time.")
        section_n += 1

    # ---------------------------------------------------------- 6. every stat
    st.header(f"{section_n}. Every stat, before and after halftime", divider="gray", anchor="stats")
    with st.expander("Open the full stat sheet", icon=":material/table:"):
        sheet = D.full_stat_sheet(data.ranks, season_now)
        st.dataframe(sheet, hide_index=True, width="stretch", height=38 + 35 * len(sheet))
    st.caption("League ranks: 1st = best of 30 teams. The change rank compares each team's second half with its "
               "own first half. These values come from the league's own team stats, so every team is measured the "
               "same way; they can differ slightly from the box-score totals above. Data: stats.nba.com via the open-source nba_api package; not affiliated with the "
               "NBA or the Clippers. Full method: reports/findings.md.")

# =================================================================== games

TONE_COLOR = {"good": GOOD, "bad": BAD, "warn": FOCUS, "info": GRAY}


def _minus(text: str) -> str:
    """A hyphen before a digit becomes a true minus sign, so every number on the page looks the same."""
    return re.sub(r"(?<![\w])-(?=\d)", "−", text)


def answer_cards(cards: list[W.Verdict], link: bool = True, big_size: str | None = None) -> None:
    """Short-headline cards with one big number each; with ``link`` a click jumps to the card's section."""
    cards = [W.Verdict(c.title, _minus(c.big), _minus(c.small), c.tone, c.anchor, c.viz, c.image) for c in cards]
    big_style = f' style="font-size:{big_size}"' if big_size else ""
    html = "".join(
        (f'<a class="vcard" href="#{c.anchor}"' if link else '<div class="vcard"')
        + ">"
        + (f'<img class="v-img" src="{c.image}" alt="" loading="lazy">' if c.image else "")
        + (f'<div class="v-title">{c.title}</div>' if c.viz or c.image else
           f'<div class="v-title"><span class="v-dot" style="background:{TONE_COLOR.get(c.tone, GRAY)}"></span>'
           f'{c.title}</div>')
        + f'<div class="v-big"{big_style}>{c.big}</div>'
        + (f'<div class="v-mini">{G.mini(c.viz, c.tone)}</div>' if c.viz else "")
        + f'<div class="v-small">{c.small}</div>' + ("</a>" if link else "</div>")
        for c in cards)
    st.markdown(f'<div class="vcards">{html}</div>', unsafe_allow_html=True)


def section(title: str, anchor: str, takeaway: str = "", badge: tuple[str, str] | None = None) -> None:
    st.header(title, divider="gray", anchor=anchor)
    if badge:
        st.badge(badge[0], color=badge[1])
    if takeaway:
        st.markdown(f'<div class="takeaway">{takeaway}</div>', unsafe_allow_html=True)


def legend_only(fig: go.Figure, items: list[tuple[str, str, str]]) -> None:
    """Legend entries that don't depend on the color of a trace's first point."""
    for name, color, symbol in items:
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=name,
                                 marker=dict(color=color, size=11, symbol=symbol)))


def page_whatnow() -> None:
    n_games = int(meta.get("games", 0))
    n = max(n_games, 1)
    page_top("What needs to happen now",
             f"What the numbers point to after {n_games} games. Click any card to see the chart behind it. "
             "Evidence, not a coaching plan.")
    res = {r["id"]: r for r in data.analysis.get("results", [])}
    lg = data.ranks[(data.ranks["season"] == season_now) & (data.ranks["split"] == "H2")
                    & (data.ranks["metric"] == "efg_pct")]["league_mean"]
    league_h2 = float(lg.iloc[0]) if len(lg) else float("nan")
    team = settings.team_abbreviation
    sd = W.shooting_drop(data.player_box, season_now, team, league_h2)
    sf = W.starting_five(data.player_box, data.lineups, season_now, team)
    bl = W.ball_load(data.player_box, data.lineups, res["H3"]["key"], season_now, team, focus, n) if "H3" in res else None
    cands = W.candidates(data.history, data.player_box, season_now, team, settings.watch_players, league_h2)
    rot = W.rotation(data.player_box, data.onoff, data.bio, season_now, team) if len(data.onoff) else None
    tracked = len(data.matchups) and len(data.bio)
    dfn = W.defenders(data.matchups, data.hustle, data.bio, season_now, team) if tracked else None
    gl = W.glass(data.game_halves, season_now, season_prev)
    ids = data.player_box.loc[data.player_box["player_name"] == focus, "player_id"]
    pr = (W.pressure(data.matchups, data.turnovers, data.bio, season_now, season_prev, team, settings.team_id,
                     int(ids.iloc[0])) if tracked and len(ids) and "stealer_id" in data.turnovers else None)
    chem = W.chemistry(data.player_box, data.lineups, season_now, season_prev, team)

    answer_cards(W.verdicts(sf, bl, cands, sd, focus, rot, dfn, gl, pr, n_games))
    with st.expander("Read it as text instead"):
        st.markdown("\n".join(f"- {line}" for line in W.summary_lines(sf, bl, cands, sd, focus, rot, dfn, gl, pr)),
                    unsafe_allow_html=True)

    # ------------------------------------------------------------ starting five
    if sf is not None:
        rest = sf.rest_h2_pm / n
        section("Does the starting five need to change?", "starters",
                f"No. The starters hold even after halftime; the other lineups lose "
                f"<b>{abs(rest):.1f} points a game</b>." if sf.not_the_problem and rest < 0 else
                f"Starters: {signed(sf.h2_pm / n)} a game after halftime; everyone else {signed(rest)}.",
                ("Keep it", "blue") if sf.not_the_problem else ("Maybe change it", "orange"))
        with st.expander("Details"):
            st.markdown(f"- Starters: {', '.join(sf.names)}; {sf.games_together} starts together.\n"
                        f"- They've played {sf.h2_min:.0f} second-half minutes together; other lineups "
                        f"{sf.rest_h2_min:.0f}.\n"
                        f"- Per 100 trips down the floor: starters {signed(sf.h2_net)} after halftime "
                        f"({signed(sf.h1_net)} before it), everyone else {signed(sf.rest_h2_net)}.\n"
                        "- Lineup numbers swing a lot on a few possessions; treat this as a direction.")

    # ------------------------------------------------------------ minutes
    if rot is not None and len(rot):
        r = rot[rot["h2_min_on"] > 0].sort_values("h2_mpg").copy()
        losing = int((r["h2_pm_pg"] < 0).sum())
        earned = list(r.loc[r["earned"], "player_name"].map(lambda x: x.split(" ", 1)[-1]))
        section("Who plays after halftime, and how it goes", "minutes",
                f"The Clippers lose the second half with <b>{losing} of {len(r)}</b> regulars on the floor."
                + (f" {' and '.join(earned)} {'is' if len(earned) == 1 else 'are'} the exception." if earned else ""))
        tag = np.where(r["struggling"], BAD, np.where(r["earned"], GOOD, MUTED_BAR))
        too_few = ~r["struggling"] & ~r["earned"] & (r["h2_min_on"] < W.EARNED_MIN)
        labels = [f"{x.split(' ', 1)[-1]}{' (R)' if rk else ''} · {m:.0f} min"
                  for x, rk, m in zip(r["player_name"], r["rookie"], r["h2_mpg"])]
        fig = go.Figure(go.Bar(
            y=labels, x=r["h2_pm_pg"], orientation="h", showlegend=False,
            marker=dict(color=tag, pattern=dict(shape=np.where(too_few, "/", ""), fgcolor=INK_MARK, solidity=0.25)),
            text=[signed(v) for v in r["h2_pm_pg"]], textposition="outside",
            customdata=np.column_stack([r["player_name"], r["h2_mpg"], r["q4_mpg"], r["gp"]]),
            hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]:.1f} second-half min a game "
                          "(%{customdata[2]:.1f} in the 4th)<br>Clippers %{x:+.1f} a game with him on after "
                          "halftime<extra></extra>"))
        legend_only(fig, [("Fewer late minutes", BAD, "square"), ("Earned more minutes", GOOD, "square"),
                          ("Keep as is", MUTED_BAR, "square")])
        if too_few.any():
            fig.add_trace(go.Bar(x=[None], y=[None], orientation="h", name="Too few minutes to tell",
                                 marker=dict(color=MUTED_BAR, pattern=dict(shape="/", fgcolor=INK_MARK,
                                                                           solidity=0.25))))
        lim = float(r["h2_pm_pg"].abs().max()) * 1.35
        fig.add_vline(x=0, line=dict(color=GRAY, width=1))
        fig.update_xaxes(range=[-lim, lim], title="Points a game with him on the floor after halftime")
        show(fig, 110 + 34 * len(r))
        st.caption("Sorted by second-half minutes a game (most at the top). (R) = rookie. Blue = the team does much "
                   "better with him on than off, even if the bar is small; red = the team falls well behind with him "
                   "on. Open Details for the exact rules.")
        with st.expander("Details and the full minutes table"):
            st.markdown(f"- **Fewer late minutes** = the team is {abs(W.STRUGGLE_NET):.0f}+ points per 100 trips "
                        f"behind with him on after halftime, over {W.STRUGGLE_MIN}+ minutes.\n"
                        f"- **Earned more minutes** = the team is {W.EARNED_ON_OFF:.0f}+ points per 100 trips better "
                        f"with him on than off, over {W.EARNED_MIN}+ minutes, while he plays under {W.EARNED_MPG} a game.\n"
                        "- Small samples swing on a few possessions, and on-court numbers depend on who else is "
                        "out there.")
            st.dataframe(
                rot[["player_name", "rookie", "gp", "mpg", "h2_mpg", "q4_mpg", "h2_pm_pg", "h2_on_off", "h2_reb36"]],
                hide_index=True, width="stretch",
                column_config={
                    "player_name": "Player", "rookie": st.column_config.CheckboxColumn("Rookie"),
                    "gp": "Games", "mpg": st.column_config.NumberColumn("Min/game", format="%.1f"),
                    "h2_mpg": st.column_config.NumberColumn("2nd-half min/game", format="%.1f"),
                    "q4_mpg": st.column_config.NumberColumn("4th-qtr min/game", format="%.1f"),
                    "h2_pm_pg": st.column_config.NumberColumn("2nd-half +/- a game", format="%+.1f"),
                    "h2_on_off": st.column_config.NumberColumn("On minus off (per 100)", format="%+.1f"),
                    "h2_reb36": st.column_config.NumberColumn("2nd-half rebounds/36", format="%.1f"),
                })

    # ------------------------------------------------------------ Garland
    if bl is not None:
        ratio = bl.tov_pct_h2 / bl.prior_tov_pct_h2 if bl.prior_tov_pct_h2 else float("nan")
        cause = ""
        if pr is not None:
            cause = (" The cause looks like traps." if pr.trapped else
                     " The cause looks like bigger defenders, not double teams." if pr.length_problem else "")
        section(f"Should {focus_last} handle the ball less?", "garland",
                f"<b>{bl.tov_pct_h2:.0%}</b> of his second-half plays end in a turnover, {ratio:.1f}x his usual "
                f"rate.{cause}",
                ("Yes, for now", "orange") if bl.should_handle_less else ("Not really", "blue"))
        left, right = st.columns([1.3, 1], gap="large")
        with left:
            safest = bl.safest.head(4)
            names = [f"<b>{focus_last}</b>"] + [x.split(" ", 1)[-1] for x in safest["player_name"]]
            vals = [bl.tov_pct_h2] + list(safest["tov_pct"])
            fig = go.Figure(go.Bar(
                y=names, x=vals, orientation="h", marker_color=[FOCUS] + [MUTED_BAR] * len(safest),
                text=[f"1 in {round(1 / v)}" if v else "" for v in vals], textposition="outside",
                hovertemplate="%{y}: %{x:.1%} of second-half plays end in a turnover<extra></extra>"))
            fig.add_vline(x=bl.prior_tov_pct_h2, line=dict(color=INK_MARK, width=1, dash="dash"),
                          annotation_text=f"{focus_last} last season", annotation_position="top")
            fig.update_xaxes(tickformat=".0%", range=[0, max(vals) * 1.35],
                             title="Second-half plays that end in a turnover")
            fig.update_yaxes(autorange="reversed")
            st.markdown("**How often his plays end in a turnover**")
            show(fig, 250, showlegend=False)
        with right:
            if pr is not None and pr.tov:
                st.markdown("**Is he being double-teamed into them?**")
                picture(G.turnover_strip(pr.steals - pr.help, pr.help, pr.tov - pr.steals))
                st.caption(f"Only {pr.help_share:.0%} are stolen by a second defender, the way a double team "
                           f"usually ends (teammates: {pr.mates_help_share:.0%}). Double teams aren't logged "
                           "publicly; this is the closest proxy.")
        if pr is not None:
            st.markdown(f"**Bigger defenders are the new problem.** How often {focus_last} loses the ball, by the "
                        "height of the player guarding him:")
            me = data.bio[data.bio["player_name"] == focus].sort_values("season").tail(1)
            ht = float(me["height_in"].iloc[0]) if len(me) else 73.0
            wt = float(me["weight_lb"].iloc[0]) if len(me) else 190.0
            picture(G.height_matchup(focus_last, ht, wt, 100 / pr.tov100_long, 100 / pr.tov100_short,
                                     100 / pr.prior_tov100_long, 100 / pr.prior_tov100_short, W.LONG_IN),
                    max_width="820px")
        with st.expander("Details"):
            st.markdown(
                f"- He handles {bl.play_share_h2:.0%} of the team's second-half plays and makes "
                f"{bl.tov_share_h2:.0%} of its turnovers. Getting back to his usual rate is worth about "
                f"{signed(bl.pts_per_game)} points a game.\n"
                f"- Lineups with him: {signed(bl.net_with)} per 100 trips ({bl.min_with:.0f} min); without him: "
                f"{signed(bl.net_without)} ({bl.min_without:.0f} min). Mixed up with who he plays alongside, so a "
                "flag, not proof.")
            if pr is not None:
                types = {"bad_pass": "bad passes", "lost_ball": "lost balls", "violation": "travels and violations",
                         "offensive_foul": "offensive fouls", "shot_clock": "shot-clock violations"}
                mix = ", ".join(f"{k_} {types.get(t_, t_)}" for t_, k_ in
                                sorted(pr.h2_types.items(), key=lambda kv: -kv[1]))
                f_ = pr.forcers.head(5)
                st.markdown(
                    f"- His teammates show no size gap ({pr.mates_tov100_long:.1f} vs {pr.mates_tov100_short:.1f} "
                    f"per 100), so it's specific to him.\n"
                    f"- After halftime: {mix}.\n"
                    "- Defenders credited with the most: " + "; ".join(
                        f"{x.def_player} ({x.def_team}, {W._ft_in(x.height_in) if x.height_in == x.height_in else '?'}) "
                        f"{x.tov:.0f}" for x in f_.itertuples()) + ".\n"
                    f"- Tracking covers {pr.games} of {n_games} games (it's published a few hours after each game).")

    # ------------------------------------------------------------ defense
    section("Defense, rebounding and size", "defense",
            f"After halftime opponents get back <b>{gl.opp_oreb_h2:.0%}</b> of their misses (before it "
            f"{gl.opp_oreb_h1:.0%})." + (
                " And some of the rotation is giving up a lot of size." if dfn is not None and
                dfn.table["gives_up_size"].any() else ""),
            ("Rebounding got worse", "orange") if gl.worse else ("Rebounding holds up", "blue"))
    left, right = st.columns([1, 1.2], gap="large")
    with left:
        st.markdown("**Of every 100 shots the other team misses, how many they grab back**")
        picture(G.missed_shots(gl.opp_oreb_h1, gl.opp_oreb_h2, gl.prior_opp_oreb_h2))
    with right:
        small = dfn.table[dfn.table["gives_up_size"]] if dfn is not None and len(dfn.table) else None
        if small is not None and len(small):
            q4 = rot.set_index("player_id")["q4_mpg"] if rot is not None else pd.Series(dtype=float)
            r_ = small.assign(q4=small["player_id"].map(q4).fillna(0)).sort_values("q4", ascending=False).iloc[0]
            who = r_["player_name"].split(" ", 1)[-1]
            st.markdown(f"**{who} gives up size{' as a rookie' if r_['rookie'] else ''}**")
            picture(G.size_matchup(who, r_["own_ht"], r_["own_wt"], r_["opp_ht"], r_["opp_wt"], r_["fg_pct"],
                                   dfn.team_fg))
    left = right = st.container()  # one column: the shooting chart, then the size table
    with left:
        if dfn is not None and len(dfn.table):
            t = dfn.table.sort_values("fg_pct", ascending=True)
            st.markdown("**How well opponents shoot when he's the defender**")
            fig = go.Figure(go.Bar(
                y=t["player_name"].str.split(" ", n=1).str[-1] + np.where(t["rookie"], " (R)", ""), x=t["fg_pct"],
                orientation="h", marker_color=[BAD if a_ else MUTED_BAR for a_ in t["attacked"]],
                text=[f"{v:.0%}" for v in t["fg_pct"]], textposition="outside",
                customdata=np.column_stack([t["player_name"], t["fga"]]),
                hovertemplate="%{customdata[0]}: opponents shoot %{x:.1%} on %{customdata[1]:.0f} shots"
                              "<extra></extra>"))
            fig.add_vline(x=dfn.team_fg, line=dict(color=INK_MARK, width=1, dash="dash"),
                          annotation_text=f"team {dfn.team_fg:.0%}", annotation_position="top")
            fig.update_xaxes(tickformat=".0%", range=[0, float(t["fg_pct"].max()) * 1.25])
            show(fig, 90 + 30 * len(t), showlegend=False)
            st.caption(f"Red = opponents shoot {W.ATTACKED_FG:.0%}+ better on him than on the team. (R) = rookie.")
    with right:
        if dfn is not None and len(dfn.table):
            t = dfn.table.sort_values("wt_gap")
            with st.expander("How big are the players each one guards?"):
                fig = go.Figure()
                for r_ in t.itertuples():
                    fig.add_trace(go.Scatter(x=[r_.own_wt, r_.opp_wt], y=[r_.player_name.split(" ", 1)[-1]] * 2,
                                             mode="lines", hoverinfo="skip", showlegend=False,
                                             line=dict(color=FOCUS if r_.gives_up_size else LIGHT, width=4)))
                y = [x.split(" ", 1)[-1] for x in t["player_name"]]
                fig.add_trace(go.Scatter(x=t["own_wt"], y=y, mode="markers", name="His weight",
                                         marker=dict(color=H1C, size=12, line=dict(color=BG, width=1.5)),
                                         hovertemplate="%{y}: %{x:.0f} lb<extra>his weight</extra>"))
                fig.add_trace(go.Scatter(x=t["opp_wt"], y=y, mode="markers", name="Who he guards (average)",
                                         marker=dict(color=FOCUS, size=12, line=dict(color=BG, width=1.5)),
                                         hovertemplate="%{y} guards players averaging %{x:.0f} lb<extra></extra>"))
                fig.update_xaxes(title="Weight (lb)")
                show(fig, 90 + 30 * len(t))
                st.caption("Orange line = guards players 25+ lb heavier than himself.")
    if dfn is not None and len(dfn.table):
        with st.expander("Details and effort stats (contests, deflections, box-outs)"):
            for r_ in dfn.table[dfn.table["gives_up_size"]].itertuples():
                st.markdown(f"- **{r_.player_name}** ({W._ft_in(r_.own_ht)}, {r_.own_wt:.0f} lb"
                            f"{', rookie' if r_.rookie else ''}): guards players averaging {r_.opp_wt:.0f} lb; "
                            f"{r_.def_box_outs:.0f} defensive box-outs in {r_.hustle_min:.0f} minutes.")
            st.dataframe(
                dfn.table[["player_name", "rookie", "hustle_min", "contested_shots_36", "deflections_36",
                           "def_box_outs_36", "fg_pct", "shooting_fouls"]].sort_values("hustle_min", ascending=False),
                hide_index=True, width="stretch",
                column_config={
                    "player_name": "Player", "rookie": st.column_config.CheckboxColumn("Rookie"),
                    "hustle_min": st.column_config.NumberColumn("Minutes tracked", format="%.0f"),
                    "contested_shots_36": st.column_config.NumberColumn("Contests/36", format="%.1f"),
                    "deflections_36": st.column_config.NumberColumn("Deflections/36", format="%.1f"),
                    "def_box_outs_36": st.column_config.NumberColumn("Def. box-outs/36", format="%.1f"),
                    "fg_pct": st.column_config.NumberColumn("Opp. FG%", format="percent"),
                    "shooting_fouls": st.column_config.NumberColumn("Shooting fouls", format="%.0f"),
                })
            st.caption(f"Player tracking, whole games, {dfn.games} games so far. Rebound shares: "
                       f"{gl.opp_2nd_h2:.1f} opponent second-chance points a game after halftime "
                       f"({gl.opp_2nd_h1:.1f} before it).")

    # ------------------------------------------------------------ chemistry
    p = chem.pairs.sort_values("pm")
    best = p.iloc[-1] if len(p) else None
    section("Chemistry: which duos work?", "chemistry",
            f"<b>{chem.continuity:.0%}</b> of minutes go to players who were here last season."
            + (f" Best duo after halftime: <b>{best['a'].split(' ', 1)[-1]} + {best['b'].split(' ', 1)[-1]}</b>."
               if best is not None else ""))
    left, right = st.columns([1, 2.2], gap="large")
    with left:
        st.markdown("**How much of the team is new**")
        answer_cards([W.Verdict("Minutes from players who were here last season", f"{chem.continuity:.0%}",
                                f"New this season: {', '.join(n.split(' ', 1)[-1] for n in chem.new[:5]) or 'nobody'}.",
                                "info", "chemistry", ("share", chem.continuity, None))], link=False)
    with right:
        if len(p):
            show_p = pd.concat([p.head(5), p.tail(5)]).drop_duplicates(subset=["a_id", "b_id"]).sort_values("pm")
            labels = [f"{a.split(' ', 1)[-1]} + {b.split(' ', 1)[-1]}{' ✦' if new else ''}"
                      for a, b, new in zip(show_p["a"], show_p["b"], show_p["new_pair"])]
            fig = go.Figure(go.Bar(
                y=labels, x=show_p["pm"], orientation="h",
                marker_color=[GOOD if v > 0 else BAD if v < 0 else MUTED_BAR for v in show_p["pm"]],
                text=[f"{signed(v, 0)} in {m:.0f} min" for v, m in zip(show_p["pm"], show_p["min"])],
                textposition="outside",
                hovertemplate="%{y}: Clippers %{x:+.0f} after halftime with both on<extra></extra>"))
            lim = float(show_p["pm"].abs().max()) * 1.6
            fig.add_vline(x=0, line=dict(color=GRAY, width=1))
            fig.update_xaxes(range=[-lim, lim], title="Clippers' score with both on the floor after halftime")
            st.markdown("**The best and worst duos after halftime**")
            show(fig, 90 + 32 * len(show_p), showlegend=False)
            st.caption("✦ = never played together last season. Duos with 25+ second-half minutes together.")
    with st.expander("Details"):
        st.markdown(f"- **New this season:** {', '.join(chem.new) or 'nobody'}\n"
                    f"- **Back from last season:** {', '.join(chem.returning)}\n"
                    f"- {int(p['new_pair'].sum())} of {len(p)} regular duos never played together last season.\n"
                    "- Chemistry itself can't be measured. Duos overlap, so one player's bad nights show up in all "
                    "of his duos.")

    # ------------------------------------------------------------ returning players
    shooters = [c for c in cands if c.fg3a >= W.THREE_MIN_ATT]
    section("Will the players coming back help?", "returning",
            ("The returning shooters hit " + ", ".join(f"<b>{c.fg3_pct:.0%}</b>" for c in shooters)
             + f" of their threes after halftime. The Clippers hit {sd.team_fg3_h2:.0%} now.") if shooters else "")
    left, right = st.columns(2, gap="large")
    with left:
        if shooters:
            xs = [c.name.split(" ", 1)[-1] for c in shooters] + ["Clippers now"]
            ys = [c.fg3_pct for c in shooters] + [sd.team_fg3_h2]
            fig = go.Figure(go.Bar(x=xs, y=ys, marker_color=[FOCUS] * len(shooters) + [MUTED_BAR],
                                   text=[f"{v:.0%}" for v in ys], textposition="outside",
                                   hovertemplate="%{x}: %{y:.1%} from three after halftime<extra></extra>"))
            fig.update_yaxes(tickformat=".0%", range=[0, max(ys) * 1.3])
            st.markdown("**Three-point shooting after halftime**")
            show(fig, 300, showlegend=False)
    with right:
        bigs = [c for c in cands if "rebounding" in c.helps]
        if bigs and rot is not None:
            cur = rot[(rot["h2_mpg"] * rot["gp"] >= 40) & ~rot["player_name"].isin([c.name for c in bigs])]
            top = cur.sort_values("h2_reb36", ascending=False).head(1)
            xs = [c.name.split(" ", 1)[-1] for c in bigs] + [f"{x.split(' ', 1)[-1]} (best now)"
                                                            for x in top["player_name"]]
            ys = [c.reb36 for c in bigs] + list(top["h2_reb36"])
            fig = go.Figure(go.Bar(x=xs, y=ys, marker_color=[FOCUS] * len(bigs) + [MUTED_BAR] * len(top),
                                   text=[f"{v:.1f}" for v in ys], textposition="outside",
                                   hovertemplate="%{x}: %{y:.1f} rebounds per 36 minutes after halftime"
                                                 "<extra></extra>"))
            fig.update_yaxes(range=[0, max(ys) * 1.3])
            st.markdown("**Rebounds per 36 minutes after halftime**")
            show(fig, 300, showlegend=False)
    help_color = {"shooting": "green", "threes": "green", "ball security": "blue", "rebounding": "violet"}
    cols = st.columns(len(cands) or 1)
    for col, c in zip(cols, cands):
        with col.container(border=True, height="stretch"):
            st.markdown(f'<div class="big-q">{c.name}</div>', unsafe_allow_html=True)
            st.caption(("Out so far" if c.this_season_minutes == 0
                        else f"Playing: {c.this_season_minutes:.0f} min so far")
                       + (f" · {c.mpg:.0f} min a game ({c.sample.split(',')[0]})" if c.mpg == c.mpg else ""))
            for h in c.helps:
                st.badge(h.capitalize(), color=help_color.get(h, "gray"))
            if c.small_sample:
                st.badge("Small sample", color="gray")
    with st.expander("Details"):
        st.markdown("Numbers for players who haven't played much this season come from their most recent season with "
                    f"{W.SMALL_SAMPLE_MIN}+ second-half minutes, on whatever team they played for.")
        st.dataframe(pd.DataFrame([{
            "Player": c.name, "Sample": c.sample, "2nd-half min": c.h2_minutes, "Shooting (eFG%)": c.efg,
            "3P%": c.fg3_pct, "3PA": c.fg3a, "Turnover rate": c.tov_pct, "Rebounds/36": c.reb36} for c in cands]),
            hide_index=True, width="stretch",
            column_config={"2nd-half min": st.column_config.NumberColumn(format="%.0f"),
                           "Shooting (eFG%)": st.column_config.NumberColumn(format="percent"),
                           "3P%": st.column_config.NumberColumn(format="percent"),
                           "3PA": st.column_config.NumberColumn(format="%.0f"),
                           "Turnover rate": st.column_config.NumberColumn(format="percent"),
                           "Rebounds/36": st.column_config.NumberColumn(format="%.1f")})

    # ------------------------------------------------------------ who stopped shooting
    section("Who stopped shooting after halftime?", "shooting",
            f"<b>{sd.n_worse} of {sd.n_players}</b> regulars shoot worse after halftime.")
    t = sd.table.sort_values("change")
    fig = go.Figure()
    for _, r_ in t.iterrows():
        fig.add_trace(go.Scatter(x=[r_["efg_h1"], r_["efg_h2"]], y=[r_["player_name"]] * 2, mode="lines",
                                 line=dict(color=BAD if r_["change"] < 0 else LIGHT, width=3), hoverinfo="skip",
                                 showlegend=False))
    for col, color, name in (("efg_h1", H1C, "Before halftime"), ("efg_h2", H2C, "After halftime")):
        fig.add_trace(go.Scatter(
            x=t[col], y=t["player_name"], mode="markers", name=name,
            marker=dict(color=color, size=13, line=dict(color=BG, width=1.5)), customdata=t[f"fga_{col[-2:]}"],
            hovertemplate="%{y}: %{x:.1%} on %{customdata:.0f} shots<extra>" + name + "</extra>"))
    fig.add_vline(x=league_h2, line=dict(color=INK_MARK, width=1, dash="dash"), annotation_text="league average",
                  annotation_position="top")
    fig.update_xaxes(tickformat=".0%", title="Shooting accuracy (eFG%, which counts threes as worth more)")
    show(fig, 90 + 40 * len(t))
    with st.expander("Details"):
        st.markdown("The biggest drops come from players who were unusually hot before halftime, so some of this "
                    "should even out. Players with 40+ minutes in each half.")

    # ------------------------------------------------------------ what would change this
    with st.expander("What would change this advice", icon=":material/update:"):
        watch = []
        if "H3" in res:
            k3 = res["H3"]["key"]
            watch.append(f"**{focus_last} becomes the main cause** if his extra turnovers grow from "
                         f"{k3['share_of_swing']:.0%} to {DRIVES:.0%} of the second-half drop.")
        if "H1" in res:
            k1 = res["H1"]["key"]
            watch.append("**The collapse is confirmed (not bad luck)** once the 90% range for the drop stops "
                         f"including zero. It's {signed(k1['boot_lo'])} to {signed(k1['boot_hi'])} today.")
        watch.append(f"**The shooting problem is fading** if second-half shooting climbs back above the league "
                     f"average ({league_h2:.1%}). It's {sd.team_h2:.1%} today.")
        if pr is not None:
            watch.append(f"**{focus_last} has adjusted to bigger defenders** once his turnovers against them drop "
                         f"under {W.LENGTH_RATIO}x his rate against shorter ones. It's "
                         f"{pr.tov100_long / pr.tov100_short:.1f}x today.")
        watch.append("**The rebounding is fixed** when opponents get back under "
                     f"{gl.prior_opp_oreb_h2 + W.GLASS_WORSE:.1%} of their misses after halftime. It's "
                     f"{gl.opp_oreb_h2:.1%} today.")
        st.markdown("\n".join(f"- {w}" for w in watch))
        st.caption("Everything on this page reruns with `courtside all` after each game. The rules behind each "
                   "verdict are listed at the top of src/courtside/viz/whatnow.py.")


def _step_game(key: str, labels: list[str], step: int) -> None:
    """Previous / next game buttons."""
    cur = st.session_state.get(key, labels[-1])
    i = labels.index(cur) if cur in labels else len(labels) - 1
    st.session_state[key] = labels[max(0, min(len(labels) - 1, i + step))]


def _set_game(key: str, label: str) -> None:
    st.session_state[key] = label


def _upcoming(season: str) -> tuple[pd.DataFrame, list, pd.DataFrame, float | None]:
    """Games left this season, their previews, the team table, and the season's second-half shooting cost."""
    if season != season_now or data.schedule.empty or data.league_periods.empty:
        return pd.DataFrame(), [], pd.DataFrame(), None
    teams = P.blended(P.team_table(data.league_periods, season_now), P.team_table(data.league_periods, season_prev))
    att = data.attribution.get(season_now)
    shooting = float(att.set_index("factor").loc["shooting", "per_game"]) if att is not None else None
    tid = settings.team_id
    up = data.schedule[((data.schedule["home_id"] == tid) | (data.schedule["away_id"] == tid))
                       & (data.schedule["status"] < 3)].sort_values("tipoff_utc").reset_index(drop=True)
    return up, [P.preview(g, teams, tid, shooting) for _, g in up.iterrows()], teams, shooting


def page_replay() -> None:
    page_top("Games", "Every game this season: the report for games played, the game plan for games coming up. "
             "Updates after each game.")
    leads = data.game_leads.set_index("game_id")
    g_all = data.games.set_index("game_id")
    season = st.session_state.get("game_season", data.seasons[0])
    if season not in data.seasons:
        season = data.seasons[0]
    options = D.game_options(data.game_leads, season).iloc[::-1].reset_index(drop=True)  # oldest first
    strip = leads.loc[options["game_id"]].reset_index()
    gi = g_all.reindex(strip["game_id"])
    entries = [{"kind": "past", "gid": g, "date": pd.Timestamp(d), "home": bool(h), "opp": o, "opp_id": int(oid),
                "status": f"{r} {int(tp)}-{int(op)}", "won": r == "W",
                "label": f"{pd.Timestamp(d):%b} {pd.Timestamp(d).day} · {'vs' if h else '@'} {o} · {r} {int(tp)}-{int(op)}"}
               for g, d, h, o, oid, r, tp, op in zip(strip["game_id"], gi["game_date"], gi["is_home"],
                                                     gi["opponent_abbreviation"], gi["opponent_id"], gi["result"],
                                                     gi["team_pts"], gi["opp_pts"])]
    up, previews, teams, shooting = _upcoming(season)
    for j, (pv, (_, g)) in enumerate(zip(previews, up.iterrows())):
        t = g["tipoff_utc"].tz_convert("America/Los_Angeles")
        entries.append({"kind": "future", "idx": j, "date": t, "home": pv.home, "opp": pv.opponent,
                        "opp_id": int(g["away_id"] if pv.home else g["home_id"]), "status": f"{pv.chance:.0%} to win",
                        "won": None, "label": f"{t:%b} {t.day} · {'vs' if pv.home else '@'} {pv.opponent} · "
                                              f"{'next game, ' if j == 0 else ''}{pv.chance:.0%} to win"})
    labels = [e["label"] for e in entries]
    n_past = len(strip)
    xs = pd.to_datetime(strip["game_date"]).dt.strftime("%b %d").str.replace(" 0", " ") + " " + strip[
        "opponent_abbreviation"]
    by_x = dict(zip(xs, labels[:n_past]))
    pick_key = f"pick_{season}"
    st.session_state.setdefault(pick_key, labels[n_past - 1] if n_past else labels[0])  # the latest result
    jump = st.session_state.pop("jump_game", None)
    if jump in set(strip["game_id"]):
        st.session_state[pick_key] = labels[list(strip["game_id"]).index(jump)]
    x = clicked(st.session_state.get(f"strip_{season}"), "x", f"strip_click_{season}")  # a click on the strip
    if x in by_x:
        st.session_state[pick_key] = by_x[x]
    if st.session_state[pick_key] not in labels:
        st.session_state[pick_key] = labels[n_past - 1] if n_past else labels[0]

    # ---- the game picker: pinned to the top of the page while scrolling
    with st.container(key="gamebar"):
        c_season, c_prev, c_pick, c_next, c_logo = st.columns([1.2, 0.45, 4, 0.45, 0.5], vertical_alignment="bottom",
                                                                gap="small")
        c_season.selectbox("Season", data.seasons, key="game_season")
        c_prev.button("◀", key="prev_game", on_click=_step_game, args=(pick_key, labels, -1), help="Previous game",
                      width="stretch")
        pick = c_pick.selectbox("Game", labels, key=pick_key)
        c_next.button("▶", key="next_game", on_click=_step_game, args=(pick_key, labels, 1), help="Next game",
                      width="stretch")
        idx = labels.index(pick)
        entry = entries[idx]
        c_logo.image(D.logo_url(entry["opp_id"]), width=46)

    # ---- the schedule around this game: played games and the ones coming up
    lo = max(0, min(idx - 3, len(entries) - 7))
    window = entries[lo:lo + 7]
    cols = st.columns(len(window), gap="small")
    for k, (col, e) in enumerate(zip(cols, window), start=lo):
        sel = k == idx
        with col.container(border=True, key=f"gcard_sel" if sel else f"gcard_{k}"):
            st.image(D.logo_url(e["opp_id"]), width=40)
            color = GOOD if e["won"] else BAD if e["won"] is False else INK_MARK
            st.markdown(f'<div class="gc"><div class="gc-date">{e["date"]:%a %b} {e["date"].day}</div>'
                        f'<div class="gc-opp">{"vs" if e["home"] else "@"} {e["opp"]}</div>'
                        f'<div class="gc-status" style="color:{color}">{e["status"]}</div></div>',
                        unsafe_allow_html=True)
            st.button("Showing" if sel else ("Report" if e["kind"] == "past" else "Game plan"), key=f"open_{k}",
                      on_click=_set_game, args=(pick_key, e["label"]), disabled=sel, width="stretch")

    if entry["kind"] == "future":
        _game_plan(up.iloc[entry["idx"]], previews[entry["idx"]], teams, shooting)
        return

    gid = entry["gid"]
    row = leads.loc[gid]
    where = "Home, Intuit Dome" if bool(g_all.loc[gid, "is_home"]) else "Away"
    st.markdown(f'<div class="kicker">{pd.Timestamp(row["game_date"]):%A, %B} {pd.Timestamp(row["game_date"]).day} · '
                f'{where}</div>', unsafe_allow_html=True)

    with st.expander("Season at a glance: every game from halftime to the final (click one to open it)"):
        fig = go.Figure()
        for xx, h, f, blown in zip(xs, strip["halftime_margin"], strip["final_margin"],
                                   strip["lead_class"] == "blown_lead"):
            fig.add_trace(go.Scatter(x=[xx, xx], y=[h, f], mode="lines",
                                     line=dict(color=BAD if blown else LIGHT, width=3), hoverinfo="skip",
                                     showlegend=False))
        fig.add_trace(go.Scatter(x=xs, y=strip["halftime_margin"], mode="markers", name="At halftime",
                                 marker=dict(color=H1C, size=11, line=dict(color=BG, width=1.5)),
                                 hovertemplate="%{x}<br>At halftime: %{y:+d}<extra></extra>"))
        fig.add_trace(go.Scatter(x=xs, y=strip["final_margin"], mode="markers", name="Final",
                                 marker=dict(color=H2C, size=11, symbol="diamond", line=dict(color=BG, width=1.5)),
                                 hovertemplate="%{x}<br>Final: %{y:+d}<extra></extra>"))
        fig.add_hline(y=0, line=dict(color=GRAY, width=1))
        if len(strip) > 20:
            fig.update_xaxes(showticklabels=False)
        show(fig, 230, key=f"strip_{season}", select=True, yaxis_title="Clippers margin")
        st.caption("Halftime (●) to final (◆). Red = a blown early lead.")

    halves = data.game_halves[(data.game_halves["game_id"] == gid)
                              & data.game_halves["split"].astype(str).isin(["H1", "H2"])]
    att = D.game_attribution(data.game_halves, gid)
    mq = D.game_margin(data.margins, gid).groupby("period")[["team_score", "opp_score"]].max()
    team_q = mq["team_score"].diff().fillna(mq["team_score"]).astype(int).tolist()
    opp_q = mq["opp_score"].diff().fillna(mq["opp_score"]).astype(int).tolist()
    opp = row["opponent_abbreviation"]
    names = data.player_box.drop_duplicates("player_id").set_index("player_id")["player_name"].to_dict()
    lines = R.player_lines(data.player_box, gid, "LAC")
    opp_lines = R.player_lines(data.player_box, gid, opp)
    factors = R.vs_normal(data.game_halves, gid)
    seg = data.segments[data.segments["game_id"] == gid] if len(data.segments) else pd.DataFrame()
    game_units = R.units(seg, names)
    plays = data.plays[data.plays["game_id"] == gid] if len(data.plays) else pd.DataFrame()
    run_against, run_for = R.biggest_runs(data.margins, gid)
    if len(plays) and len(seg):
        run_against = R.describe_run(run_against, plays, seg, names) if run_against else None
        run_for = R.describe_run(run_for, plays, seg, names) if run_for else None
    game_def = R.defenders_for_game(data.matchups, gid, "LAC") if len(data.matchups) else pd.DataFrame()
    sbox = data.player_box[(data.player_box["season"] == season) & (data.player_box["team_abbreviation"] == "LAC")
                           & (data.player_box["split"].astype(str) == "FULL")].groupby("player_name")[
        ["minutes", "fga", "fta", "tov", "oreb", "dreb", "fgm", "fg3m"]].sum().reset_index()
    sd_ = W.defenders(data.matchups, data.hustle, data.bio, season, "LAC") if len(data.matchups) and len(
        data.bio) else None
    season_def = sd_.table if sd_ is not None else None
    lvs = R.levers(factors, lines, sbox, game_def, season_def, game_units) if len(factors) else []

    # ---- the result
    picture(G.scoreboard("LAC", opp, team_q, opp_q, row["final_margin"] > 0, team_logo=D.logo_url(settings.team_id),
                         opp_logo=D.logo_url(g_all.loc[gid, "opponent_id"])), max_width="980px")
    with st.container(border=True):
        tag = S.LEAD_TAG.get(row["lead_class"], "")
        if tag:
            st.badge(tag, color="red" if row["lead_class"] == "blown_lead" else "blue")
        st.markdown(f'<div class="story">{S.game_story(row, halves, att)}</div>', unsafe_allow_html=True)
        if len(factors):
            st.markdown(f'<div class="takeaway"><b>{R.verdict(factors, lvs)}</b></div>', unsafe_allow_html=True)

    # ---- the tiles: best, worst, the runs, turnovers
    best, worst = R.best_and_worst(lines)
    tiles = []
    for who, title, tone in ((best, "Best Clipper", "good"), (worst, "Toughest night", "bad")):
        if who is not None:
            tiles.append(W.Verdict(
                f"{title}: {who['player_name']}", f"{who['game_score']:.1f}",
                f"Game Score. {int(who['pts'])} pts, {int(who['reb'])} reb, {int(who['ast'])} ast, "
                f"{int(who['tov'])} TO in {who['minutes']:.0f} min; {signed(who['plus_minus'], 0)} on the floor.",
                tone, "players"))
    for run, title, tone in ((run_against, f"Biggest {opp} run", "bad"), (run_for, "Biggest Clippers run", "good")):
        if run is not None:
            who = ", ".join(n.split(" ", 1)[-1] for n in run.unit) if run.unit else "?"
            tiles.append(W.Verdict(title, run.label, f"{R.run_window(run)}. On the floor: {who}.", tone, "runs",
                                   ("diverge", (run.team_pts, -run.opp_pts), ("LAC", opp))))
    tiles.append(S.game_cards(row, halves, data.turnovers, focus_last)[-1])
    answer_cards(tiles)

    # ---- the replay, with the runs shaded
    st.header("How it played out", divider="gray", anchor="runs")
    m = D.game_margin(data.margins, gid)
    end = max(48.0, float(m["minute"].max()))
    blown = row["lead_class"] == "blown_lead"
    line_color = BAD if blown else H1C
    show_tov = st.toggle("Show Clippers turnovers on the chart", value=True,
                         help=f"Orange diamonds are {focus}'s turnovers; gray circles are everyone else's.")
    fig = go.Figure()
    for run, color in ((run_against, "rgba(230,103,103,0.16)"), (run_for, "rgba(57,135,229,0.16)")):
        if run is not None:
            fig.add_vrect(x0=run.start / 60, x1=run.end / 60, fillcolor=color, line_width=0, layer="below",
                          annotation_text=("LAC " if not run.against else f"{opp} ") + run.label,
                          annotation_position="top left", annotation_font_size=12)
    fig.add_vline(x=24, line=dict(color=GRAY, width=1, dash="dot"))
    fig.add_hline(y=0, line=dict(color=GRAY, width=1))
    custom = np.column_stack([m["team_score"], m["opp_score"], m["period"]])
    fig.add_trace(go.Scatter(x=m["minute"], y=m["margin"], mode="lines", line=dict(color=LIGHT, width=1.5, shape="hv"),
                             hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(
        x=m["minute"], y=m["margin"], mode="lines", line=dict(color=line_color, width=2.5, shape="hv"),
        customdata=custom, name="Clippers margin",
        hovertemplate="Minute %{x:.1f}<br>LAC %{customdata[0]} – " + opp + " %{customdata[1]}"
                      "<br>Margin %{y:+d}<extra></extra>"))
    if show_tov:
        tm = S.turnover_markers(data.turnovers, data.margins, gid, focus_last)
        for is_focus, name, symbol, color, size in ((False, "Other turnovers", "circle", GRAY, 9),
                                                    (True, f"{focus_last} turnovers", "diamond", FOCUS, 12)):
            sub = tm[tm["is_focus"] == is_focus]
            fig.add_trace(go.Scatter(
                x=sub["minute"], y=sub["margin"], mode="markers", name=name,
                marker=dict(symbol=symbol, color=color, size=size, line=dict(color=BG, width=1.2)),
                customdata=np.column_stack([sub["who"], sub["sub_type"], sub["kind"], sub["pts_off"]]),
                hovertemplate="<b>%{customdata[0]}</b>: %{customdata[1]}<br>%{customdata[2]}"
                              "<br>Opponent scored %{customdata[3]} off it<extra></extra>"))
    stops = S.replay_frames(m, step_seconds=60)
    frames = []
    for t in stops:
        cut = m[m["minute"] <= t]
        last = cut.iloc[-1] if len(cut) else m.iloc[0]
        frames.append(go.Frame(
            name=f"{t:.1f}", traces=[1],
            data=[go.Scatter(x=cut["minute"], y=cut["margin"], customdata=custom[: len(cut)])],
            layout=dict(title=dict(text=f"Minute {t:.0f} · LAC {int(last['team_score'])} – "
                                        f"{opp} {int(last['opp_score'])}"))))
    fig.frames = frames
    fig.update_layout(
        title=dict(text=f"Final · LAC {int(m['team_score'].iloc[-1])} – {opp} {int(m['opp_score'].iloc[-1])}",
                   x=0, font=dict(size=15)),
        updatemenus=[dict(type="buttons", showactive=False, x=1, xanchor="right", y=1.18, direction="left", buttons=[
            dict(label="▶ Replay", method="animate",
                 args=[None, dict(frame=dict(duration=25, redraw=True), transition=dict(duration=0),
                                  fromcurrent=False, mode="immediate")]),
            dict(label="❚❚", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
        ])],
    )
    n_ot = int(row["overtime_periods"])
    fig.update_xaxes(range=[0, end + 0.8], tickvals=[0, 12, 24, 36, 48] + [50.5 + 5 * i for i in range(n_ot)],
                     ticktext=["Start", "Q2", "Half", "Q4", "End of Q4"]
                     + [("OT" if i == 0 else f"{i + 1}OT") for i in range(n_ot)])
    lim = max(10, float(m["margin"].abs().max()) + 4)
    show(fig, 430, yaxis=dict(title="Clippers ahead ↑ / behind ↓", range=[-lim, lim]))
    st.caption("Shaded: the biggest run each way. Press ▶ Replay to watch it unfold; hover a turnover for who, "
               "what and what it cost.")
    with st.expander("Inside the runs"):
        for run in (run_against, run_for):
            if run is None:
                continue
            side = f"{opp} run" if run.against else "Clippers run"
            scorers = ", ".join(f"{n} {v}" for n, v in run.opp_scorers) or "nobody"
            st.markdown(f"**{side} {run.label}** ({R.run_window(run)}). On the floor for LAC: "
                        f"{', '.join(run.unit) or '?'}. Clippers turnovers: {', '.join(run.turnovers) or 'none'}; "
                        f"missed shots: {run.misses}; {opp} offensive rebounds: {run.opp_orebs}. "
                        f"{opp} points: {scorers}.")

    # ---- what decided it, and what needed to happen
    if len(factors):
        won = row["final_margin"] > 0
        st.header("What decided it" if won else "What needed to happen", divider="gray", anchor="decided")
        st.markdown(f'<div class="takeaway">Compared with their average game this season '
                    f'({signed(factors.attrs["normal_margin"])} a game), here\'s where this one was better or worse.'
                    "</div>", unsafe_allow_html=True)
        fz = factors[factors["factor"] != "residual"]
        better = fz[fz["points"] >= 0.5].sort_values("points", ascending=False)
        worse = fz[fz["points"] <= -0.5].sort_values("points")
        top = float(fz["points"].abs().max() or 1)

        def ledger(rows: pd.DataFrame, color: str, title: str, empty: str) -> str:
            items = "".join(
                f'<div class="ledger-row"><div class="lname">{n}</div><div class="lbar"><span style="width:'
                f'{100 * abs(v) / top:.0f}%;background:{color}"></span></div>'
                f'<div class="lval" style="color:{color}">{signed(v)}</div></div>'
                for n, v in zip(rows["name"], rows["points"]))
            return (f'<div class="ledger"><div class="lhead" style="color:{color}">{title}</div>'
                    + (items or f'<div class="lempty">{empty}</div>') + "</div>")

        normal, final = factors.attrs["normal_margin"], factors.attrs["margin"]
        st.markdown(f'<div class="ledger-sum">A normal night: <b>{signed(normal)}</b> &nbsp;→&nbsp; this game: '
                    f'<b style="color:{GOOD if final > 0 else BAD}">{signed(final, 0)}</b></div>',
                    unsafe_allow_html=True)
        left_, right_ = st.columns(2, gap="large")
        left_.markdown(ledger(better, GOOD, "▲ Better than a normal night", "Nothing stood out."),
                       unsafe_allow_html=True)
        right_.markdown(ledger(worse, BAD, "▼ Worse than a normal night", "Nothing stood out."),
                        unsafe_allow_html=True)
        st.caption("Points each part of the game was worth compared with their average game this season. Better and "
                   "worse add up to the difference between the two margins (plus a little the bookkeeping can't place).")
        if not won and lvs:
            st.markdown("**The fixes, biggest first**")
            answer_cards([W.Verdict(lv.title, f"{lv.points:.0f} pts", "; ".join(lv.suggestions) or "—",
                                    "bad", "decided") for lv in lvs], link=False, big_size="1.5rem")

    # ---- who played
    st.header("Who played well", divider="gray", anchor="players")
    if len(lines):
        st.markdown('<div class="takeaway">Every Clipper who played, best night first. The bar is his Game Score '
                    '(a one-number box-score summary); the tick marks 10, a good night.</div>', unsafe_allow_html=True)
        answer_cards([W.Verdict(
            f"{r_.player_name}{' · starter' if r_.starter else ''}", f"{int(r_.pts)} pts",
            f"{int(r_.reb)} reb · {int(r_.ast)} ast · {int(r_.tov)} TO · {int(r_.fgm)}/{int(r_.fga)} FG in "
            f"{r_.minutes:.0f} min · {signed(r_.plus_minus, 0)} on the floor. Game Score {r_.game_score:.1f}.",
            "good" if r_.game_score >= 10 else "bad" if r_.game_score < 3 else "info", "players",
            ("share", max(r_.game_score, 0) / 25, 10 / 25), D.headshot_url(r_.player_id))
            for r_ in lines.itertuples()], link=False, big_size="1.5rem")

    # ---- who guarded whom
    mu = R.matchups_for_game(data.matchups, gid, "LAC", opp_lines) if len(data.matchups) else pd.DataFrame()
    st.header("Who guarded whom", divider="gray", anchor="matchups")
    if mu.empty:
        st.info("Player tracking for this game isn't published yet (it usually appears a few hours after the game).")
    else:
        st.markdown(f'<div class="takeaway">{opp}\'s top scorers, and what they scored against each Clippers '
                    "defender.</div>", unsafe_allow_html=True)
        cards = []
        for (scorer, total), grp in mu.groupby(["scorer", "scorer_pts"], sort=False):
            pid = opp_lines.loc[opp_lines["player_name"] == scorer, "player_id"]
            worst = grp.sort_values("pts", ascending=False).iloc[0]
            other = int(total - grp["pts"].sum())
            cards.append(W.Verdict(
                scorer, f"{int(total)} pts",
                "<br>".join(f"Guarded by <b>{d.split(' ', 1)[-1]}</b>: {int(p_)} pts on {int(m_)}/{int(a_)} "
                            f"({pp:.0f} possessions)"
                            for d, p_, m_, a_, pp in zip(grp["defender"], grp["pts"], grp["fgm"], grp["fga"],
                                                         grp["poss"]))
                + (f"<br>The other {other}: against other defenders, in transition or at the line." if other > 0
                   else ""),
                "bad" if worst["pts"] >= 10 else "info", "matchups",
                image=D.headshot_url(pid.iloc[0]) if len(pid) else None))
        answer_cards(cards, link=False, big_size="1.6rem")
        st.caption("Points scored while each Clipper was his main defender (player tracking). Fast breaks and "
                   "scrambles with no clear defender aren't counted.")

    # ---- five-man units
    if len(game_units):
        st.header("Which lineups worked", divider="gray", anchor="units")
        ids_by_name = {v: k for k, v in names.items()}
        gu = game_units[game_units["seconds"] >= 120].sort_values("plus_minus", ascending=False)
        show_units = pd.concat([gu.head(3), gu.tail(3)]).drop_duplicates(subset=["team"])
        rows_html = []
        for u in show_units.itertuples():
            faces = "".join(f'<img src="{D.headshot_url(ids_by_name[n])}" alt="{n}" title="{n}">' for n in u.unit
                            if n in ids_by_name)
            color = GOOD if u.plus_minus > 0 else BAD if u.plus_minus < 0 else GRAY
            rows_html.append(
                f'<div class="lineup"><div class="faces">{faces}</div>'
                f'<div class="who">{", ".join(n.split(" ", 1)[-1] for n in u.unit)}</div>'
                f'<div class="min">{u.minutes:.0f} min</div>'
                f'<div class="pm" style="color:{color}">{signed(u.plus_minus, 0)}</div></div>')
        st.markdown('<div class="takeaway">The best and worst five-man units (2+ minutes together): points scored '
                    "minus allowed while they shared the floor.</div>" + "".join(rows_html), unsafe_allow_html=True)
        with st.expander("Every unit, as a chart"):
            gu2 = gu.sort_values("plus_minus")
            fig = go.Figure(go.Bar(
                y=[", ".join(n.split(" ", 1)[-1] for n in u_) for u_ in gu2["unit"]], x=gu2["plus_minus"],
                orientation="h", marker_color=[GOOD if v > 0 else BAD if v < 0 else MUTED_BAR for v in gu2["plus_minus"]],
                text=[f"{signed(v, 0)} in {mm:.0f} min" for v, mm in zip(gu2["plus_minus"], gu2["minutes"])],
                textposition="outside", cliponaxis=False, hovertemplate="%{y}: %{x:+d}<extra></extra>"))
            lim = float(gu2["plus_minus"].abs().max() or 1) * 1.6
            fig.add_vline(x=0, line=dict(color=INK_MARK, width=1))
            fig.update_xaxes(range=[-lim, lim], title="Points scored minus allowed with that five on the floor")
            show(fig, 90 + 32 * len(gu2), showlegend=False)

    # ---- the details
    with st.expander("What changed after halftime in this game"):
        a = att.set_index("factor")
        fig = go.Figure(go.Bar(
            x=a["points"], y=[S.FACTOR_PLAIN[f][0] for f in a.index], orientation="h",
            marker_color=[GRAY if f == "residual" else GOOD if v >= 0 else BAD for f, v in a["points"].items()],
            customdata=[S.FACTOR_PLAIN[f][1] for f in a.index],
            hovertemplate="<b>%{y}</b>: %{x:+.1f} pts<br>%{customdata}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.add_vline(x=0, line=dict(color=GRAY, width=1))
        show(fig, 300, xaxis_title=f"Points vs the first half (total {signed(a['margin_change'].iloc[0], 0)})")
    with st.expander("Box score"):
        cols = ["player_name", "starter", "minutes", "pts", "reb", "ast", "stl", "blk", "tov", "pf", "fgm", "fga",
                "fg3m", "fg3a", "ftm", "fta", "plus_minus", "game_score"]
        st.dataframe(lines[cols].round(1), hide_index=True, width="stretch")
        if len(game_def):
            st.markdown("**Each Clippers defender's night (tracking)**")
            st.dataframe(game_def[["def_player", "main_assignment", "poss", "pts", "fgm", "fga"]].round(1),
                         hide_index=True, width="stretch")
    with st.expander("Before vs after halftime: every stat (table)"):
        st.dataframe(D.game_half_table(data.game_halves, gid), hide_index=True, width="stretch",
                     column_config={"stat": st.column_config.TextColumn("", width="medium")})
    with st.expander("Every Clippers turnover in this game"):
        st.dataframe(D.game_turnovers(data.turnovers, gid), hide_index=True, width="stretch")

# ================================================================= players

def page_players() -> None:
    page_top("Players", "Who loses the ball after halftime. Click a card or a player to dig in.")
    season_p = st.selectbox("Season", data.seasons, index=0, key="player_season")
    ph = D.player_halves(data.player_stats, season_p)
    if ph.empty:
        st.info("No players with enough minutes in both halves yet.")
    else:
        answer_cards(S.player_cards(ph, focus))
        h2 = ph[ph["split"] == "H2"].sort_values("tov_share")
        section("Who's losing the ball after halftime?", "who",
                "Gray = how much of the offense goes through him. Orange = how many of the turnovers he makes.")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=h2["player_name"], x=h2["play_share"], orientation="h", name="Share of team plays",
                             marker_color=LIGHT, hovertemplate="%{y}: %{x:.0%} of plays<extra></extra>"))
        fig.add_trace(go.Bar(y=h2["player_name"], x=h2["tov_share"], orientation="h", name="Share of team turnovers",
                             marker_color=FOCUS,
                             customdata=h2["tov"],
                             hovertemplate="%{y}: %{x:.0%} of turnovers (%{customdata:.0f})<extra></extra>"))
        fig.update_layout(barmode="group", bargap=0.25)
        fig.update_xaxes(tickformat=".0%")
        fig.update_yaxes(tickvals=list(h2["player_name"]),
                         ticktext=[f"<b>{n}</b>" if n == focus else n for n in h2["player_name"]])
        names = sorted(ph["player_name"].unique())
        st.session_state.setdefault("player_pick", focus if focus in names else names[0])
        if st.session_state["player_pick"] not in names:
            st.session_state["player_pick"] = names[0]
        event = show(fig, 80 + 46 * len(h2), key=f"who_{season_p}", select=True)
        name = clicked(event, "y", f"who_click_{season_p}")
        if name in names:
            st.session_state["player_pick"] = name
        st.caption("Orange longer than gray = more than his share of turnovers. Click a player to see his games.")

        section("Who gets sloppier after halftime?", "sloppier",
                "Each row is one player after halftime: count the balls to his red one, and that's how many plays "
                "he gets per turnover. Fewer balls = sloppier.")
        wl = ph.pivot(index="player_name", columns="split", values="tov_pct")
        every = lambda v: 1 / v if v and v > 0 else 99.0  # noqa: E731
        ladder = sorted(((n_.split(" ", 1)[-1], every(wl.loc[n_, "H1"]), every(wl.loc[n_, "H2"])) for n_ in wl.index),
                        key=lambda r_: r_[2])
        picture(G.ball_ladder(ladder), max_width="980px")
        w = ph.pivot(index="player_name", columns="split", values="tov_pct").sort_values("H2")
        tov = ph.pivot(index="player_name", columns="split", values="tov").reindex(w.index)
        fig = go.Figure()
        for name in w.index:
            fig.add_trace(go.Scatter(x=[w.loc[name, "H1"], w.loc[name, "H2"]], y=[name, name], mode="lines",
                                     line=dict(color=LIGHT, width=3), hoverinfo="skip", showlegend=False))
        for split, color, label in (("H1", H1C, "1st half"), ("H2", H2C, "2nd half")):
            fig.add_trace(go.Scatter(
                x=w[split], y=w.index, mode="markers", name=label, customdata=tov[split],
                marker=dict(color=color, size=13, line=dict(color=BG, width=1.5)),
                hovertemplate="%{y}<br>%{x:.1%} of plays end in a turnover<br>%{customdata:.0f} turnovers"
                              "<extra>" + label + "</extra>"))
        fig.update_xaxes(tickformat=".0%", title="Turnover rate (share of a player's plays that end in a turnover)")
        fig.update_yaxes(tickvals=list(w.index), ticktext=[f"<b>{n}</b>" if n == focus else n for n in w.index])
        with st.expander("Details: the same numbers as a before/after chart"):
            st.markdown("Players with 40+ minutes in each half. Hover for the counts: a big jump on very few plays can "
                        "be a couple of bad possessions.")
            show(fig, 80 + 40 * len(w))

        section("One player, game by game", "one")
        player = st.selectbox("Player", names, key="player_pick")
        t = ph[ph["player_name"] == player].set_index("split")
        c = st.columns(4)
        c[0].metric("Turnover rate, 1st half", f"{t.loc['H1', 'tov_pct']:.1%}", border=True,
                    help="Share of his plays (shots, free-throw trips, turnovers) that end in a turnover.")
        c[1].metric("Turnover rate, 2nd half", f"{t.loc['H2', 'tov_pct']:.1%}", border=True,
                    delta=f"{100 * (t.loc['H2', 'tov_pct'] - t.loc['H1', 'tov_pct']):+.1f} pts",
                    delta_color="inverse")
        c[2].metric("Share of team turnovers after half", f"{t.loc['H2', 'tov_share']:.0%}", border=True,
                    help=f"He handles {t.loc['H2', 'play_share']:.0%} of the team's plays after halftime.")
        c[3].metric("Turnovers that were steals", f"{t.loc['H2', 'live_share']:.0%}", border=True,
                    help="Live-ball turnovers after halftime: play continues, so they often lead to fast breaks.")
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

# ================================================================== league

def page_league() -> None:
    page_top("Vs. the league", "Does every team get worse after halftime? No. Most teams barely change.")
    stat_names = {"net_rtg": "Overall (net rating)", "efg_pct": "Shooting accuracy",
                  "opp_efg_pct": "Opponents' shooting", "tov_pct": "Turnovers",
                  "opp_ft_rate": "Fouling (opponent free throws)", "opp_oreb_pct": "Second chances given up"}
    season_l = st.selectbox("Season", data.seasons, index=0, key="league_season")
    cr = D.clippers_ranks(data.ranks, season_l)
    answer_cards(S.league_cards(cr))
    section("How every team changes after halftime", "teams",
            "Pick a stat. Right = better after halftime than before it, left = worse. Clippers in orange.")
    metric = st.pills("Stat", list(stat_names), default="net_rtg", format_func=stat_names.get, key="league_metric")
    metric = metric or "net_rtg"
    lv = D.league_view(data.ranks, season_l, metric, "change")
    flip = D.better_direction(metric) == "lower is better"
    lv = lv.assign(shown=-lv["value"] if flip else lv["value"]).sort_values("shown")
    pct = metric not in ("net_rtg", "tov_pct")
    fmt = (lambda v: f"{signed(v * 100)} pts") if pct else (lambda v: signed(v))
    fig = go.Figure(go.Bar(
        x=lv["shown"], y=lv["team_name"], orientation="h",
        marker_color=[FOCUS if t else MUTED_BAR for t in lv["is_team"]],
        customdata=np.column_stack([lv["rank"], [fmt(v) for v in lv["value"]]]),
        hovertemplate="<b>%{y}</b><br>Change after halftime: %{customdata[1]}<br>%{customdata[0]} of 30"
                      "<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=GRAY, width=1))
    fig.update_yaxes(tickvals=list(lv["team_name"]),
                     ticktext=[f"<b>{n}</b>" if t else n for n, t in zip(lv["team_name"], lv["is_team"])])
    fig.update_xaxes(showticklabels=False, title="← worse after halftime        better after halftime →")
    row = lv[lv["is_team"]].iloc[0]
    left, right = st.columns([2, 1], gap="large")
    with left:
        show(fig, 720)
    with right:
        st.metric(f"Clippers: {stat_names[metric]}", f"{ordinal(int(row['rank']))} of 30", border=True,
                  help="Rank of the change from first half to second half. 1st = improves the most.")
        st.metric("Clippers' change", fmt(row["value"]), border=True,
                  help=f"League average change: {fmt(row['league_mean'])}.")

    section("Where the Clippers rank on everything", "ranks",
            "Blue dot = their league rank in first halves, orange = second halves. Right = better.")
    stats = [x for x in cr.index if x != "Pace (possessions)"]
    fig = go.Figure()
    for name in stats:
        a, b = cr.loc[name, cr.columns[0]], cr.loc[name, cr.columns[1]]
        fig.add_trace(go.Scatter(x=[a, b], y=[name, name], mode="lines", hoverinfo="skip", showlegend=False,
                                 line=dict(color=BAD if b > a + 2 else GOOD if b < a - 2 else LIGHT, width=3)))
    for col, color, label in ((cr.columns[0], H1C, "1st half"), (cr.columns[1], H2C, "2nd half")):
        fig.add_trace(go.Scatter(
            x=cr.loc[stats, col], y=stats, mode="markers", name=label,
            marker=dict(color=color, size=13, line=dict(color=BG, width=1.5)),
            text=[ordinal(int(v)) for v in cr.loc[stats, col]],
            hovertemplate="%{y}: %{text} of 30<extra>" + label + "</extra>"))
    fig.update_xaxes(range=[30.5, 0.5], tickvals=[30, 20, 10, 1], ticktext=["30th", "20th", "10th", "1st"],
                     title="League rank (1st = best)")
    fig.update_yaxes(autorange="reversed")
    show(fig, 90 + 38 * len(stats))
    st.caption("Red line = they drop 3+ places after halftime; blue = they climb.")
    with st.expander("The ranks as a table"):
        st.dataframe(cr.map(ordinal), width="stretch", height=38 + 35 * len(cr))


# ================================================================== next game

TAPE = [  # (label, column, higher is better)
    ("Overall (net rating)", "net", True), ("Scoring (per 100)", "ortg", True), ("Defense (allowed per 100)", "drtg", False),
    ("Shooting accuracy", "efg", True), ("Opponents' shooting", "opp_efg", False),
    ("Ball security", "tov_pct", False), ("Forcing turnovers", "opp_tov_pct", True),
    ("Offensive rebounding", "oreb_pct", True), ("Defensive rebounding", "opp_oreb_pct", False),
    ("Getting to the line", "ft_rate", True),
]


def _faces(lean: list[tuple[str, int, str]]) -> str:
    """Small round headshots with name and the stat behind each pick."""
    return "".join(
        f'<div class="lean"><img src="{D.headshot_url(pid)}" alt=""><div><b>{name.split(" ", 1)[-1]}</b>'
        f'<br><span>{stat}</span></div></div>' for name, pid, stat in lean)


def _game_plan(game: pd.Series, pv, teams: pd.DataFrame, shooting: float | None) -> None:
    """An upcoming game: the odds, a checklist of keys that updates them, matchups, and the tale of the tape."""
    tid = settings.team_id
    tip = game["tipoff_utc"].tz_convert("America/Los_Angeles")
    opp_id = int(game["away_id"] if pv.home else game["home_id"])
    st.markdown(f'<div class="kicker">{tip:%a %b} {tip.day} · {tip:%I:%M %p}'.replace(" 0", " ") + " PT · "
                f'{pv.arena} · {"home" if pv.home else "away"}</div>'
                f'<div class="hero">Clippers {"vs" if pv.home else "at"} {pv.opponent_name}</div>',
                unsafe_allow_html=True)
    lac, opp = pv.lac, pv.opp
    players = GP.player_table(data.player_box, season_now, "LAC")
    misc = data.team_misc[data.team_misc["season"] == season_now] if len(data.team_misc) else pd.DataFrame(
        columns=["team_id", "opp_pts_paint"])
    keys = GP.game_plan(pv, teams, misc, players, shooting)

    # ---- the odds, live with the keys picked below
    odds = st.empty()
    h2h = P.head_to_head(data.games, season_now, pv.opponent)
    tiles = [
        W.Verdict("Clippers this season", signed(lac["net"]),
                  f"Points per 100 possessions, scored minus allowed: {P._ord(lac['net_rank'])} of 30.",
                  "bad" if lac["net"] < 0 else "good", "plan", ("rank", int(lac["net_rank"]), 30), D.logo_url(tid)),
        W.Verdict(f"{pv.opponent} this season", signed(opp["net"]), f"{P._ord(opp['net_rank'])} of 30.",
                  "bad" if opp["net"] > lac["net"] else "good", "plan", ("rank", int(opp["net_rank"]), 30),
                  D.logo_url(opp_id)),
    ]
    if len(h2h):
        wins = int((h2h["result"] == "W").sum())
        tiles.append(W.Verdict("Earlier this season", f"{wins}-{len(h2h) - wins}",
                               "; ".join(f"{r.result} {int(r.team_pts)}-{int(r.opp_pts)}" for r in h2h.itertuples()),
                               "info", "plan"))

    section(f"How to beat the {pv.opponent_name.split(' ')[-1]}", "plan",
            "The keys to this game from both teams' season numbers. Switch them on and off: the odds above "
            "update.")
    chosen = []
    if keys:
        for j, k in enumerate(keys):
            with st.container(border=True, key=f"key_{k.kind}"):
                c1, c2, c3 = st.columns([1.15, 2.4, 1.45], vertical_alignment="center", gap="medium")
                on = c1.toggle(k.title, value=False, key=f"plan_{game['game_id']}_{k.kind}")
                c1.markdown(f'<div class="kval">+{k.points:.1f} pts a game<br><span>{k.chance:.0%} to win with '
                            f"this alone</span></div>", unsafe_allow_html=True)
                c2.markdown(f'<div class="kwhy">{k.why}</div>', unsafe_allow_html=True)
                c3.markdown(_faces(k.lean_on), unsafe_allow_html=True)
                if on:
                    chosen.append(k)
        st.caption("Keys overlap a little (a turnover avoided can also be a shot that misses), so several together "
                   "are a best case.")
    else:
        st.write("Nothing in either team's numbers points to a specific key: it's about playing their usual game.")
    pts = sum(k.points for k in chosen)
    chance = P.win_chance(pv.margin + pts)
    with odds.container():
        picture(G.odds_bar("LAC", pv.opponent, chance, pv.margin + pts, team_logo=D.logo_url(tid),
                           opp_logo=D.logo_url(opp_id)), max_width="900px")
        st.markdown(f'<div class="odds-note">Today <b>{pv.chance:.0%}</b>'
                    + (f' → with {len(chosen)} key{"s" if len(chosen) != 1 else ""} below <b>{chance:.0%}</b>'
                       if chosen else " · switch on keys below to see what they're worth") + "</div>",
                    unsafe_allow_html=True)
        answer_cards(tiles, link=False)

    # ---- matchups
    defenders = W.defenders(data.matchups, data.hustle, data.bio, season_now, "LAC") if len(data.matchups) else None
    watch = P.who_to_watch(data.player_season, data.bio, season_now, pv.opponent,
                           defenders.table if defenders is not None else None)
    longs = P.long_rotation(data.player_season, data.bio, season_now, pv.opponent)
    if len(watch):
        section("Matchups: who guards their scorers", "watch",
                f"{pv.opponent}'s top scorers this season, and the Clipper best suited to guard each (closest in size, "
                "fewest points allowed per possession from player tracking).")
        ids = data.player_season.set_index("player_name")["player_id"].to_dict()
        answer_cards([W.Verdict(
            r.player_name, f"{r.pts:.1f} ppg",
            (f"{G._ft_in(r.height_in)}, {r.weight_lb:.0f} lb. " if pd.notna(r.height_in) and pd.notna(r.weight_lb)
             else "") + f"{r.reb:.1f} reb, {r.ast:.1f} ast in {r.min:.0f} min. "
            + (f"Guard him with: {r.guard_with.rstrip('.')}." if r.guard_with else ""), "warn", "watch",
            image=D.headshot_url(ids[r.player_name]) if r.player_name in ids else None)
            for r in watch.itertuples()], link=False)
        if len(longs) >= P.LONG_ROTATION:
            st.warning(f"{pv.opponent} plays {len(longs)} players between 6'6\" and 6'10\" in its top eight "
                       f"({', '.join(longs)}). {focus_last} turns it over about twice as often against defenders "
                       "that size: get him the ball on the move or off screens.", icon=":material/height:")
        elif longs:
            st.caption(f"{pv.opponent}'s 6'6\"-6'10\" rotation players: {', '.join(longs)}.")

    # ---- tale of the tape
    section("Tale of the tape", "tape", "League rank on each part of the game (1st = best). Further right is better.")
    fig = go.Figure()
    labels = [x[0] for x in TAPE]
    for name, row_, color in (("LAC", lac, H1C), (pv.opponent, opp, FOCUS)):
        fig.add_trace(go.Scatter(
            x=[row_[f"{c}_rank"] for _, c, _ in TAPE], y=labels, mode="markers+text", name=name,
            text=[P._ord(row_[f"{c}_rank"]) for _, c, _ in TAPE], textposition="top center",
            marker=dict(color=color, size=14, line=dict(color=BG, width=1.5)),
            hovertemplate="%{y}: %{text}<extra>" + name + "</extra>"))
    for lab, c, _ in TAPE:
        fig.add_shape(type="line", x0=lac[f"{c}_rank"], x1=opp[f"{c}_rank"], y0=lab, y1=lab,
                      line=dict(color=LIGHT, width=3), layer="below")
    fig.update_xaxes(range=[31, 0], tickvals=[30, 20, 10, 1], ticktext=["30th", "20th", "10th", "1st"],
                     title="League rank this season")
    fig.update_yaxes(autorange="reversed")
    show(fig, 90 + 40 * len(TAPE))

    with st.expander("How the odds and the game plan are worked out"):
        st.markdown(
            f"- **Team strength** is points scored minus allowed per 100 possessions this season, blended with last "
            f"season's while the sample is small: (games x this season + {P.PRIOR_GAMES} x last season) / "
            f"(games + {P.PRIOR_GAMES}).\n"
            f"- **Expected margin** = the difference in strength x expected possessions / 100, plus "
            f"{P.HOME_POINTS:.0f} points for the home team.\n"
            f"- **Win chance** treats the final margin as landing within about {P.MARGIN_SD:.1f} points of that, the "
            "usual spread of an NBA game.\n"
            "- **Keys** only appear when the numbers point to them; each rule (and how its points are counted) is "
            "written at the top of src/courtside/analysis/gameplan.py.\n"
            "- It doesn't know about injuries, rest or travel, and isn't betting advice. Models this simple pick "
            "roughly two in three NBA games correctly.")
        st.dataframe(teams[["team_name", "gp", "net", "prior_net", "blend_net", "pace"]].sort_values(
            "blend_net", ascending=False).round(1), hide_index=True, width="stretch")


PAGE_ANSWER = st.Page(page_answer, title="Blown leads", icon=":material/lightbulb:", url_path="answer")
PAGE_WHATNOW = st.Page(page_whatnow, title="What now?", icon=":material/checklist:", url_path="what-now")
PAGE_REPLAY = st.Page(page_replay, title="Games", icon=":material/sports_basketball:", url_path="games",
                      default=True)
PAGE_PLAYERS = st.Page(page_players, title="Players", icon=":material/groups:", url_path="players")
PAGE_LEAGUE = st.Page(page_league, title="Vs. the league", icon=":material/leaderboard:", url_path="league")
PAGES = {"replay": PAGE_REPLAY, "answer": PAGE_ANSWER, "whatnow": PAGE_WHATNOW,
         "players": PAGE_PLAYERS, "league": PAGE_LEAGUE}
nav = st.navigation(list(PAGES.values()), position="top")
# Tests run a specific page by setting this in session state (AppTest can't change the URL).
forced = st.session_state.get("_dashboard_page")
if forced in PAGES:
    {"answer": page_answer, "whatnow": page_whatnow, "replay": page_replay, "players": page_players,
     "league": page_league}[forced]()
else:
    nav.run()
