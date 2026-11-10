"""Clippers lead-collapse dashboard.

Run with ``leadcollapse dashboard`` (or ``streamlit run dashboard/app.py``).
Reads ``data/processed/`` and ``reports/analysis.json`` only: no API calls.
Written for two audiences at once: plain-English headlines and explanations
up front, the basketball numbers one hover or expander away.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from leadcollapse.analysis.hypotheses import CONTRIBUTES, DRIVES, ordinal
from leadcollapse.config import get_settings
from leadcollapse.viz import dashboard_data as D
from leadcollapse.viz import graphics as G
from leadcollapse.viz import story as S
from leadcollapse.viz import whatnow as W

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

st.set_page_config(page_title="Why the Clippers blow leads", layout="wide", initial_sidebar_state="collapsed")
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
.vcard {display: block; text-decoration: none !important; color: inherit !important; background: #1f1f1d;
        border-radius: 10px; border-left: 5px solid #898781; padding: .85rem 1rem; transition: transform .12s}
.vcard:hover {transform: translateY(-2px); background: #262624}
.v-title {font-weight: 700; font-size: 1rem; line-height: 1.25}
.v-big {font-size: 1.9rem; font-weight: 800; line-height: 1.15; margin: .35rem 0 .2rem 0}
.v-small {font-size: .86rem; opacity: .72; line-height: 1.3}
.gfx svg {width: 100%; height: auto; display: block}
.gfx {margin: .3rem 0 .6rem 0}
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
    """Render one of the SVG graphics from leadcollapse.viz.graphics."""
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
    st.error(f"Processed data not found ({e.filename}). Run `leadcollapse build` and `leadcollapse analyze` first.")
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
        answer_cards(S.answer_page_cards(hl, reasons, res, season_prev))
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
                + ('<a href="#hiding">What\'s hiding it</a>' if helps else '')
                + '<a href="#theories">Theories checked</a><a href="#changes">Since last season</a>'
                '<a href="#whatif">What if?</a><a href="#stats">Every stat</a></div>', unsafe_allow_html=True)

    # ---------------------------------------------------------- 1. ranked reasons
    section("1. Why they blow leads, ranked", "reasons",
            "Each pair of bars is one part of the game: <b>red = this season</b>, gray = last season. Left of zero = "
            "it costs them points after halftime; right = it helps." if hurts else "")
    if hurts:
        names = [f"{i}. {r.name}" for i, r in enumerate(hurts, 1)]
        fig = go.Figure()
        for label, vals, color in ((f"This season ({season_now})", [r.per_game for r in hurts], BAD),
                                   (f"Last season ({season_prev})", [r.prev_per_game for r in hurts], MUTED_BAR)):
            fig.add_trace(go.Bar(
                y=names, x=vals, orientation="h", name=label, marker_color=color,
                text=[f"{signed(v)} a game" for v in vals], textposition="outside", cliponaxis=False,
                customdata=[r.detail for r in hurts],
                hovertemplate="<b>%{y}</b><br>" + label + ": %{x:+.1f} points a game after halftime"
                              "<br>%{customdata}<extra></extra>"))
        lo = min(min(r.per_game, r.prev_per_game) for r in hurts)
        hi = max(max(r.per_game, r.prev_per_game) for r in hurts)
        fig.add_vline(x=0, line=dict(color=INK_MARK, width=1))
        fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Points a game after halftime  (← costs them · helps them →)",
                         range=[lo * 1.45, max(hi, 0) * 1.45 + 0.8], zeroline=False)
        show(fig, 110 + 70 * len(hurts))
        st.caption("Ranked by how much each costs this season. Hover a bar for the stat behind it.")
        with st.expander("Details for each reason"):
            if helps:
                st.markdown("The shares add up to more than 100% because a few things got *better* after halftime "
                            "and cancel some of it out (section 2).")
            for i, r in enumerate(hurts, 1):
                st.markdown(f"**{i}. {r.name}** costs **{-r.per_game:.1f} points a game** after halftime "
                            f"({r.share:.0%} of the drop). {r.detail} Last season: {signed(r.prev_per_game)} a game.")
                st.badge(r.status, color=STATUS_BADGE[r.status])

    # ---------------------------------------------------------- 2. what's hiding it
    if helps:
        total_help = sum(r.per_game for r in helps)
        residual = att_now.set_index("factor").loc["residual", "per_game"] if att_now is not None else 0.0
        section("2. What's hiding how bad it is", "hiding",
                f"A few things got <b>better</b> after halftime, worth {signed(total_help)} points a game. "
                + (f"Without them, second halves would be {signed(hl['h2_per_game'] - total_help)} a game, not "
                   f"{signed(hl['h2_per_game'])}." if hl else ""))
        fig = go.Figure(go.Bar(
            y=[r.name for r in helps], x=[r.per_game for r in helps], orientation="h", marker_color=GOOD,
            text=[f"{signed(r.per_game)} pts a game" for r in helps], textposition="outside",
            customdata=[r.detail for r in helps], hovertemplate="<b>%{y}</b><br>%{customdata}<extra></extra>"))
        fig.update_xaxes(title="Points a game gained after halftime",
                         range=[0, max(r.per_game for r in helps) * 1.5])
        fig.update_yaxes(autorange="reversed")
        show(fig, 90 + 50 * len(helps), showlegend=False)
        with st.expander("Details"):
            for r in helps:
                st.markdown(f"- **{r.name}**: {r.detail}")
            st.markdown(f"- A further {signed(residual)} a game can't be placed in any one bucket.")

    # ---------------------------------------------------------- 3. theories
    section(f"{3 if helps else 2}. The popular theories, checked", "theories")
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
            fig = go.Figure(go.Bar(
                y=["Share of team plays", "Share of team turnovers"], x=[k3["play_share_h2"], k3["tov_share_h2"]],
                orientation="h", marker_color=[GRAY, FOCUS], text=[f"{k3['play_share_h2']:.0%}",
                                                                   f"{k3['tov_share_h2']:.0%}"],
                textposition="outside", hovertemplate="%{y}: %{x:.0%}<extra>after halftime</extra>"))
            fig.update_xaxes(range=[0, max(0.5, k3["tov_share_h2"] * 1.3)], tickformat=".0%")
            show(fig, 170, showlegend=False, title=dict(text="Garland after halftime", font=dict(size=14)))
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
        left, right = st.columns([1, 2.2], gap="large", vertical_alignment="center")
        with left:
            st.markdown("The Wall is 51 rows of Clippers fans behind one basket. If it rattled opponents at the "
                        "line after halftime, their second-half free throws would drop at home.")
            st.link_button("See the full Intuit Dome study", settings.wall_study_url,
                           icon=":material/open_in_new:", width="stretch")
        with right:
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
            st.markdown("**Opponents at the free-throw line, before vs after halftime**")
            show(fig, 320)
        with st.expander("Details"):
            st.markdown(
                "- Teams switch baskets at halftime, so visitors face The Wall for only one half. The box score "
                "doesn't record which end a team shot at, so this checks whether opponents miss more after "
                "halftime at home at all. They don't.\n"
                "- Includes late free throws from intentional fouls.\n"
                "- The home-court study looks at every arena since 2013-14 and finds the same thing: visitors' "
                "free-throw shooting at Intuit Dome isn't consistently worse than at other arenas.")

    # ---------------------------------------------------------- 4. since last season
    section_n = 4 if helps else 3
    if not data.comparison.empty and season_prev in data.attribution:
        st.header(f"{section_n}. What changed since last season", divider="gray", anchor="changes")
        comp = data.comparison.set_index("season")
        cur, prev = comp.loc[season_now], comp.loc[season_prev]
        c = st.columns(4)
        c[0].metric(f"Early leads blown, {season_prev}",
                    f"{int(prev['blown_leads'])} of {int(prev['blown_leads'] + prev['held_leads'])}", border=True)
        c[1].metric(f"Early leads blown, {season_now}",
                    f"{int(cur['blown_leads'])} of {int(cur['blown_leads'] + cur['held_leads'])}", border=True)
        c[2].metric(f"2nd halves per game, {season_prev}", signed(prev["margin_h2_per_game"]), border=True)
        c[3].metric(f"2nd halves per game, {season_now}", signed(cur["margin_h2_per_game"]), border=True)
        new = [r.name for r in reasons if r.status == S.NEW]
        same = [r.name for r in reasons if r.status == S.SAME]
        better = [r.name for r in reasons if r.status == S.BETTER]
        answer_cards([W.Verdict(title, " · ".join(names) or "Nothing", sub, tone, "changes")
                      for title, names, sub, tone in (
                          ("New this season", new, "Worse than last season by a point a game or more.", "bad"),
                          ("Same as last season", same, "Hurt then, hurts now.", "warn"),
                          ("Better than last season", better, "Hurts less than it did.", "good"))], link=False,
                     big_size="1.15rem")

        factors = [f for f in S.factor_order() if f != "residual"]
        fig = go.Figure()
        for f in factors:
            fig.add_trace(go.Scatter(x=[prev[f"{f}_per_game"], cur[f"{f}_per_game"]],
                                     y=[S.FACTOR_PLAIN[f][0]] * 2, mode="lines", line=dict(color=LIGHT, width=3),
                                     hoverinfo="skip", showlegend=False))
        for season_, vals, color, symbol in ((season_now, cur, FOCUS, "circle"),
                                             (season_prev, prev, GRAY, "circle-open")):
            fig.add_trace(go.Scatter(
                x=[vals[f"{f}_per_game"] for f in factors], y=[S.FACTOR_PLAIN[f][0] for f in factors],
                mode="markers", name=f"{season_} ({int(vals['games'])} games)",
                marker=dict(color=color, size=14, symbol=symbol, line=dict(width=2.5, color=color)),
                hovertemplate="%{y}: %{x:+.1f} points per game<extra>" + season_ + "</extra>"))
        fig.add_vline(x=0, line=dict(color=GRAY, width=1))
        fig.update_yaxes(autorange="reversed")
        st.markdown("**Each bucket, this season vs last**")
        show(fig, 360, xaxis_title="Points per game gained (+) or lost (−) after halftime")
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


def answer_cards(cards: list[W.Verdict], link: bool = True, big_size: str | None = None) -> None:
    """Short-headline cards with one big number each; with ``link`` a click jumps to the card's section."""
    big_style = f' style="font-size:{big_size}"' if big_size else ""
    html = "".join(
        (f'<a class="vcard" href="#{c.anchor}"' if link else '<div class="vcard"')
        + f' style="border-left-color:{TONE_COLOR.get(c.tone, GRAY)}">'
        f'<div class="v-title">{c.title}</div><div class="v-big"{big_style}>{c.big}</div>'
        f'<div class="v-small">{c.small}</div>' + ("</a>" if link else "</div>")
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
                f"Starters: {sf.h2_pm / n:+.1f} a game after halftime; everyone else {rest:+.1f}.",
                ("Keep it", "blue") if sf.not_the_problem else ("Maybe change it", "orange"))
        vals = [sf.h2_pm / n, rest]
        fig = go.Figure(go.Bar(
            y=["Starting five together", "Every other lineup"], x=vals, orientation="h",
            marker_color=[GOOD if v >= 0 else BAD for v in vals], text=[f"{signed(v)} a game" for v in vals],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:+.1f} points a game after halftime<extra></extra>"))
        lim = max(abs(v) for v in vals + [2]) * 2.2
        fig.add_vline(x=0, line=dict(color=GRAY, width=1))
        fig.update_xaxes(range=[-lim, lim], title="Points a game after halftime (+ = Clippers outscore the other team)")
        fig.update_yaxes(autorange="reversed")
        show(fig, 190, showlegend=False)
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
                own = pr.steals - pr.help
                parts = [("Stolen by his defender", own, MUTED_BAR),
                         ("Stolen by a 2nd defender (trap or help)", pr.help, FOCUS),
                         ("No steal (bad pass out, travel, foul)", pr.tov - pr.steals, "#3d3c39")]
                fig = go.Figure(go.Pie(
                    labels=[p[0] for p in parts], values=[p[1] for p in parts], hole=0.6, sort=False,
                    marker=dict(colors=[p[2] for p in parts], line=dict(color=BG, width=2)),
                    textinfo="value", hovertemplate="%{label}: %{value} (%{percent})<extra></extra>"))
                fig.add_annotation(text=f"<b>{pr.tov}</b><br>turnovers", showarrow=False, font=dict(size=15))
                st.markdown("**Is he being double-teamed into them?**")
                show(fig, 250, legend=dict(orientation="v", x=1.0, y=0.5))
                st.caption(f"Only {pr.help_share:.0%} come from a second defender (teammates: "
                           f"{pr.mates_help_share:.0%}). Double teams aren't logged publicly; this is the closest "
                           "proxy.")
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
    left, right = st.columns([1, 1], gap="large")
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
            st.markdown("**How big are the players each one guards?**")
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
        fig = go.Figure(go.Pie(
            labels=["Back from last season", "New this season"], values=[chem.continuity, 1 - chem.continuity],
            hole=0.6, sort=False, marker=dict(colors=[H1C, FOCUS], line=dict(color=BG, width=2)),
            texttemplate="%{percent:.0%}", hovertemplate="%{label}: %{percent} of minutes<extra></extra>"))
        fig.add_annotation(text="share of<br>minutes", showarrow=False)
        st.markdown("**How much of the team is new**")
        show(fig, 280, legend=dict(orientation="h", y=-0.1))
    with right:
        if len(p):
            show_p = pd.concat([p.head(5), p.tail(5)]).drop_duplicates(subset=["a_id", "b_id"]).sort_values("pm")
            labels = [f"{a.split(' ', 1)[-1]} + {b.split(' ', 1)[-1]}{' ✦' if new else ''}"
                      for a, b, new in zip(show_p["a"], show_p["b"], show_p["new_pair"])]
            fig = go.Figure(go.Bar(
                y=labels, x=show_p["pm"], orientation="h",
                marker_color=[GOOD if v > 0 else BAD if v < 0 else MUTED_BAR for v in show_p["pm"]],
                text=[f"{v:+.0f} in {m:.0f} min" for v, m in zip(show_p["pm"], show_p["min"])],
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
                         f"including zero. It's {k1['boot_lo']:+.1f} to {k1['boot_hi']:+.1f} today.")
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
        st.caption("Everything on this page reruns with `leadcollapse all` after each game. The rules behind each "
                   "verdict are listed at the top of src/leadcollapse/viz/whatnow.py.")


def page_replay() -> None:
    page_top("Game replay", "Pick a game to see what happened and watch the score play out.")
    top_l, top_r = st.columns([1, 3])
    season = top_l.selectbox("Season", data.seasons, index=0, key="game_season")
    options = D.game_options(data.game_leads, season).iloc[::-1].reset_index(drop=True)  # oldest first
    leads = data.game_leads.set_index("game_id")

    # Season strip: halftime margin -> final margin for every game.
    strip = leads.loc[options["game_id"]].reset_index()
    xs = pd.to_datetime(strip["game_date"]).dt.strftime("%b %d").str.replace(" 0", " ") + " " + strip[
        "opponent_abbreviation"]
    fig = go.Figure()
    for x, h, f, blown in zip(xs, strip["halftime_margin"], strip["final_margin"],
                              strip["lead_class"] == "blown_lead"):
        fig.add_trace(go.Scatter(x=[x, x], y=[h, f], mode="lines", line=dict(color=BAD if blown else LIGHT, width=3),
                                 hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=xs, y=strip["halftime_margin"], mode="markers", name="At halftime",
                             marker=dict(color=H1C, size=11, line=dict(color=BG, width=1.5)),
                             hovertemplate="%{x}<br>At halftime: %{y:+d}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=strip["final_margin"], mode="markers", name="Final",
                             marker=dict(color=H2C, size=11, symbol="diamond", line=dict(color=BG, width=1.5)),
                             hovertemplate="%{x}<br>Final: %{y:+d}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=GRAY, width=1))
    if len(strip) > 20:
        fig.update_xaxes(showticklabels=False)
    top_r.markdown("**Every game: halftime (●) to final (◆).** Red = a blown early lead. *Click one to open it.*")
    labels = [f"{x} · {r}" for x, r in zip(xs, strip["result"])]
    by_x = dict(zip(xs, labels))
    default = next((i for i in range(len(strip) - 1, -1, -1) if strip["lead_class"].iloc[i] == "blown_lead"),
                   len(strip) - 1)
    pick_key = f"pick_{season}"
    st.session_state.setdefault(pick_key, labels[default])
    jump = st.session_state.pop("jump_game", None)
    if jump in set(strip["game_id"]):
        st.session_state[pick_key] = labels[list(strip["game_id"]).index(jump)]
    with top_r:
        event = show(fig, 230, key=f"strip_{season}", select=True, yaxis_title="Clippers margin")
    x = clicked(event, "x", f"strip_click_{season}")
    if x in by_x:
        st.session_state[pick_key] = by_x[x]
    if len(labels) <= 16:
        pick = st.pills("Pick a game", labels, key=pick_key)
    else:
        pick = st.selectbox("Pick a game", labels, key=pick_key)
    pick = pick or labels[default]
    gid = strip["game_id"].iloc[labels.index(pick)]
    row = leads.loc[gid]
    halves = data.game_halves[(data.game_halves["game_id"] == gid)
                              & data.game_halves["split"].astype(str).isin(["H1", "H2"])]
    att = D.game_attribution(data.game_halves, gid)

    answer_cards(S.game_cards(row, halves, data.turnovers, focus_last), link=False)
    with st.container(border=True):
        tag = S.LEAD_TAG.get(row["lead_class"], "")
        if tag:
            st.badge(tag, color="red" if row["lead_class"] == "blown_lead" else "blue")
        st.markdown(f'<div class="story">{S.game_story(row, halves, att)}</div>', unsafe_allow_html=True)

    m = D.game_margin(data.margins, gid)
    end = max(48.0, float(m["minute"].max()))
    blown = row["lead_class"] == "blown_lead"
    line_color = BAD if blown else H1C
    opp = row["opponent_abbreviation"]
    show_tov = st.toggle("Show Clippers turnovers on the chart", value=True,
                         help=f"Orange diamonds are {focus}'s turnovers; gray circles are everyone else's.")

    fig = go.Figure()
    fig.add_vrect(x0=24, x1=end, fillcolor="rgba(137,135,129,0.12)", line_width=0, layer="below",
                  annotation_text="2nd half", annotation_position="top left")
    fig.add_vline(x=24, line=dict(color=GRAY, width=1))
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

    # Replay animation: the colored line grows minute by minute.
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
    st.caption("Above the line = Clippers ahead. Press ▶ Replay. Hover a turnover for who, what and what it cost.")

    st.markdown("**What changed after halftime in this game** (green = better than the first half, red = worse)")
    a = att.set_index("factor")
    fig = go.Figure(go.Bar(
        x=a["points"], y=[S.FACTOR_PLAIN[f][0] for f in a.index], orientation="h",
        marker_color=[GRAY if f == "residual" else GOOD if v >= 0 else BAD for f, v in a["points"].items()],
        customdata=[S.FACTOR_PLAIN[f][1] for f in a.index],
        hovertemplate="<b>%{y}</b>: %{x:+.1f} pts<br>%{customdata}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")
    fig.add_vline(x=0, line=dict(color=GRAY, width=1))
    show(fig, 300, xaxis_title=f"Points vs the first half (total {signed(a['margin_change'].iloc[0], 0)})")
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
                "Each line goes from a player's first half (blue) to his second half (orange). Further right = "
                "more of his plays end in a turnover.")
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
        show(fig, 80 + 40 * len(w))
        with st.expander("Details"):
            st.markdown("Players with 40+ minutes in each half. Hover for the counts: a big jump on very few plays can "
                        "be a couple of bad possessions.")

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
    fmt = (lambda v: f"{v * 100:+.1f} pts") if pct else (lambda v: f"{v:+.1f}")
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


PAGE_ANSWER = st.Page(page_answer, title="The answer", icon=":material/lightbulb:", url_path="answer", default=True)
PAGE_WHATNOW = st.Page(page_whatnow, title="What now?", icon=":material/checklist:", url_path="what-now")
PAGE_REPLAY = st.Page(page_replay, title="Game replay", icon=":material/replay:", url_path="replay")
PAGE_PLAYERS = st.Page(page_players, title="Players", icon=":material/groups:", url_path="players")
PAGE_LEAGUE = st.Page(page_league, title="Vs. the league", icon=":material/leaderboard:", url_path="league")
PAGES = {"answer": PAGE_ANSWER, "whatnow": PAGE_WHATNOW, "replay": PAGE_REPLAY, "players": PAGE_PLAYERS, "league": PAGE_LEAGUE}
nav = st.navigation(list(PAGES.values()), position="top")
# Tests run a specific page by setting this in session state (AppTest can't change the URL).
forced = st.session_state.get("_dashboard_page")
if forced in PAGES:
    {"answer": page_answer, "whatnow": page_whatnow, "replay": page_replay, "players": page_players, "league": page_league}[forced]()
else:
    nav.run()
