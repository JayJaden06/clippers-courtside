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
from leadcollapse.viz import story as S

H1C, H2C = "#2a78d6", "#eb6834"  # 1st half / 2nd half
GOOD, BAD, GRAY = "#2a78d6", "#e34948", "#898781"
FOCUS = "#eb6834"
INK_MARK = "#52514e"
LIGHT = "rgba(137,135,129,0.35)"
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
</style>
""", unsafe_allow_html=True)

settings = get_settings()


@st.cache_data
def load() -> D.DashboardData:
    return D.load(settings.processed_dir, settings.reports_dir / "analysis.json")


def show(fig: go.Figure, height: int, **layout) -> None:
    fig.update_layout(height=height, margin=dict(l=8, r=8, t=36, b=8), hovermode="closest",
                      legend=dict(orientation="h", yanchor="bottom", y=1.0, x=0, title=None),
                      hoverlabel=dict(align="left"), **layout)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(zeroline=False)
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)


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


tab_story, tab_games, tab_players, tab_league = st.tabs(["The answer", "Game replay", "Players", "Vs. the league"])

# ================================================================ the answer
STATUS_BADGE = {S.NEW: "red", S.SAME: "orange", S.SIMILAR: "gray", S.BETTER: "blue"}
VERDICT_MARK = {"Worse than last season": "▼ Worse than last season", "Better than last season": "▲ Better than last season",
                "About the same": "= About the same"}

with tab_story:
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
        main = "Darius Garland's turnovers"
        garland_main = "H3" in res and res["H3"]["key"]["share_of_swing"] >= DRIVES
        st.markdown(
            f'<div class="lede">The Clippers are {hl["wins"]}-{hl["losses"]}. In {hl["leads"]} games they jumped '
            f'out to an early lead of 8 or more, and they lost {hl["blown"]} of them. A popular explanation is '
            f'{main}. '
            + ("The numbers back that up." if garland_main else
               "The numbers say that's part of it, but not the main reason. Here's the whole answer.")
            + "</div>", unsafe_allow_html=True)
        glossary_button()
        c = st.columns(5)
        c[0].metric("Record", f"{hl['wins']}-{hl['losses']}", border=True)
        c[1].metric("1st halves, per game", signed(hl["h1_per_game"]), border=True,
                    help="Average score difference over the first two quarters. Negative = outscored.")
        c[2].metric("2nd halves, per game", signed(hl["h2_per_game"]), border=True,
                    delta=f"{hl['h2_per_game'] - hl['h1_per_game']:+.1f} vs 1st half",
                    help="Average score difference over the last two quarters (overtime not included).")
        c[3].metric("Early leads blown", f"{hl['blown']} of {hl['leads']}", border=True,
                    help="Games where they led by 8+ in the first 30 minutes and still lost.")
        c[4].metric("Shooting after halftime", f"{hl['efg_h2']:.1%}", border=True,
                    delta=f"{100 * (hl['efg_h2'] - hl['efg_h1']):+.1f} pts vs 1st half",
                    help="Shooting accuracy (eFG%): makes per shot, with 3-pointers worth extra.")

    # ---------------------------------------------------------- 1. ranked reasons
    hurts = [r for r in reasons if r.per_game < 0]
    helps = [r for r in reasons if r.per_game >= 0]
    st.header("1. Why they blow leads, ranked", divider="gray")
    st.write("Every point they lose after halftime, sorted into buckets. Each bar is how many points per game that "
             "part of the game costs them in the second half compared with the first."
             + (" The percentages add up to more than 100% because a few things got *better* after halftime "
                "(section 2) and cancel some of it out." if helps else ""))
    if hurts:
        names = [f"{i}. {r.name}" for i, r in enumerate(hurts, 1)]
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=names, x=[-r.per_game for r in hurts], orientation="h", name=f"{season_now}",
            marker_color=BAD, text=[f"−{-r.per_game:.1f} pts/game" for r in hurts], textposition="outside",
            customdata=[r.detail for r in hurts],
            hovertemplate="<b>%{y}</b><br>%{customdata}<extra></extra>"))
        fig.add_trace(go.Scatter(
            y=names, x=[max(-r.prev_per_game, 0) for r in hurts], mode="markers", name=f"{season_prev}",
            marker=dict(symbol="line-ns", size=26, line=dict(width=3, color=INK_MARK)),
            customdata=[r.prev_per_game for r in hurts],
            hovertemplate=f"{season_prev}: " + "%{customdata:+.1f} pts/game<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Points per game lost after halftime", range=[0, max(-hurts[0].per_game * 1.3, 1)])
        show(fig, 90 + 58 * len(hurts))
        st.caption(f"Red bars = this season. Black ticks = the same bucket last season ({season_prev}); a tick at 0 "
                   "means it didn't cost them anything last season. Hover a bar for the stat behind it.")
        for i, r in enumerate(hurts, 1):
            with st.container(border=True):
                n, body = st.columns([0.07, 0.93], vertical_alignment="center")
                n.markdown(f'<div class="rank">{i}</div>', unsafe_allow_html=True)
                with body:
                    st.markdown(f"**{r.name}** costs **{-r.per_game:.1f} points per game** after halftime "
                                f"({r.share:.0%} of the drop).")
                    st.badge(r.status, color=STATUS_BADGE[r.status])
                    st.write(f"{r.detail} Last season this bucket was {signed(r.prev_per_game)} per game.")

    # ---------------------------------------------------------- 2. what's hiding it
    if helps:
        st.header("2. What's hiding how bad it is", divider="gray")
        total_help = sum(r.per_game for r in helps)
        residual = att_now.set_index("factor").loc["residual", "per_game"] if att_now is not None else 0.0
        for r in helps:
            st.markdown(f"**{r.name}** got *better* after halftime: {r.detail[0].lower() + r.detail[1:]} That's worth "
                        f"**{signed(r.per_game)} points per game** in their favour.")
        if hl:
            st.markdown(f"Without that, their second halves would be about **{signed(hl['h2_per_game'] - total_help)} "
                        f"per game** instead of {signed(hl['h2_per_game'])}. (A further {signed(residual)} per game "
                        "can't be placed in any one bucket.)")

    # ---------------------------------------------------------- 3. theories
    st.header(f"{3 if helps else 2}. The popular theories, checked", divider="gray")
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.subheader("Is it Garland?")
        if "H3" in res:
            k3 = res["H3"]["key"]
            share = k3["share_of_swing"]
            fig = go.Figure()
            for x0, x1, label, shade in ((0, CONTRIBUTES, "minor", 0.05), (CONTRIBUTES, DRIVES, "part of it", 0.12),
                                         (DRIVES, 1.0, "the main cause", 0.22)):
                fig.add_shape(type="rect", x0=x0, x1=x1, y0=-0.5, y1=0.5, line_width=0,
                              fillcolor=f"rgba(227,73,72,{shade})", layer="below")
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
        for start in range(0, len(cards), 2):
            cols = st.columns(2)
            for col, card in zip(cols, cards[start:start + 2]):
                with col.container(border=True, height="stretch"):
                    st.markdown(f'<div class="big-q">{card.question}</div>', unsafe_allow_html=True)
                    st.badge(card.answer, color=TONE_BADGE[card.tone][0])
                    st.write(card.text)

    # ---------------------------------------------------------- 4. since last season
    section = 4 if helps else 3
    if not data.comparison.empty and season_prev in data.attribution:
        st.header(f"{section}. What changed since last season", divider="gray")
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
        st.markdown(f"- **New this season:** {', '.join(new) or 'nothing'}\n"
                    f"- **Same as last season:** {', '.join(same) or 'nothing'}\n"
                    f"- **Better than last season:** {', '.join(better) or 'nothing'}")
        rows = S.scorecard(sh, season_now, season_prev)
        table = pd.DataFrame([{
            "": r.stat, f"{season_prev} 1st half": r.prev_h1, f"{season_prev} 2nd half": r.prev_h2,
            f"{season_now} 1st half": r.cur_h1, f"{season_now} 2nd half": r.cur_h2,
            "This season's 2nd-half drop-off": VERDICT_MARK[r.verdict]} for r in rows])
        st.dataframe(table, hide_index=True, width="stretch", height=38 + 35 * len(table))
        st.caption("The last column compares how much each stat changes from 1st to 2nd half this season with the "
                   "same change last season. Opponent free throws include late fouls by a team already behind; "
                   "without those, the rise is about the same as last season (see 'Is it fouling?').")

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
        section += 1

    # ---------------------------------------------------------- 5. what if
    if hl and reasons:
        st.header(f"{section}. What if they fixed it?", divider="gray")
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
        st.caption("This is bookkeeping, not a prediction: the buckets add up exactly, so swapping one for last "
                   "season's value shows its size, but real games don't change one thing at a time.")
        section += 1

    # ---------------------------------------------------------- 6. every stat
    st.header(f"{section}. Every stat, before and after halftime", divider="gray")
    st.dataframe(D.full_stat_sheet(data.ranks, season_now), hide_index=True, width="stretch",
                 height=38 + 35 * len(D.full_stat_sheet(data.ranks, season_now)))
    st.caption("League ranks: 1st = best of 30 teams. The change rank compares each team's second half with its "
               "own first half. These values come from the league's own team stats, so every team is measured the "
               "same way; they can differ slightly from the box-score totals above. Data: stats.nba.com via the open-source nba_api package; not affiliated with the "
               "NBA or the Clippers. Full method: reports/findings.md.")

# =================================================================== games
with tab_games:
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
                             marker=dict(color=H1C, size=11, line=dict(color="white", width=1.5)),
                             hovertemplate="%{x}<br>At halftime: %{y:+d}<extra></extra>"))
    fig.add_trace(go.Scatter(x=xs, y=strip["final_margin"], mode="markers", name="Final",
                             marker=dict(color=H2C, size=11, symbol="diamond", line=dict(color="white", width=1.5)),
                             hovertemplate="%{x}<br>Final: %{y:+d}<extra></extra>"))
    fig.add_hline(y=0, line=dict(color=GRAY, width=1))
    if len(strip) > 20:
        fig.update_xaxes(showticklabels=False)
    top_r.markdown("**Every game: where they stood at halftime, and how it ended.** Red lines = blown early leads.")
    with top_r:
        show(fig, 230, yaxis_title="Clippers margin")

    labels = [f"{x} · {r}" for x, r in zip(xs, strip["result"])]
    default = next((i for i in range(len(strip) - 1, -1, -1) if strip["lead_class"].iloc[i] == "blown_lead"),
                   len(strip) - 1)
    if len(labels) <= 16:
        pick = st.pills("Pick a game", labels, default=labels[default], key=f"pick_{season}")
    else:
        pick = st.selectbox("Pick a game", labels, index=default, key=f"pick_{season}")
    pick = pick or labels[default]
    gid = strip["game_id"].iloc[labels.index(pick)]
    row = leads.loc[gid]
    halves = data.game_halves[(data.game_halves["game_id"] == gid)
                              & data.game_halves["split"].astype(str).isin(["H1", "H2"])]
    att = D.game_attribution(data.game_halves, gid)

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
    fig.add_vrect(x0=24, x1=end, fillcolor="rgba(137,135,129,0.08)", line_width=0, layer="below",
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
                marker=dict(symbol=symbol, color=color, size=size, line=dict(color="white", width=1.2)),
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
    fig.update_xaxes(range=[0, end], tickvals=[0, 12, 24, 36, 48] + [50.5 + 5 * i for i in range(n_ot)],
                     ticktext=["Start", "Q2", "Half", "Q4", "End of Q4"]
                     + [("OT" if i == 0 else f"{i + 1}OT") for i in range(n_ot)])
    lim = max(10, float(m["margin"].abs().max()) + 4)
    show(fig, 430, yaxis=dict(title="Clippers ahead ↑ / behind ↓", range=[-lim, lim]))
    st.caption("Above the middle line = Clippers winning. Press ▶ Replay to watch the game unfold. Hover any "
               "turnover to see who, what kind, and what it cost.")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown("**Before vs after halftime**")
        st.dataframe(D.game_half_table(data.game_halves, gid), hide_index=True, width="stretch",
                     column_config={"stat": st.column_config.TextColumn("", width="medium")})
    with c2:
        st.markdown("**Where this game's second-half swing came from**")
        a = att.set_index("factor")
        fig = go.Figure(go.Bar(
            x=a["points"], y=[S.FACTOR_PLAIN[f][0] for f in a.index], orientation="h",
            marker_color=[GRAY if f == "residual" else GOOD if v >= 0 else BAD for f, v in a["points"].items()],
            customdata=[S.FACTOR_PLAIN[f][1] for f in a.index],
            hovertemplate="<b>%{y}</b>: %{x:+.1f} pts<br>%{customdata}<extra></extra>"))
        fig.update_yaxes(autorange="reversed")
        fig.add_vline(x=0, line=dict(color=GRAY, width=1))
        show(fig, 300, xaxis_title=f"Points vs the first half (total {signed(a['margin_change'].iloc[0], 0)})")
    with st.expander("Every Clippers turnover in this game"):
        st.dataframe(D.game_turnovers(data.turnovers, gid), hide_index=True, width="stretch")

# ================================================================= players
with tab_players:
    season_p = st.selectbox("Season", data.seasons, index=0, key="player_season")
    ph = D.player_halves(data.player_stats, season_p)
    if ph.empty:
        st.info("No players with enough minutes in both halves yet.")
    else:
        h2 = ph[ph["split"] == "H2"].sort_values("tov_share")
        st.subheader("Who's losing the ball after halftime?")
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
        show(fig, 80 + 46 * len(h2))
        st.caption("If a player's turnover bar is longer than his plays bar, he gives the ball away more than his "
                   "share. Players with 40+ minutes in each half.")

        st.subheader("Who gets sloppier after halftime?")
        w = ph.pivot(index="player_name", columns="split", values="tov_pct").sort_values("H2")
        tov = ph.pivot(index="player_name", columns="split", values="tov").reindex(w.index)
        fig = go.Figure()
        for name in w.index:
            fig.add_trace(go.Scatter(x=[w.loc[name, "H1"], w.loc[name, "H2"]], y=[name, name], mode="lines",
                                     line=dict(color=LIGHT, width=3), hoverinfo="skip", showlegend=False))
        for split, color, label in (("H1", H1C, "1st half"), ("H2", H2C, "2nd half")):
            fig.add_trace(go.Scatter(
                x=w[split], y=w.index, mode="markers", name=label, customdata=tov[split],
                marker=dict(color=color, size=13, line=dict(color="white", width=1.5)),
                hovertemplate="%{y}<br>%{x:.1%} of plays end in a turnover<br>%{customdata:.0f} turnovers"
                              "<extra>" + label + "</extra>"))
        fig.update_xaxes(tickformat=".0%", title="Turnover rate (share of a player's plays that end in a turnover)")
        fig.update_yaxes(tickvals=list(w.index), ticktext=[f"<b>{n}</b>" if n == focus else n for n in w.index])
        show(fig, 80 + 40 * len(w))
        st.caption("A long line pointing right = that player turns it over more after halftime. Watch the counts "
                   "on hover: a big jump on very few plays can be a couple of bad possessions.")

        st.subheader("One player, game by game")
        names = sorted(ph["player_name"].unique())
        player = st.selectbox("Player", names, index=names.index(focus) if focus in names else 0)
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
with tab_league:
    st.subheader("Does every team get worse after halftime?")
    st.write("No. Most teams barely change. Pick a stat to see how much each team's second half differs from its "
             "first, with every chart flipped so that **right = better after halftime**.")
    stat_names = {"net_rtg": "Overall (net rating)", "efg_pct": "Shooting accuracy",
                  "opp_efg_pct": "Opponents' shooting", "tov_pct": "Turnovers",
                  "opp_ft_rate": "Fouling (opponent free throws)", "opp_oreb_pct": "Second chances given up"}
    season_l = st.selectbox("Season", data.seasons, index=0, key="league_season")
    metric = st.pills("Stat", list(stat_names), default="net_rtg", format_func=stat_names.get, key="league_metric")
    metric = metric or "net_rtg"
    lv = D.league_view(data.ranks, season_l, metric, "change")
    flip = D.better_direction(metric) == "lower is better"
    lv = lv.assign(shown=-lv["value"] if flip else lv["value"]).sort_values("shown")
    pct = metric not in ("net_rtg", "tov_pct")
    fmt = (lambda v: f"{v * 100:+.1f} pts") if pct else (lambda v: f"{v:+.1f}")
    fig = go.Figure(go.Bar(
        x=lv["shown"], y=lv["team_name"], orientation="h",
        marker_color=[FOCUS if t else "#c3c2b7" for t in lv["is_team"]],
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
        st.write(f"From first half to second half, the Clippers change by **{fmt(row['value'])}** here. The "
                 f"league average change is {fmt(row['league_mean'])}.")
        st.markdown("**Clippers' rank on everything** (1st = best)")
        cr = D.clippers_ranks(data.ranks, season_l)
        st.dataframe(cr.map(ordinal), width="stretch", height=38 + 35 * len(cr))
