"""Blown leads: why they lose leads after halftime, ranked, with the theories checked.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from courtside.analysis.hypotheses import CONTRIBUTES
from courtside.analysis.hypotheses import DRIVES
from courtside.viz import dashboard_data as D
from courtside.viz import graphics as G
from courtside.viz import story as S
from courtside.viz import whatnow as W
from courtside.web.context import current
from courtside.web.theme import BAD, FOCUS, GOOD, GRAY, H1C, H2C, INK_MARK, MUTED_BAR, signed
from courtside.web.ui import TOC_LABELS, answer_cards, compare_panel, fill_toc, glossary_button, heading, next_step, picture, section, show, toc_slot


STATUS_BADGE = {S.NEW: "orange", S.SAME: "violet", S.SIMILAR: "gray", S.BETTER: "blue"}


VERDICT_MARK = {"Worse than last season": "▼ Worse than last season", "Better than last season": "▲ Better than last season",
                "About the same": "= About the same"}


def page_answer() -> None:
    ctx = current()
    TOC = ctx.toc
    data = ctx.data
    focus = ctx.focus
    focus_last = ctx.focus_last
    season_now = ctx.season_now
    season_prev = ctx.season_prev
    settings = ctx.settings
    PAGE_LEAGUE = ctx.pages["league"]
    PAGE_PLAYERS = ctx.pages["players"]
    PAGE_REPLAY = ctx.pages["replay"]
    PAGE_WHATNOW = ctx.pages["whatnow"]
    hl = S.headline(data.analysis, data.comparison, data.games, season_now) if not data.comparison.empty else None
    att_now, att_prev = data.attribution.get(season_now), data.attribution.get(season_prev)
    res = {r["id"]: r for r in data.analysis.get("results", [])}
    sh = D.season_halves(data.game_halves)
    reasons = (S.ranked_reasons(att_now, att_prev, sh.loc[season_now])
               if att_now is not None and att_prev is not None else [])
    top = reasons[0].factor if reasons else None
    last_game = pd.to_datetime(data.game_leads.loc[data.game_leads["season"] == season_now, "game_date"]).max()
    title = ("They're fine for a half. Then the shots stop falling." if top == "shooting"
             else "They're fine for a half. Then it falls apart.")
    with st.container(key="band_light_answer"):
        st.markdown(f'<div class="kicker">LA Clippers · {season_now} season · through {last_game:%b} '
                    f'{last_game.day}</div><div class="hero">{title}</div>', unsafe_allow_html=True)
        if hl:
            garland_main = "H3" in res and res["H3"]["key"]["share_of_swing"] >= DRIVES
            st.markdown(
                f'<div class="lede">{hl["wins"]}-{hl["losses"]}, and {hl["blown"]} of {hl["leads"]} early leads '
                f"blown. Is it {focus}'s turnovers? "
                + ("Mostly, yes." if garland_main else "Partly, but not mainly.")
                + " Click any card to see the chart behind it.</div>", unsafe_allow_html=True)
            glossary_button()
    if hl:
        with st.container(key="band_navy_answer"):
            h_ = D.season_halves(data.game_halves).loc[season_now]
            n_g = max(int(h_.loc["H1", "games"]), 1)
            court_col, num_col = st.columns([1.55, 1], gap="medium", vertical_alignment="center")
            with court_col:
                picture(G.court_halves(hl["h1_per_game"], hl["h2_per_game"]))
            num_col.markdown(compare_panel("Half by half", [
                ("Score margin, a game", signed(hl["h1_per_game"]), signed(hl["h2_per_game"]),
                 hl["h1_per_game"], hl["h2_per_game"], True, -10, 10, "first half → second half"),
                ("Shooting accuracy (eFG%)", f'{h_.loc["H1", "efg_pct"]:.0%}', f'{h_.loc["H2", "efg_pct"]:.0%}',
                 h_.loc["H1", "efg_pct"], h_.loc["H2", "efg_pct"], True, 0.40, 0.65, ""),
                ("Three-point shooting", f'{h_.loc["H1", "fg3_pct"]:.0%}', f'{h_.loc["H2", "fg3_pct"]:.0%}',
                 h_.loc["H1", "fg3_pct"], h_.loc["H2", "fg3_pct"], True, 0.25, 0.50, ""),
                ("Turnovers, a game", f'{h_.loc["H1", "tov_all"] / n_g:.1f}', f'{h_.loc["H2", "tov_all"] / n_g:.1f}',
                 h_.loc["H1", "tov_all"] / n_g, h_.loc["H2", "tov_all"] / n_g, False, 4, 14,
                 f'opponents score {h_.loc["H1", "opp_pts_off_tov"] / n_g:.1f} → '
                 f'{h_.loc["H2", "opp_pts_off_tov"] / n_g:.1f} a game off them'),
            ]), unsafe_allow_html=True)
            # The court above already shows the half-by-half margin, and section 3 lists what's new.
            answer_cards([c_ for c_ in S.answer_page_cards(hl, reasons, res, season_prev)
                          if not c_.title.startswith(("Fine before", "New problems"))])

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
    nav_ = toc_slot("answer")

    # ---------------------------------------------------------- 1. ranked reasons (costs and helps)
    total_help = sum(r.per_game for r in helps)
    section("1. Why they blow leads, ranked", "reasons",
            "Each part of the game, after halftime: <b>orange = this season</b>, gray = last season. Left of zero "
            "costs them points; right of zero helps." if reasons else "")
    if reasons:
        ordered = hurts + sorted(helps, key=lambda r: -r.per_game)
        names = [f"{i}. {r.name}" for i, r in enumerate(ordered, 1)]
        fig = go.Figure()
        for label, vals, color in ((f"This season ({season_now})", [r.per_game for r in ordered], BAD),
                                   (f"Last season ({season_prev})", [r.prev_per_game for r in ordered], MUTED_BAR)):
            fig.add_trace(go.Bar(
                y=names, x=vals, orientation="h", name=label, marker_color=color,
                # the top row names its series right on the bar (the legend stays too)
                text=[f"{signed(v)} a game" + (f" · {label.split(' (')[0].lower()}" if i == 0 else "")
                      for i, v in enumerate(vals)], textposition="outside", cliponaxis=False,
                customdata=[r.detail for r in ordered],
                hovertemplate="<b>%{y}</b><br>" + label + ": %{x:+.1f} points a game after halftime"
                              "<br>%{customdata}<extra></extra>"))
        lo = min(min(r.per_game, r.prev_per_game) for r in ordered)
        hi = max(max(r.per_game, r.prev_per_game) for r in ordered)
        fig.add_vline(x=0, line=dict(color=INK_MARK, width=1))
        fig.update_layout(barmode="group", bargap=0.3, bargroupgap=0.08)
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(title="Points a game after halftime  (← costs them · helps them →)",
                         range=[min(lo, 0) * 1.45 - 0.8, max(hi, 0) * 1.45 + 1.8], zeroline=False)
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

    # ---------------------------------------------------------- where the shots stop falling
    SHOTS_SAID = {"At the rim": "Shots at the rim", "Short range": "Short-range shots",
                  "Mid-range": "Mid-range shots", "Three": "Threes"}
    if len(data.shots) and (data.shots["season"] == season_now).any():
        z = D.shot_zones(data.shots, season_now)
        ch = (z["mean_h2"] - z["mean_h1"]).sort_values()
        worst, steady = ch.index[0], ch.abs().idxmin()
        st.subheader("Where the shots stop falling", anchor="shots")
        TOC.append((TOC_LABELS["shots"], "shots"))
        st.markdown(
            f'<div class="takeaway"><b>{SHOTS_SAID[worst]}</b> go from '
            f'{z.loc[worst, "mean_h1"]:.0%} before halftime to <b>{z.loc[worst, "mean_h2"]:.0%}</b> after'
            + (f"; {SHOTS_SAID[steady].lower()} hold about steady ({z.loc[steady, 'mean_h1']:.0%} → "
               f"{z.loc[steady, 'mean_h2']:.0%})." if steady != worst else ".") + "</div>",
            unsafe_allow_html=True)
        cur = data.shots[data.shots["season"] == season_now]
        pts = lambda h: list(zip(cur.loc[cur["half"] == h, "loc_x"], cur.loc[cur["half"] == h, "loc_y"],  # noqa: E731
                                 cur.loc[cur["half"] == h, "made"]))
        zones = [(zn, r.mean_h1, r.mean_h2, int(r.size_h1), int(r.size_h2)) for zn, r in z.iterrows()]
        n_g = max(cur["game_id"].nunique(), 1)
        map_col, zone_col = st.columns([2.35, 1], gap="medium", vertical_alignment="center")
        with map_col:
            picture(G.shot_map(pts("H1"), pts("H2"), zones))
        zone_col.markdown(compare_panel("Zone by zone", [
            (zn, f"{r.mean_h1:.0%}", f"{r.mean_h2:.0%}", r.mean_h1, r.mean_h2, True, 0.2, 0.75,
             f"{r.size_h1 / n_g:.1f} → {r.size_h2 / n_g:.1f} shots a game") for zn, r in z.iterrows()]),
            unsafe_allow_html=True)
        with st.expander("Details"):
            st.markdown(
                "- Every Clippers field-goal attempt this season from the NBA's shot charts; overtime left out. "
                "The counts match the box scores half by half.\n"
                "- Zones: *at the rim* is the restricted area, *short range* the rest of the paint, *three* "
                "includes the corners.\n- "
                + "; ".join(f"{zn}: {int(r.size_h1)} shots before halftime, {int(r.size_h2)} after"
                            for zn, r in z.iterrows()) + ".")

    # ---------------------------------------------------------- 3. theories
    section("2. The popular theories, checked", "theories")
    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.subheader(f"Is it {focus_last}?")
        if "H3" in res:
            k3 = res["H3"]["key"]
            share = k3["share_of_swing"]
            fig = go.Figure()
            for x0, x1, label, shade in ((0, CONTRIBUTES, "minor", 0.08), (CONTRIBUTES, DRIVES, "part of it", 0.18),
                                         (DRIVES, 1.0, "the main cause", 0.32)):
                fig.add_shape(type="rect", x0=x0, x1=x1, y0=-0.5, y1=0.5, line_width=0,
                              fillcolor=f"rgba(217,89,38,{shade})", layer="below")
                fig.add_annotation(x=(x0 + x1) / 2, y=0.72, text=label, showarrow=False, font=dict(size=12))
            fig.add_trace(go.Bar(x=[min(share, 1)], y=[0], orientation="h", width=0.34, marker_color=FOCUS,
                                 hovertemplate=f"{focus_last}'s extra turnovers explain {share:.0%} of the "
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
                "halftime at home at all. "
                + ("They do, by more than chance would explain." if wall.big == "Maybe" else "They don't.") + "\n"
                "- Includes late free throws from intentional fouls.\n"
                "- The home-court study looks at every arena since 2013-14: visitors' free-throw shooting at "
                "Intuit Dome isn't consistently worse than at other arenas.")

    # ---------------------------------------------------------- 3. since last season
    section_n = 3
    if not data.comparison.empty and season_prev in data.attribution:
        heading(f"{section_n}. What changed since last season", "changes")
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
        heading(f"{section_n}. What if they fixed it?", "whatif")
        st.write("Switch a problem back to last season's level and see what their second halves would look like.")
        fixable = [r for r in hurts if r.prev_per_game - r.per_game >= 0.1]
        left, right = st.columns([1, 1.2], gap="large")
        with left:
            fixed = {r.factor for r in fixable
                     if st.toggle(f"{r.name} back to last season ({signed(r.prev_per_game)} instead of "
                                  f"{signed(r.per_game)} per game)", key=f"fix_{r.factor}")}
            g_pts = res["H3"]["key"]["excess_pts"] / max(hl["games"], 1) if "H3" in res else 0.0
            fix_g = st.toggle(f"{focus_last}'s turnovers back to his last-season rate (about {signed(g_pts)} per game)",
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
    heading(f"{section_n}. Every stat, before and after halftime", "stats")
    with st.expander("Open the full stat sheet", icon=":material/table:"):
        sheet = D.full_stat_sheet(data.ranks, season_now)
        st.dataframe(sheet, hide_index=True, width="stretch", height=38 + 35 * len(sheet))
    st.caption("League ranks: 1st = best of 30 teams. The change rank compares each team's second half with its "
               "own first half. These values come from the league's own team stats, so every team is measured the "
               "same way; they can differ slightly from the box-score totals above. Data: stats.nba.com via the open-source nba_api package; not affiliated with the "
               "NBA or the Clippers. Full method: reports/findings.md.")
    fill_toc(nav_)
    next_step(PAGE_PLAYERS, "Who is losing the ball after halftime, player by player.")
