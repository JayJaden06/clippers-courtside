"""What now?: what the numbers point to, as verdict cards and the evidence behind each.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from courtside.analysis.hypotheses import DRIVES
from courtside.names import short
from courtside.viz import dashboard_data as D
from courtside.viz import graphics as G
from courtside.viz import whatnow as W
from courtside.web.context import current
from courtside.web.theme import BAD, BG, FOCUS, GOOD, GRAY, H1C, H2C, INK_MARK, LIGHT, MUTED_BAR, signed
from courtside.web.ui import PHOTO_COL, answer_cards, axis_faces, face, fill_toc, heading, legend_only, page_top, picture, section, show, toc_slot, with_photos


def page_whatnow() -> None:
    ctx = current()
    data = ctx.data
    focus = ctx.focus
    focus_last = ctx.focus_last
    meta = ctx.meta
    season_now = ctx.season_now
    season_prev = ctx.season_prev
    settings = ctx.settings
    n_games = int(meta.get("games", 0))
    n = max(n_games, 1)
    page_top("What needs to happen now",
             f"What the numbers point to after {n_games} games. Click any card to see the chart behind it. "
             "Evidence, not a coaching plan.")
    nav_ = toc_slot("whatnow")
    res = {r["id"]: r for r in data.analysis.get("results", [])}
    lg = data.ranks[(data.ranks["season"] == season_now) & (data.ranks["split"] == "H2")
                    & (data.ranks["metric"] == "efg_pct")]["league_mean"]
    league_h2 = float(lg.iloc[0]) if len(lg) else float("nan")
    team = settings.team_abbreviation
    sd = W.shooting_drop(data.player_box, season_now, team, league_h2)
    sf = W.starting_five(data.player_box, data.lineups, season_now, team)
    bl = W.ball_load(data.player_box, data.lineups, res["H3"]["key"], season_now, team, focus, n) if "H3" in res else None
    back = (W.returning(data.roster, data.player_box, data.games, data.history, season_now, team,
                        settings.watch_players) if len(data.roster) else settings.watch_players)
    cands = W.candidates(data.history, data.player_box, season_now, team, back, league_h2)
    rot = W.rotation(data.player_box, data.onoff, data.bio, season_now, team) if len(data.onoff) else None
    tracked = len(data.matchups) and len(data.bio)
    dfn = W.defenders(data.matchups, data.hustle, data.bio, season_now, team) if tracked else None
    gl = W.glass(data.game_halves, season_now, season_prev)
    ids = data.player_box.loc[data.player_box["player_name"] == focus, "player_id"]
    pr = (W.pressure(data.matchups, data.turnovers, data.bio, season_now, season_prev, team, settings.team_id,
                     int(ids.iloc[0])) if tracked and len(ids) and "stealer_id" in data.turnovers else None)
    chem = W.chemistry(data.player_box, data.lineups, season_now, season_prev, team)

    with st.container(key="band_navy_whatnow"):
        answer_cards(W.verdicts(sf, bl, cands, sd, focus, rot, dfn, gl, pr, n_games))
    with st.expander("Read it as text instead"):
        st.markdown("\n".join(f"- {line}" for line in W.summary_lines(sf, bl, cands, sd, focus, rot, dfn, gl, pr)),
                    unsafe_allow_html=True)

    # ------------------------------------------------------------ minutes
    if rot is not None and len(rot):
        r = rot[rot["h2_min_on"] > 0].sort_values("h2_mpg").copy()
        losing = int((r["h2_pm_pg"] < 0).sum())
        earned = list(r.loc[r["earned"], "player_name"].map(lambda x: short(x)))
        section("Who plays after halftime, and how it goes", "minutes",
                f"The Clippers lose the second half with <b>{losing} of {len(r)}</b> regulars on the floor."
                + (f" {' and '.join(earned)} {'is' if len(earned) == 1 else 'are'} the exception." if earned else ""))
        tag = np.where(r["struggling"], BAD, np.where(r["earned"], GOOD, MUTED_BAR))
        too_few = ~r["struggling"] & ~r["earned"] & (r["h2_min_on"] < W.EARNED_MIN)
        labels = [f"{short(x)}{' (R)' if rk else ''} · {m:.0f} min"
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
        axis_faces(fig, [face(x) for x in r["player_name"]], 34)
        show(fig, 110 + 34 * len(r))
        st.caption("Sorted by second-half minutes a game (most at the top). (R) = rookie. Blue = the team does much "
                   "better with him on than off, even if the bar is small; orange = the team falls well behind with him "
                   "on. Open Details for the exact rules.")
        with st.expander("Details and the full minutes table"):
            st.markdown(f"- **Fewer late minutes** = the team is {abs(W.STRUGGLE_NET):.0f}+ points per 100 trips "
                        f"behind with him on after halftime, over {W.STRUGGLE_MIN}+ minutes.\n"
                        f"- **Earned more minutes** = the team is {W.EARNED_ON_OFF:.0f}+ points per 100 trips better "
                        f"with him on than off, over {W.EARNED_MIN}+ minutes, while he plays under {W.EARNED_MPG} a game.\n"
                        "- Small samples swing on a few possessions, and on-court numbers depend on who else is "
                        "out there.")
            st.dataframe(
                with_photos(rot[["player_name", "rookie", "gp", "mpg", "h2_mpg", "q4_mpg", "h2_pm_pg", "h2_on_off",
                                 "h2_reb36"]]),
                hide_index=True, width="stretch",
                column_config=PHOTO_COL | {
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
            names = [f"<b>{focus_last}</b>"] + [short(x) for x in safest["player_name"]]
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
            axis_faces(fig, [face(focus)] + [face(x) for x in safest["player_name"]], 33)
            st.markdown("**How often his plays end in a turnover**")
            show(fig, 250, showlegend=False)
        with right:
            if pr is not None and pr.tov:
                st.markdown("**Is he being double-teamed into them?**")
                picture(G.turnover_strip(pr.steals - pr.help, pr.help, pr.tov - pr.steals))
                st.caption((f"{pr.help_share:.0%} are stolen by a second defender, the way a double team usually "
                            f"ends: more than his teammates ({pr.mates_help_share:.0%})." if pr.trapped else
                            f"Only {pr.help_share:.0%} are stolen by a second defender, the way a double team "
                            f"usually ends (teammates: {pr.mates_help_share:.0%}).")
                           + " Double teams aren't logged publicly; this is the closest proxy.")
        if pr is not None:
            if pr.length_problem:
                size_head = ("Bigger defenders are still a problem." if pr.size_gap_last_season
                             else "Bigger defenders are the new problem.")
            else:
                size_head = "Bigger defenders aren't a problem right now."
            st.markdown(f"**{size_head}** How often {focus_last} loses the ball, by the height of the player "
                        "guarding him:")
            me = data.bio[data.bio["player_name"] == focus].sort_values("season").tail(1)
            ht = float(me["height_in"].iloc[0]) if len(me) else 73.0
            wt = float(me["weight_lb"].iloc[0]) if len(me) else 190.0
            draw_col, who_col = st.columns([1.6, 1], gap="medium", vertical_alignment="center")
            with draw_col:
                picture(G.height_matchup(focus_last, ht, wt, 100 / pr.tov100_long, 100 / pr.tov100_short,
                                         100 / pr.prior_tov100_long, 100 / pr.prior_tov100_short, W.LONG_IN,
                                         photo=face(focus)))
            rows_ = "".join(
                f'<div class="who-row"><img src="{face(x.def_player) or ""}" alt=""><div class="nm"><b>'
                f'{x.def_player}</b><br><span>{x.def_team} · '
                f'{W._ft_in(x.height_in) if x.height_in == x.height_in else "?"}</span></div>'
                f'<div class="n">{x.tov:.0f}</div></div>' for x in pr.forcers.head(6).itertuples())
            who_col.markdown(f'<div class="cmp"><h4>Who forces them</h4><div class="cmp-sub" style="margin-bottom:'
                             f'.3rem">The defenders credited with the most of {focus_last}\'s turnovers this '
                             f"season (player tracking).</div>{rows_}</div>", unsafe_allow_html=True)
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
            who = short(r_["player_name"])
            st.markdown(f"**{who} gives up size{' as a rookie' if r_['rookie'] else ''}**")
            picture(G.size_matchup(who, r_["own_ht"], r_["own_wt"], r_["opp_ht"], r_["opp_wt"], r_["fg_pct"],
                                   dfn.team_fg, photo=face(r_["player_name"])))
    left = right = st.container()  # one column: the shooting chart, then the size table
    with left:
        if dfn is not None and len(dfn.table):
            t = dfn.table.sort_values("fg_pct", ascending=True)
            st.markdown("**How well opponents shoot when he's the defender**")
            fig = go.Figure(go.Bar(
                y=t["player_name"].map(short) + np.where(t["rookie"], " (R)", ""), x=t["fg_pct"],
                orientation="h", marker_color=[BAD if a_ else MUTED_BAR for a_ in t["attacked"]],
                text=[f"{v:.0%}" for v in t["fg_pct"]], textposition="outside",
                customdata=np.column_stack([t["player_name"], t["fga"]]),
                hovertemplate="%{customdata[0]}: opponents shoot %{x:.1%} on %{customdata[1]:.0f} shots"
                              "<extra></extra>"))
            fig.add_vline(x=dfn.team_fg, line=dict(color=INK_MARK, width=1, dash="dash"),
                          annotation_text=f"team {dfn.team_fg:.0%}", annotation_position="top")
            fig.update_xaxes(tickformat=".0%", range=[0, float(t["fg_pct"].max()) * 1.25])
            axis_faces(fig, [face(x) for x in t["player_name"]], 30)
            show(fig, 90 + 30 * len(t), showlegend=False)
            st.caption(f"Orange = opponents shoot {W.ATTACKED_FG:.0%}+ better on him than on the team. (R) = rookie.")
    with right:
        if dfn is not None and len(dfn.table):
            t = dfn.table.sort_values("wt_gap")
            with st.expander("How big are the players each one guards?"):
                fig = go.Figure()
                for r_ in t.itertuples():
                    fig.add_trace(go.Scatter(x=[r_.own_wt, r_.opp_wt], y=[short(r_.player_name)] * 2,
                                             mode="lines", hoverinfo="skip", showlegend=False,
                                             line=dict(color=FOCUS if r_.gives_up_size else LIGHT, width=4)))
                y = [short(x) for x in t["player_name"]]
                fig.add_trace(go.Scatter(x=t["own_wt"], y=y, mode="markers", name="His weight",
                                         marker=dict(color=H1C, size=12, line=dict(color=BG, width=1.5)),
                                         hovertemplate="%{y}: %{x:.0f} lb<extra>his weight</extra>"))
                fig.add_trace(go.Scatter(x=t["opp_wt"], y=y, mode="markers", name="Who he guards (average)",
                                         marker=dict(color=FOCUS, size=12, line=dict(color=BG, width=1.5)),
                                         hovertemplate="%{y} guards players averaging %{x:.0f} lb<extra></extra>"))
                fig.update_xaxes(title="Weight (lb)")
                axis_faces(fig, [face(x) for x in t["player_name"]], 30)
                show(fig, 90 + 30 * len(t))
                st.caption("Orange line = guards players 25+ lb heavier than himself.")
    if dfn is not None and len(dfn.table):
        with st.expander("Details and effort stats (contests, deflections, box-outs)"):
            for r_ in dfn.table[dfn.table["gives_up_size"]].itertuples():
                st.markdown(f"- **{r_.player_name}** ({W._ft_in(r_.own_ht)}, {r_.own_wt:.0f} lb"
                            f"{', rookie' if r_.rookie else ''}): guards players averaging {r_.opp_wt:.0f} lb; "
                            f"{r_.def_box_outs:.0f} defensive box-outs in {r_.hustle_min:.0f} minutes.")
            st.dataframe(
                with_photos(dfn.table[["player_name", "rookie", "hustle_min", "contested_shots_36", "deflections_36",
                                       "def_box_outs_36", "fg_pct", "shooting_fouls"]].sort_values(
                    "hustle_min", ascending=False)),
                hide_index=True, width="stretch",
                column_config=PHOTO_COL | {
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

    # ------------------------------------------------------------ returning players
    shooters = [c for c in cands if c.fg3a >= W.THREE_MIN_ATT]
    section("Will the returning players help?", "returning",
            ("The returning shooters hit " + ", ".join(f"<b>{c.fg3_pct:.0%}</b>" for c in shooters)
             + f" of their threes after halftime. The Clippers hit {sd.team_fg3_h2:.0%} now.") if shooters else "")
    left, right = st.columns(2, gap="large")
    with left:
        if shooters:
            xs = [short(c.name) for c in shooters] + ["Clippers now"]
            ys = [c.fg3_pct for c in shooters] + [sd.team_fg3_h2]
            fig = go.Figure(go.Bar(x=xs, y=ys, marker_color=[FOCUS] * len(shooters) + [MUTED_BAR],
                                   text=[f"{v:.0%}" for v in ys], textposition="outside",
                                   hovertemplate="%{x}: %{y:.1%} from three after halftime<extra></extra>"))
            fig.update_yaxes(tickformat=".0%", range=[0, max(ys) * 1.3])
            axis_faces(fig, [face(c.name) for c in shooters] + [D.logo_url(settings.team_id, dark=False)], 40, axis="x")
            st.markdown("**Three-point shooting after halftime**")
            show(fig, 300, showlegend=False)
    with right:
        bigs = [c for c in cands if "rebounding" in c.helps]
        if bigs and rot is not None:
            cur = rot[(rot["h2_mpg"] * rot["gp"] >= 40) & ~rot["player_name"].isin([c.name for c in bigs])]
            top = cur.sort_values("h2_reb36", ascending=False).head(1)
            xs = [short(c.name) for c in bigs] + [f"{short(x)} (best now)"
                                                            for x in top["player_name"]]
            ys = [c.reb36 for c in bigs] + list(top["h2_reb36"])
            fig = go.Figure(go.Bar(x=xs, y=ys, marker_color=[FOCUS] * len(bigs) + [MUTED_BAR] * len(top),
                                   text=[f"{v:.1f}" for v in ys], textposition="outside",
                                   hovertemplate="%{x}: %{y:.1f} rebounds per 36 minutes after halftime"
                                                 "<extra></extra>"))
            fig.update_yaxes(range=[0, max(ys) * 1.3])
            axis_faces(fig, [face(c.name) for c in bigs] + [face(x) for x in top["player_name"]], 40, axis="x")
            st.markdown("**Rebounds per 36 minutes after halftime**")
            show(fig, 300, showlegend=False)
    chip_color = {"shooting": "#2e8b57", "threes": "#1d7dd1", "ball security": "#6b4fbb", "rebounding": "#c8102e"}
    roster = data.roster.set_index("player_name") if len(data.roster) else pd.DataFrame(columns=["number", "position"])
    rcards = []
    for c in cands:
        info = roster.loc[c.name] if c.name in roster.index else None
        num = f"#{info['number']}" if info is not None and str(info["number"]).strip() else ""
        status = ("Out so far" if c.this_season_minutes == 0 else f"Back: {c.this_season_minutes:.0f} min so far")
        chips = "".join(f'<span class="chip" style="background:{chip_color.get(h, "#7f7d77")}">{h.capitalize()}</span>'
                        for h in c.helps) + ('<span class="chip" style="background:#7f7d77">Small sample</span>'
                                             if c.small_sample else "")
        rcards.append(
            f'<div class="pcard"><div class="pc-top"><span class="pc-num">{num}</span>'
            f'<span class="pc-pos">{info["position"] if info is not None else ""}</span>'
            f'<img src="{face(c.name) or ""}" alt="" loading="lazy"></div><div class="pc-body">'
            f'<div class="pc-name">{c.name}</div><div class="pc-status">{status}'
            + (f" · {c.mpg:.0f} min a game in {c.sample.split(',')[0]}" if c.mpg == c.mpg else "")
            + f'</div><div class="chips">{chips}</div></div></div>')
    st.markdown(f'<div class="rcards" style="--cols:{min(max(len(rcards), 1), 6)}">{"".join(rcards)}</div>',
                unsafe_allow_html=True)
    with st.expander("Details"):
        st.markdown(f"- **Who's here:** everyone on today's roster who has missed {W.MISSED_SHARE:.0%} or more of "
                    f"the games since joining the team and played {W.PRIOR_MPG:.0f}+ minutes a game in a recent "
                    "season. The list rebuilds itself after every refresh. The public stats don't say *why* a "
                    "player sat (injury, rest or coach's decision), so it includes all three.\n"
                    "- Numbers for players who haven't played much this season come from their most recent season "
                    f"with {W.SMALL_SAMPLE_MIN}+ second-half minutes, on whatever team they played for.")
        st.dataframe(with_photos(pd.DataFrame([{
            "Player": c.name, "Sample": c.sample, "2nd-half min": c.h2_minutes, "Shooting (eFG%)": c.efg,
            "3P%": c.fg3_pct, "3PA": c.fg3a, "Turnover rate": c.tov_pct, "Rebounds/36": c.reb36} for c in cands],
            columns=["Player", "Sample", "2nd-half min", "Shooting (eFG%)", "3P%", "3PA", "Turnover rate",
                     "Rebounds/36"]), "Player"),
            hide_index=True, width="stretch",
            column_config=PHOTO_COL | {"2nd-half min": st.column_config.NumberColumn(format="%.0f"),
                           "Shooting (eFG%)": st.column_config.NumberColumn(format="percent"),
                           "3P%": st.column_config.NumberColumn(format="percent"),
                           "3PA": st.column_config.NumberColumn(format="%.0f"),
                           "Turnover rate": st.column_config.NumberColumn(format="percent"),
                           "Rebounds/36": st.column_config.NumberColumn(format="%.1f")})

    heading("More evidence", "more")
    st.markdown('<div class="takeaway">The starting five, chemistry, and who stopped shooting: the cards at the top already give the answer for each.</div>', unsafe_allow_html=True)
    with st.expander("Open the starting five, chemistry and shooting", icon=":material/expand_more:"):
        # ------------------------------------------------------------ starting five
        if sf is not None:
            rest = sf.rest_h2_pm / n
            section("Does the starting five need to change?", "starters",
                    f"No. The starters hold even after halftime; the other lineups lose "
                    f"<b>{abs(rest):.1f} points a game</b>." if sf.not_the_problem and rest < 0 else
                    f"Starters: {signed(sf.h2_pm / n)} a game after halftime; everyone else {signed(rest)}.",
                    ("Keep it", "blue") if sf.not_the_problem else ("Maybe change it", "orange"), toc=False)
            with st.expander("Details"):
                st.markdown(f"- Starters: {', '.join(sf.names)}; {sf.games_together} starts together.\n"
                            f"- They've played {sf.h2_min:.0f} second-half minutes together; other lineups "
                            f"{sf.rest_h2_min:.0f}.\n"
                            f"- Per 100 trips down the floor: starters {signed(sf.h2_net)} after halftime "
                            f"({signed(sf.h1_net)} before it), everyone else {signed(sf.rest_h2_net)}.\n"
                            "- Lineup numbers swing a lot on a few possessions; treat this as a direction.")

        # ------------------------------------------------------------ chemistry
        p = chem.pairs.sort_values("pm")
        best = p.iloc[-1] if len(p) else None
        section("Chemistry: which duos work?", "chemistry",
                f"<b>{chem.continuity:.0%}</b> of minutes go to players who were here last season."
                + (f" Best duo after halftime: <b>{short(best['a'])} + {short(best['b'])}</b>."
                   if best is not None else ""), toc=False)
        left, right = st.columns([1, 2.2], gap="large")
        with left:
            st.markdown("**How much of the team is new**")
            answer_cards([W.Verdict("Minutes from players who were here last season", f"{chem.continuity:.0%}",
                                    f"New this season: {', '.join(short(n) for n in chem.new[:5]) or 'nobody'}.",
                                    "info", "chemistry", ("share", chem.continuity, None))], link=False)
        with right:
            if len(p):
                show_p = pd.concat([p.head(5), p.tail(5)]).drop_duplicates(subset=["a_id", "b_id"]).sort_values("pm")
                labels = [f"{short(a)} + {short(b)}{' ✦' if new else ''}"
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

        # ------------------------------------------------------------ who stopped shooting
        section("Who stopped shooting after halftime?", "shooting",
                f"<b>{sd.n_worse} of {sd.n_players}</b> regulars shoot worse after halftime.", toc=False)
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
        axis_faces(fig, [face(x) for x in t["player_name"]], 40)
        show(fig, 90 + 40 * len(t))
        with st.expander("Details"):
            st.markdown("The biggest drops come from players who were unusually hot before halftime, so some of this "
                        "should even out. Players with 40+ minutes in each half.")

    # ------------------------------------------------------------ what would change this
    with st.expander("What would change this advice", icon=":material/update:"):
        watch = []
        if "H3" in res:
            k3 = res["H3"]["key"]
            watch.append(f"**{focus_last} stops being the main cause** if his extra turnovers fall from "
                         f"{k3['share_of_swing']:.0%} to under {DRIVES:.0%} of the second-half drop."
                         if k3["share_of_swing"] >= DRIVES else
                         f"**{focus_last} becomes the main cause** if his extra turnovers grow from "
                         f"{k3['share_of_swing']:.0%} to {DRIVES:.0%} of the second-half drop.")
        if "H1" in res:
            k1 = res["H1"]["key"]
            sure = k1["boot_lo"] > 0 or k1["boot_hi"] < 0
            watch.append(("**The collapse could still be bad luck** if the 90% range for the drop widens to "
                          "include zero again." if sure else
                          "**The collapse is confirmed (not bad luck)** once the 90% range for the drop stops "
                          "including zero.") + f" It's {signed(k1['boot_lo'])} to {signed(k1['boot_hi'])} today.")
        watch.append((f"**The shooting has recovered**: second-half shooting is above the league average "
                      f"({league_h2:.1%}). The problem is back if it dips under." if sd.team_h2 >= league_h2 else
                      f"**The shooting problem is fading** if second-half shooting climbs back above the league "
                      f"average ({league_h2:.1%}).") + f" It's {sd.team_h2:.1%} today.")
        if pr is not None:
            watch.append((f"**{focus_last} has adjusted to bigger defenders** once his turnovers against them "
                          f"drop under {W.LENGTH_RATIO}x his rate against shorter ones." if pr.length_problem else
                          f"**Bigger defenders become a problem for {focus_last}** if his turnovers against them "
                          f"climb to {W.LENGTH_RATIO}x his rate against shorter ones.")
                         + f" It's {pr.size_ratio:.1f}x today.")
        watch.append(("**The rebounding is fixed** when opponents get back under " if gl.worse else
                      "**The rebounding slips** if opponents get back more than ")
                     + f"{gl.prior_opp_oreb_h2 + W.GLASS_WORSE:.1%} of their misses after halftime. It's "
                     f"{gl.opp_oreb_h2:.1%} today.")
        st.markdown("\n".join(f"- {w}" for w in watch))
        st.caption("Everything on this page reruns with `courtside all` after each game. The rules behind each "
                   "verdict are listed at the top of src/courtside/viz/whatnow.py.")
    fill_toc(nav_)
