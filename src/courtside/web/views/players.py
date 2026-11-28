"""Players: who loses the ball after halftime, the rotation as cards, and one player game by game.
"""

from __future__ import annotations

import math
from html import escape
from urllib.parse import urlencode

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from courtside.names import short
from courtside.viz import dashboard_data as D
from courtside.viz import story as S
from courtside.web.context import current
from courtside.web.theme import BAD, GOOD, H1C, INK_MARK, MUTED_BAR
from courtside.web.ui import answer_cards, clicked, face, fill_toc, next_step, page_top, section, show, toc_slot, track


# A card click stays on the page: this script hands the player to Python and glides down to his games while
# they redraw. The link's address is the fallback (a new tab, or a click before the script has loaded).
CARD_CLICK_JS = """
// the "One player" pill's own glide: lands the heading just below the pinned bar, instant under reduced motion
const goToGames = () => {
  const pill = document.querySelector('nav.toc a[data-a="one"]');
  if (pill) pill.click(); else document.getElementById('one')?.scrollIntoView({ block: 'start' });
};

export default function ({ setTriggerValue }) {
  window.__pcPick = setTriggerValue;  // the latest mount's sender; the listener below is bound once
  if (window.__pcGlide) {  // this is the redraw a click asked for: the page above may have shifted, so re-aim once
    window.__pcGlide = false;
    setTimeout(() => {
      const h = document.getElementById('one');
      const n = document.querySelector('nav.toc');
      const bar = n ? n.getBoundingClientRect().bottom : 120;
      if (h && Math.abs(h.getBoundingClientRect().top - bar - 12) > 24) goToGames();
    }, 350);
  }
  if (window.__pcBound) return;
  window.__pcBound = true;
  document.addEventListener('click', (e) => {
    const a = e.target.closest && e.target.closest('a.pc-link[data-pick]');
    if (!a || e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
    e.preventDefault();
    window.__pcGlide = true;
    window.__pcPick('pick', a.dataset.pick);
    goToGames();
  });
}
"""
card_clicks = st.components.v2.component("player_card_clicks", js=CARD_CLICK_JS)


def _picked_on_card() -> None:
    name = (st.session_state.get("card_clicks") or {}).get("pick")
    if name:
        st.session_state["player_pick"] = name


def card_link(name: str, season: str) -> str:
    """A link laid over the whole card: a click picks this player and scrolls to his games."""
    href = escape("?" + urlencode({"pick": name, "season": season}))
    return (f'<a class="pc-link" href="{href}" target="_self" data-pick="{escape(name)}" '
            f'aria-label="See {escape(name)}&#39;s games"></a>')


def opened_from_card(seasons: list[str]) -> None:
    """A card click lands here once: pick the season and player it named, then clear the address bar."""
    qp = st.query_params
    if "pick" not in qp:
        return
    if qp.get("season") in seasons:
        st.session_state["player_season"] = qp["season"]
    st.session_state["player_pick"] = qp["pick"]
    st.session_state["player_jump"] = True
    del qp["pick"]
    qp.pop("season", None)


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
        rate = f"{r.tov_pct:.0%}"  # the page's one unit for "how often his plays end in a turnover"
        cards.append(
            f'<div class="pcard" title="{r.player_name}">{card_link(r.player_name, season)}'
            f'<div class="pc-top"><span class="pc-num">{num}</span>'
            f'<span class="pc-pos">{pos}</span><img src="{D.headshot_url(int(r.player_id))}" alt="" loading="lazy">'
            f'</div><div class="pc-body"><div class="pc-name">{r.player_name}</div><div class="pc-stats">'
            f'<div><b>{r.play_share:.0%}</b><span>of the plays after halftime</span></div>'
            f'<div><b>{rate}</b><span>of his plays end in a turnover</span></div></div></div></div>')
    # every card on show, in balanced rows (11 players = 6 + 5), never a sideways scroll
    per_row = math.ceil(len(cards) / math.ceil(len(cards) / 8)) if cards else 1
    st.markdown(f'<div class="pcards" style="--cols:{per_row}">{"".join(cards)}</div>', unsafe_allow_html=True)
    card_clicks(key="card_clicks", on_pick_change=_picked_on_card)


def game_table(g: pd.DataFrame) -> pd.DataFrame:
    """The game-by-game table in words: dates, results, and halves side by side, rates as percents."""
    when = pd.to_datetime(g["game_date"])
    # a half he sat out is blank, not "nan": count columns stay whole numbers that allow a gap
    whole = lambda c: g[c].round().astype("Int64")  # noqa: E731
    pct = lambda c: (100 * g[c]).round()  # noqa: E731  (numbers, so they sort; shown as "12%" by TABLE_FORMAT)
    return pd.DataFrame({
        "Date": when.dt.strftime("%b ") + when.dt.day.astype(str),
        "Opponent": g["opponent_abbreviation"],
        "Result": g["result"] + " " + g["final_margin"].map(lambda m: f"{m:+d}".replace("-", "−")),
        "Early lead": g["lead_class"],
        "Minutes, 1st half": whole("minutes_h1"),
        "Minutes, 2nd half": whole("minutes_h2"),
        "Turnovers, 1st half": whole("tov_h1"),
        "Turnovers, 2nd half": whole("tov_h2"),
        "Steals by the other team, 2nd half": whole("tov_live_h2"),
        "Plays ending in a turnover, 1st half": pct("tov_pct_h1"),
        "Plays ending in a turnover, 2nd half": pct("tov_pct_h2"),
    })


TABLE_FORMAT = {c: st.column_config.NumberColumn(c, format="%.0f%%") for c in (
    "Plays ending in a turnover, 1st half", "Plays ending in a turnover, 2nd half")}


def turnover_bars(g: pd.DataFrame, team_games: pd.DataFrame) -> go.Figure:
    """Turnovers in each half of every team game: a game he missed says "out", a clean game says "0"."""
    allg = team_games.sort_values("game_date")
    x = (pd.to_datetime(allg["game_date"]).dt.strftime("%b ") + pd.to_datetime(allg["game_date"]).dt.day.astype(str)
         + " " + allg["opponent_abbreviation"]).tolist()
    m = allg[["game_id"]].merge(g[["game_id", "tov_h1", "tov_h2"]], on="game_id", how="left")
    played = m["tov_h1"].notna() | m["tov_h2"].notna()
    clean = played & (m["tov_h1"].fillna(0) + m["tov_h2"].fillna(0) == 0)
    fig = go.Figure([
        go.Bar(x=x, y=m["tov_h1"], name="1st half", marker_color=H1C,
               hovertemplate="%{x}: %{y} turnovers<extra>1st half</extra>"),
        # the second half in neutral ink: orange would call every game worse before the numbers say so
        go.Bar(x=x, y=m["tov_h2"], name="2nd half", marker_color=INK_MARK,
               hovertemplate="%{x}: %{y} turnovers<extra>2nd half</extra>"),
    ])
    for mask, word, tip in ((~played, "out", "didn't play"), (clean, "0", "no turnovers")):
        if mask.any():
            fig.add_trace(go.Scatter(
                x=[x_ for x_, k in zip(x, mask) if k], y=[0] * int(mask.sum()), mode="text", text=word,
                textposition="top center", textfont=dict(size=11, color="#6b6964"), showlegend=False,
                hovertemplate=f"%{{x}}: {tip}<extra></extra>"))
    fig.update_layout(barmode="group", bargap=0.3)
    fig.update_xaxes(categoryorder="array", categoryarray=x)
    fig.update_yaxes(dtick=1, rangemode="tozero")  # whole turnovers only
    return fig


def page_players() -> None:
    ctx = current()
    data = ctx.data
    focus = ctx.focus
    season_now = ctx.season_now
    opened_from_card(data.seasons)
    PAGE_LEAGUE = ctx.pages["league"]
    page_top("Players", "Who loses the ball after halftime. Click a card or a player to dig in.",
             season=st.session_state.get("player_season"))  # the line above the title follows the season picked
    nav_ = toc_slot("players")
    band = st.container(key="band_navy_players")  # the season picker and the key cards share the navy band
    season_p = band.columns([1, 5])[0].selectbox("Season", data.seasons, key="player_season")
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
        busiest = ph[ph["split"] == "H2"].sort_values("play_share").iloc[-1]["player_name"]
        fallback = focus if focus in names else busiest
        st.session_state.setdefault("player_pick", fallback)
        if st.session_state["player_pick"] not in names:  # he didn't play this season: say so, don't just swap
            st.session_state["player_note"] = (f"{st.session_state['player_pick']} didn't play enough in {season_p}, "
                                               f"so this shows {fallback}.")
            st.session_state["player_pick"] = fallback
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
                "Biggest rise after halftime first; players with too few plays to tell come last.")
        wl = ph.pivot(index="player_name", columns="split", values="tov_pct").sort_values("H2", ascending=False)
        pct = lambda v: f"{v:.0%}"  # noqa: E731  (one unit: "1 in N" rounding made 10% → 12% look like a jump)
        hi_ = float(wl.to_numpy().max()) * 1.08
        ok_ = S.enough_plays(ph)  # rank by the change itself, the same measure as the "sloppiest" card
        wl = wl.assign(_ok=wl.index.isin(ok_), _d=wl["H2"] - wl["H1"]).sort_values(
            ["_ok", "_d"], ascending=False)[["H1", "H2"]]
        rows_ = []
        for n_, r_ in wl.iterrows():
            d_ = r_["H2"] - r_["H1"]
            color, verdict = ((BAD, "Sloppier") if d_ >= 0.02 else (GOOD, "Safer") if d_ <= -0.02
                              else (MUTED_BAR, "About the same"))
            if n_ not in ok_:
                color, verdict = MUTED_BAR, "Too few plays to tell"
            rows_.append(
                f'<div class="slop-row"><img src="{face(n_) or ""}" alt=""><div class="nm">{short(n_)}'
                f'<span class="chip sm" style="background:{color}">{verdict}</span></div>'
                f'<div class="vals"><span class="b">{pct(r_["H1"])}</span> → '
                f'<span style="color:{color}">{pct(r_["H2"])}</span></div>'
                + track(r_["H1"], r_["H2"], 0, hi_, color) + "</div>")
        st.markdown(f'<div class="slop">{"".join(rows_)}</div>', unsafe_allow_html=True)
        st.caption("Plays = shots, free-throw trips and turnovers. Sloppier or safer = a change of 2+ percentage "
                   "points in how often his plays end in a turnover. Players with 40+ minutes in each half. "
                   f"Too few plays to tell = under {S.MIN_PLAYS} plays in a half.")

        section("One player, game by game", "one")
        if st.session_state.pop("player_jump", False):  # opened from a card: bring his games into view
            st.html("<script>setTimeout(() => { const p = document.querySelector('nav.toc a[data-a=\"one\"]'); "
                    "if (p) p.click(); else document.getElementById('one')?.scrollIntoView({block: 'start'}); }, 400)"
                    "</script>", unsafe_allow_javascript=True)
        player = st.selectbox("Player", names, key="player_pick")
        if note := st.session_state.pop("player_note", None):
            st.caption(note)
        t = ph[ph["player_name"] == player].set_index("split")
        pid_ = int(t["player_id"].iloc[0])
        info = data.roster.set_index("player_id").loc[pid_] if len(data.roster) and pid_ in set(
            data.roster["player_id"]) and season_p == season_now else None
        num = f"#{info['number']}" if info is not None and str(info["number"]).strip() else ""
        d = 100 * (t.loc["H2", "tov_pct"] - t.loc["H1", "tov_pct"])
        g = D.player_games(data.player_stats, data.game_leads, season_p, player)
        sure = player in S.enough_plays(ph)
        trend = ("hc-up" if d > 0 else "hc-down") if sure else ""  # a small sample isn't colored as a finding
        tov2, plays2 = int(t.loc["H2", "tov"]), round(t.loc["H2", "plays"])
        sample = f"{tov2} turnover{'s' if tov2 != 1 else ''} on {plays2} plays after halftime · {len(g)} game{"s" if len(g) != 1 else ""}"
        steals = (f'<b>{t.loc["H2", "live_share"]:.0%}</b><span>of his second-half turnovers were steals (live balls '
                  f'that lead to fast breaks)</span>' if tov2 else
                  '<b>0</b><span>second-half turnovers, so no steals against him</span>')
        st.markdown(
            f'<div class="hero-card"><div class="hc-photo"><span class="pc-num">{num}</span>'
            f'<img src="{D.headshot_url(pid_)}" alt=""></div><div class="hc-body">'
            f'<div class="kicker">{info["position"] + " · " if info is not None else ""}after halftime, {season_p}</div>'
            f'<div class="hc-name">{player}</div><div class="hc-n">{sample}</div><div class="hc-stats">'
            f'<div><b>{t.loc["H1", "tov_pct"]:.0%}</b><span>of his plays end in a turnover before halftime</span></div>'
            f'<div><b class="{trend}">{t.loc["H2", "tov_pct"]:.0%}</b><span>after halftime '
            + (f'({"+" if d > 0 else "−"}{abs(d):.1f} pts)' if sure else
               f'(too few plays to tell: under {S.MIN_PLAYS} in a half)') + '</span></div>'
            f'<div><b>{t.loc["H2", "tov_share"]:.0%}</b><span>of the team\'s turnovers after halftime, on '
            f'{t.loc["H2", "play_share"]:.0%} of its plays</span></div>'
            f'<div>{steals}</div></div></div></div>', unsafe_allow_html=True)
        show(turnover_bars(g, data.game_leads[data.game_leads["season"] == season_p]), 300, yaxis_title="Turnovers")
        with st.expander("Game-by-game table"):
            st.dataframe(game_table(g), hide_index=True, width="stretch", column_config=TABLE_FORMAT)
    fill_toc(nav_)
    next_step(PAGE_LEAGUE, "Is this normal? How every team changes after halftime.")
