"""Clippers Courtside: the website's entry point.

Run with ``courtside dashboard`` (or ``streamlit run dashboard/app.py``). Reads ``data/processed/`` and
``reports/analysis.json`` only: no API calls.

This script runs once per visitor interaction. It loads the data (cached until a refresh lands), builds the run's
context (``courtside.web.context``), sets up the menu and draws the page. The pages live in ``courtside.web.views``,
the pieces they share in ``courtside.web.ui``, and the styling in ``courtside/web/style.css``.

If anything fails, visitors get a friendly "come back later" message and the full error goes to the logs
(Streamlit Cloud: Manage app -> Logs). Tests, and anyone who sets COURTSIDE_SHOW_ERRORS=1, see the real error.
"""

from __future__ import annotations

import logging
import os

import streamlit as st

log = logging.getLogger("courtside.web")
SHOW_ERRORS = os.environ.get("COURTSIDE_SHOW_ERRORS") == "1"


def sorry() -> None:
    """What a visitor sees when the site can't draw a page. Needs nothing but Streamlit, so it works even if the
    rest of the site failed to load (say, mid-update)."""
    st.markdown(
        '<div style="max-width:40rem;margin:4rem auto;padding:2rem 2.2rem;border-radius:18px;background:#ffffff;'
        'box-shadow:0 6px 18px rgba(19,33,63,.10);text-align:center">'
        '<div style="font-size:2.6rem;line-height:1">🏀</div>'
        '<h2 style="margin:.6rem 0 .4rem 0;color:#13213f">Sorry, this page isn\'t loading right now</h2>'
        '<p style="font-size:1.05rem;opacity:.8;margin:0">The site may be updating with the latest game, which only '
        "takes a minute or two. Please try again shortly.</p></div>", unsafe_allow_html=True)
    if st.button("Try again", icon=":material/refresh:"):
        st.rerun()


try:  # if the site's own code can't load (an update landing mid-visit), still say something friendly
    from courtside.config import get_settings
    from courtside.names import set_team, short
    from courtside.viz import dashboard_data as D
    from courtside.web import ui
    from courtside.web.context import Ctx, use
    from courtside.web.views import blown_leads, games, league, players, what_now
except Exception:
    if SHOW_ERRORS:
        raise
    log.exception("the site failed to load")
    sorry()
    st.stop()

settings = get_settings()
st.set_page_config(page_title="Clippers Courtside", page_icon=D.logo_url(settings.team_id), layout="wide",
                   initial_sidebar_state="collapsed")
st.logo(D.logo_url(settings.team_id), size="large")
ui.inject_style()


def data_stamp() -> tuple[float, int]:
    """Newest modified time and file count of the data the app reads; changes whenever a refresh lands."""
    files = [p for p in settings.processed_dir.rglob("*") if p.is_file()]
    files.append(settings.reports_dir / "analysis.json")
    return max((p.stat().st_mtime for p in files if p.exists()), default=0.0), len(files)


@st.cache_data(max_entries=1)
def load(stamp: tuple[float, int]) -> D.DashboardData:
    """Cached per ``stamp``, so a running server picks up new games without a restart. ``stamp`` is only the
    cache key: don't rename it with a leading underscore, which would make Streamlit leave it out of the key."""
    return D.load(settings.processed_dir, settings.reports_dir / "analysis.json")


# Tests run a specific page by setting this in session state (AppTest can't change the URL); they want real errors.
forced = st.session_state.get("_dashboard_page")
try:
    data = load(data_stamp())
except Exception:
    if SHOW_ERRORS or forced:
        raise
    log.exception("the data failed to load (run `courtside build` and `courtside analyze`?)")
    sorry()
    st.stop()

meta = data.analysis.get("meta", {})
season_now = meta.get("season", data.seasons[0])
# Every Clipper's name (box scores, any season, plus today's roster), so two who share a last name are told apart.
set_team(set(data.player_box.loc[data.player_box["team_abbreviation"] == settings.team_abbreviation, "player_name"])
         | (set(data.roster["player_name"]) if len(data.roster) else set()))
ctx = Ctx(
    settings=settings, data=data, meta=meta, season_now=season_now,
    season_prev=meta.get("comparison_season", data.seasons[-1]),
    focus=settings.focus_player, focus_last=short(settings.focus_player),
    player_ids=D.player_ids(data),
    team_last=D.last_names(data.player_box, season_now, settings.team_abbreviation),
)
use(ctx)

# Story order: the tracker, the answer, the evidence (players, the league), then what to do about it.
RENDER = {"replay": games.page_replay, "answer": blown_leads.page_answer, "players": players.page_players,
          "league": league.page_league, "whatnow": what_now.page_whatnow}
ctx.pages.update({
    "replay": st.Page(games.page_replay, title="Games", url_path="home", default=True),
    "answer": st.Page(blown_leads.page_answer, title="Blown leads", icon=":material/lightbulb:", url_path="answer"),
    "players": st.Page(players.page_players, title="Players", icon=":material/groups:", url_path="players"),
    "league": st.Page(league.page_league, title="Vs. the league", icon=":material/leaderboard:", url_path="league"),
    "whatnow": st.Page(what_now.page_whatnow, title="What now?", icon=":material/checklist:", url_path="what-now"),
})
# The default page lives at "/"; this hidden twin makes a shared /games link work too.
games_link = st.Page(games.page_replay, title="Games", url_path="games", visibility="hidden")
nav = st.navigation(list(ctx.pages.values()) + [games_link], position="top")

# The page draws into one slot, so a failure partway through can clear what was drawn and show the apology instead.
# (Streamlit's own reruns and stops aren't Exceptions, so they pass straight through.)
slot = st.empty()
try:
    with slot.container():
        ui.behind_notice()
        # (statements, not one conditional expression: Streamlit "magic" would print a bare expression's value)
        if forced in RENDER:
            RENDER[forced]()
        else:
            nav.run()
except Exception:
    if SHOW_ERRORS or forced:
        raise
    log.exception("page %s failed", forced or "from the menu")
    slot.empty()
    sorry()
