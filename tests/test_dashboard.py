"""Dashboard data functions on the committed processed tables, plus a headless run of the app."""

from __future__ import annotations

from pathlib import Path

import pytest

from courtside.viz import dashboard_data as D

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
pytestmark = pytest.mark.skipif(not (PROCESSED / "game_halves.parquet").exists(), reason="run courtside build")


@pytest.fixture(scope="module")
def data() -> D.DashboardData:
    return D.load(PROCESSED, ROOT / "reports" / "analysis.json")


def test_game_options_cover_every_game_newest_first(data):
    for season in data.seasons:
        opts = D.game_options(data.game_leads, season)
        assert len(opts) == (data.game_leads["season"] == season).sum()
        assert opts["label"].is_unique
    assert data.seasons[0] > data.seasons[-1]


def test_game_views_agree_with_the_box_score(data):
    leads = data.game_leads.set_index("game_id")
    for gid in D.game_options(data.game_leads, data.seasons[0])["game_id"]:
        table = D.game_half_table(data.game_halves, gid).set_index("stat")
        margin_change = int(table.loc["Margin", "2nd half"]) - int(table.loc["Margin", "1st half"])
        att = D.game_attribution(data.game_halves, gid)
        assert att["points"].sum() == pytest.approx(att["margin_change"].iloc[0])
        assert att["margin_change"].iloc[0] == margin_change
        assert int(table.loc["Margin", "1st half"]) == leads.loc[gid, "halftime_margin"]
        m = D.game_margin(data.margins, gid)
        assert m["margin"].iloc[-1] == leads.loc[gid, "final_margin"]


def test_game_turnovers_match_team_half_counts(data):
    gid = D.game_options(data.game_leads, data.seasons[0])["game_id"].iloc[0]
    tov = D.game_turnovers(data.turnovers, gid)
    gh = data.game_halves[(data.game_halves["game_id"] == gid)].assign(split=lambda d: d["split"].astype(str))
    for half, split in (("1st", "H1"), ("2nd", "H2")):
        row = gh[gh["split"] == split].iloc[0]
        assert (tov["half"] == half).sum() == row["tov"] + row["tov_team"]
    assert tov["clock"].str.match(r"^(Q[1-4]|OT\d) \d+:\d\d$").all()


def test_player_halves_shares_and_games(data):
    season = data.seasons[0]
    ph = D.player_halves(data.player_stats, season)
    assert (ph.groupby("player_id")["split"].nunique() == 2).all()
    assert ph["tov_share"].between(0, 1).all()
    games = D.player_games(data.player_stats, data.game_leads, season, "Darius Garland")
    totals = ph[ph["player_name"] == "Darius Garland"].set_index("split")["tov"]
    assert games["tov_h1"].sum() == totals["H1"] and games["tov_h2"].sum() == totals["H2"]


def test_league_view_is_ranked_with_one_clippers_row(data):
    lv = D.league_view(data.ranks, data.seasons[0], "net_rtg", "change")
    assert len(lv) == 30 and lv["is_team"].sum() == 1
    assert lv["rank"].tolist() == list(range(1, 31))
    assert lv["value"].is_monotonic_decreasing  # higher net-rating change is better
    cr = D.clippers_ranks(data.ranks, data.seasons[0])
    assert cr.index[0] == "Net rating" and cr.to_numpy().max() <= 30


def _app(page: str):
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120)
    at.session_state["_dashboard_page"] = page
    at.run()
    assert not at.exception, (page, at.exception[0].value)
    return at


def test_answer_page_runs_and_what_if_toggles_work():
    at = _app("answer")
    for key in ("fix_shooting", "fix_garland"):
        at.toggle(key=key).set_value(True).run()
        assert not at.exception, (key, at.exception[0].value)
    assert any(m.label == "Second halves, per game" for m in at.metric)
    assert "Is it The Wall?" in [s.value for s in at.subheader]


def test_what_now_page_runs():
    at = _app("whatnow")
    # four main sections, then the rest in one "More evidence" dropdown
    assert [h.value for h in at.header] == [
        "Who plays after halftime, and how it goes", "Should Garland handle the ball less?",
        "Defense, rebounding and size", "Will the returning players help?", "More evidence",
        "Does the starting five need to change?", "Chemistry: which duos work?", "Who stopped shooting after halftime?"]
    page_html = " ".join(m.value for m in at.markdown)
    assert page_html.count('class="vcard"') >= 5  # the answer cards
    assert "Is he being double-teamed into them?" in page_html


@pytest.mark.parametrize("page, key", [("replay", "game_season"), ("players", "player_season"),
                                       ("league", "league_season")])
def test_each_page_runs_for_every_season(page, key):
    at = _app(page)
    for season in at.selectbox(key=key).options:
        at.selectbox(key=key).set_value(season).run()
        assert not at.exception, (page, season, at.exception[0].value)


def test_league_page_every_stat():
    at = _app("league")
    for metric in ["net_rtg", "efg_pct", "opp_efg_pct", "tov_pct", "opp_ft_rate", "opp_oreb_pct"]:
        at.button_group(key="league_metric").set_value(metric).run()
        assert not at.exception, (metric, at.exception[0].value)


def test_replay_opens_the_game_passed_from_the_answer_page(data):
    from courtside.viz import story as S

    worst = S.worst_collapse(data.game_leads, data.seasons[0])
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120)
    at.session_state["_dashboard_page"] = "replay"
    at.session_state["jump_game"] = worst["game_id"]
    at.run()
    assert not at.exception
    pick = at.selectbox(key=f"pick_{data.seasons[0]}").value
    month, day, _, opp = worst["label"].split()
    assert pick.startswith(f"{month} {day} ·") and f" {opp} ·" in pick, pick


def test_games_page_opens_an_upcoming_game_plan(data):
    at = _app("replay")
    key = f"pick_{data.seasons[0]}"
    nxt = [o for o in at.selectbox(key=key).options if "next game" in o]
    if not nxt:
        pytest.skip("no upcoming games in the schedule")
    at.selectbox(key=key).set_value(nxt[0]).run()
    assert not at.exception, at.exception[0].value if at.exception else None
    assert any(h.value.startswith("How to beat") for h in at.header)
    assert at.toggle  # the game-plan checklist


def test_games_page_opens_with_the_intro_and_it_can_be_hidden(data):
    at = _app("replay")
    html = " ".join(m.value for m in at.markdown)
    assert "Clippers Courtside" in html and 'href="answer"' in html and 'href="league"' in html
    wins = int((data.games.loc[data.games["season"] == data.seasons[0], "result"] == "W").sum())
    assert f">{wins}-" in html
    next(b for b in at.button if b.label == "Hide intro").click().run()
    assert not at.exception
    assert "Clippers Courtside</div>" not in " ".join(m.value for m in at.markdown)


def test_names_in_finds_players_in_reading_order():
    full = {"Darius Garland": 1, "Kris Dunn": 2, "Zion Williamson": 3, "Derrick Jones Jr.": 4}
    last = {"Garland": 1, "Dunn": 2, "Jones Jr.": 4}
    assert D.names_in("More minutes: Dunn & Garland", full, last) == [2, 1]
    assert D.names_in("Best Clipper: Darius Garland", full, last) == [1]
    assert D.names_in("Zion Williamson scored on Jones Jr.", full, last) == [3, 4]
    assert D.names_in("Dunnigan and Garlands", full, last) == []  # whole words only


def test_every_clipper_has_a_photo_id(data):
    ids = D.player_ids(data)
    lac = data.player_box[(data.player_box["season"] == data.seasons[0])
                          & (data.player_box["team_abbreviation"] == "LAC")]["player_name"].unique()
    assert all(n in ids for n in lac)
    last = D.last_names(data.player_box, data.seasons[0], "LAC")
    assert len(set(last.values())) == len(last)


def test_next_up_banner_opens_the_game_plan(data):
    at = _app("replay")
    html = " ".join(m.value for m in at.markdown)
    if data.schedule.empty:
        pytest.skip("no schedule")
    assert 'class="nextup"' in html and "chance to win" in html
    next(b for b in at.button if b.label == "Open the game plan").click().run()
    assert not at.exception
    assert "next game" in at.selectbox(key=f"pick_{data.seasons[0]}").value


def test_players_page_shows_the_rotation_as_cards():
    at = _app("players")
    html = " ".join(m.value for m in at.markdown)
    assert html.count('class="pcard"') >= 5 and "of the plays after halftime" in html


def test_game_cards_show_the_matchup_in_both_teams_colors():
    assert D.is_light("#FFFFFF") and D.is_light("#EEE1C6") and not D.is_light("#1d428a")
    at = _app("replay")
    html = " ".join(m.value for m in at.markdown)
    assert html.count('class="gm"') >= 5 and "--opp:" in html


def test_drawings_have_their_numbers_beside_them():
    html = " ".join(m.value for m in _app("answer").markdown)
    assert "<h4>Half by half</h4>" in html and html.count('class="cmp-row"') >= 4
    players = " ".join(m.value for m in _app("players").markdown)
    assert players.count('class="slop-row"') >= 5 and ("Sloppier" in players or "Safer" in players)
