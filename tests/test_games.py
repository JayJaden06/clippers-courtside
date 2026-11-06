"""Game-log parsing tests on a trimmed real ``LeagueGameLog`` response.

Fixture: every row for the Clippers' first 8 games of 2026-27 (both teams),
plus one unrelated game, pulled 2026-11-05.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from leadcollapse.ingest.games import games_frame, parse_game_log, team_games
from leadcollapse.models import Game

FIXTURE = Path(__file__).parent / "fixtures" / "leaguegamelog_2026-27_lac.json"
LAC = 1610612746


@pytest.fixture
def raw() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _set(raw: dict, game_id: str, team_id: int, column: str, value) -> dict:
    """Return a copy of ``raw`` with one cell changed."""
    raw = copy.deepcopy(raw)
    rs = raw["resultSets"][0]
    h = rs["headers"]
    for row in rs["rowSet"]:
        if row[h.index("GAME_ID")] == game_id and row[h.index("TEAM_ID")] == team_id:
            row[h.index(column)] = value
    return raw


def test_parses_every_row(raw):
    assert len(parse_game_log(raw)) == 18


def test_keeps_only_clippers_games_in_date_order(raw):
    games = team_games(parse_game_log(raw), LAC, "2026-27")

    assert len(games) == 8
    assert {g.team_abbreviation for g in games} == {"LAC"}
    assert "0022600001" not in {g.game_id for g in games}
    assert [g.game_date for g in games] == sorted(g.game_date for g in games)
    assert "".join(g.result for g in games) == "WLLLLLLL"


def test_opening_night_home_win(raw):
    first = team_games(parse_game_log(raw), LAC, "2026-27")[0]

    assert first.game_id == "0022600093"
    assert first.opponent_abbreviation == "SAC"
    assert first.is_home is True
    assert (first.team_pts, first.opp_pts, first.margin) == (121, 96, 25)
    assert first.overtime_periods == 0


def test_road_game_and_overtime_are_detected(raw):
    games = {g.opponent_abbreviation + str(g.game_date): g for g in team_games(parse_game_log(raw), LAC, "2026-27")}

    assert games["LAL2026-10-23"].is_home is False
    mia = games["MIA2026-11-04"]
    assert (mia.team_pts, mia.opp_pts, mia.overtime_periods) == (116, 117, 1)


def test_unfinished_game_is_skipped(raw):
    raw = _set(raw, "0022600093", LAC, "WL", None)
    games = team_games(parse_game_log(raw), LAC, "2026-27")

    assert len(games) == 7
    assert "0022600093" not in {g.game_id for g in games}


def test_plus_minus_mismatch_is_rejected(raw):
    raw = _set(raw, "0022600093", LAC, "PLUS_MINUS", 24)
    with pytest.raises(ValueError, match="PLUS_MINUS"):
        team_games(parse_game_log(raw), LAC, "2026-27")


def test_impossible_minutes_are_rejected(raw):
    raw = _set(raw, "0022600093", LAC, "MIN", 250)
    with pytest.raises(ValueError, match="minutes"):
        team_games(parse_game_log(raw), LAC, "2026-27")


def test_result_must_match_score():
    with pytest.raises(ValidationError, match="contradicts"):
        Game(
            game_id="0022600093", season="2026-27", game_date="2026-10-21",
            team_id=LAC, team_abbreviation="LAC", opponent_id=1, opponent_abbreviation="SAC",
            is_home=True, result="L", team_pts=121, opp_pts=96, overtime_periods=0,
        )


def test_bad_matchup_format_is_rejected(raw):
    raw = _set(raw, "0022600093", LAC, "MATCHUP", "LAC v SAC")
    with pytest.raises(ValidationError, match="MATCHUP"):
        parse_game_log(raw)


def test_frame_has_one_flat_row_per_game(raw):
    df = games_frame(team_games(parse_game_log(raw), LAC, "2026-27"))

    assert len(df) == 8
    assert df["game_id"].is_unique
    assert (df["margin"] == df["team_pts"] - df["opp_pts"]).all()
    assert str(df["game_date"].dtype).startswith("datetime64")
