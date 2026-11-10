"""Tracking payloads parse into flat rows, and the matchup check catches bad parses."""

from __future__ import annotations

import pandas as pd
import pytest

from courtside.ingest import tracking


def _p(pid, first, last, **extra):
    return {"personId": pid, "firstName": first, "familyName": last, **extra}


def _matchup_payload():
    stat = {"partialPossessions": 10.5, "matchupMinutesSort": 120.0, "playerPoints": 6, "matchupTurnovers": 1,
            "matchupFieldGoalsMade": 3, "matchupFieldGoalsAttempted": 5}
    return {"boxScoreMatchups": {
        "gameId": "g1",
        "homeTeam": {"teamTricode": "LAC", "players": [
            _p(1, "Home", "Guard", matchups=[_p(9, "Away", "Wing", statistics=stat)])]},
        "awayTeam": {"teamTricode": "NOP", "players": [_p(9, "Away", "Wing", matchups=[])]},
    }}


def test_parse_matchups_orients_offense_and_defense():
    rows = tracking.parse_matchups(_matchup_payload())
    assert len(rows) == 1
    r = rows[0]
    assert (r["off_team"], r["off_player"], r["def_team"], r["def_player"]) == ("LAC", "Home Guard", "NOP", "Away Wing")
    assert (r["partial_poss"], r["pts"], r["tov"], r["fga"], r["switches_on"]) == (10.5, 6, 1, 5, 0)


def test_parse_hustle_minutes():
    raw = {"boxScoreHustle": {"gameId": "g1", "homeTeam": {"teamTricode": "LAC", "players": [
        _p(1, "A", "B", statistics={"minutes": "36:30", "deflections": 3, "defensiveBoxOuts": 2})]},
        "awayTeam": {"teamTricode": "NOP", "players": []}}}
    (row,) = tracking.parse_hustle(raw)
    assert row["minutes"] == pytest.approx(36.5) and row["deflections"] == 3 and row["def_box_outs"] == 2


def test_check_matchups_coverage_and_overcount():
    box = pd.DataFrame({"game_id": ["g1", "g1"], "player_id": [1, 2], "split": ["FULL", "FULL"], "pts": [20, 10]})
    ok = pd.DataFrame({"game_id": ["g1", "g1"], "off_player_id": [1, 2], "pts": [18.0, 10.0]})
    assert tracking.check_matchups(ok, box) == pytest.approx(28 / 30)
    with pytest.raises(ValueError, match="exceed"):
        tracking.check_matchups(ok.assign(pts=[25.0, 10.0]), box)
    with pytest.raises(ValueError, match="cover only"):
        tracking.check_matchups(ok.assign(pts=[5.0, 5.0]), box)
