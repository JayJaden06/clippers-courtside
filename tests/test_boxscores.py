"""Box-score parsing and split-consistency tests on a real game.

Fixture: Clippers vs Heat, 2026-11-04 (0022600186), lost 116-117 in OT after
leading 67-52 at halftime. Team totals are untouched; player lists are cut to
Garland, a bench player and a DNP for LAC (and a starter + bench for MIA).
Splits: traditional FULL/H1/H2/OT; advanced and misc FULL (with players) and H1/H2/OT
(team totals only).
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from courtside.ingest.boxscores import (
    H2,
    OT,
    QUARTERS,
    FULL,
    check_split_totals,
    parse_box,
    splits_for,
)
from courtside.models import TraditionalStats, parse_minutes

FIXTURE = Path(__file__).parent / "fixtures" / "boxscore_0022600186_lac_mia_ot.json"
LAC = 1610612746


@pytest.fixture(scope="module")
def raw() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def team_frame(raw: dict, splits=("FULL", "H1", "H2", "OT")) -> pd.DataFrame:
    rows = []
    for s in splits:
        teams, _ = parse_box(raw[f"traditional/{s}"], "traditional", s)
        rows += teams
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "value, expected",
    [("39:13", 39 + 13 / 60), ("265:00", 265.0), ("0:45", 0.75), ("", 0.0), (None, 0.0), (12, 12.0)],
)
def test_parse_minutes(value, expected):
    assert parse_minutes(value) == pytest.approx(expected)


def test_period_splits_request_range_type_1():
    # range_type=0 (the nba_api default) silently returns the full game.
    assert H2.params()["range_type"] == 1
    assert H2.params()["start_period"] == 3 and H2.params()["end_period"] == 4
    assert FULL.params() == {}


def test_overtime_split_only_for_overtime_games():
    assert OT not in splits_for(0)
    assert OT in splits_for(1)
    assert set(QUARTERS) <= set(splits_for(0))
    assert not set(QUARTERS) & set(splits_for(0, quarters=False))


def test_team_rows_full_game(raw):
    teams, _ = parse_box(raw["traditional/FULL"], "traditional", "FULL")
    lac = next(t for t in teams if t["team_id"] == LAC)
    mia = next(t for t in teams if t["team_id"] != LAC)

    assert (lac["pts"], mia["pts"]) == (116, 117)
    assert lac["minutes"] == mia["minutes"] == 265.0  # 240 + one 25-minute OT
    assert lac["is_home"] is True and mia["is_home"] is False


def test_halftime_lead_and_second_half_collapse(raw):
    pts = team_frame(raw).pivot(index="split", columns="team_abbreviation", values="pts")

    assert (pts.loc["H1", "LAC"], pts.loc["H1", "MIA"]) == (67, 52)
    assert (pts.loc["H2", "LAC"], pts.loc["H2", "MIA"]) == (40, 55)
    assert (pts.loc["OT", "LAC"], pts.loc["OT", "MIA"]) == (9, 10)


def test_player_rows(raw):
    _, players = parse_box(raw["traditional/FULL"], "traditional", "FULL")
    lac = {p["player_name"]: p for p in players if p["team_id"] == LAC}

    garland = lac["Darius Garland"]
    assert garland["starter"] is True
    assert garland["minutes"] == pytest.approx(39 + 13 / 60)
    assert (garland["pts"], garland["tov"]) == (26, 6)

    dnp = next(p for p in lac.values() if p["status_comment"])
    assert dnp["minutes"] == 0 and dnp["pts"] == 0 and dnp["starter"] is False


def test_split_lists_only_players_who_appeared(raw):
    _, players = parse_box(raw["traditional/OT"], "traditional", "OT")
    assert [p["player_name"] for p in players if p["team_id"] == LAC] == ["Darius Garland"]


def test_advanced_rows(raw):
    teams, players = parse_box(raw["advanced/FULL"], "advanced", "FULL")
    lac = next(t for t in teams if t["team_id"] == LAC)

    assert lac["net_rating"] == pytest.approx(lac["off_rating"] - lac["def_rating"], abs=0.15)
    assert 0 < lac["efg_pct"] < 1 and lac["possessions"] > 100
    assert "starter" not in lac  # traditional-only columns stay out of advanced rows
    assert all("usg_pct" in p for p in players)


def test_halves_plus_overtime_equal_full_game(raw):
    check_split_totals(team_frame(raw))


def test_ignored_period_filter_is_caught(raw):
    # Simulate the range_type=0 bug: the "H2" response is really the full game.
    broken = copy.deepcopy(raw)
    broken["traditional/H2"] = copy.deepcopy(raw["traditional/FULL"])
    with pytest.raises(ValueError, match="H1 \\+ H2 \\+ OT != FULL"):
        check_split_totals(team_frame(broken))


def test_missing_overtime_split_is_caught(raw):
    with pytest.raises(ValueError, match="!= FULL"):
        check_split_totals(team_frame(raw, splits=("FULL", "H1", "H2")))


def test_points_must_match_makes():
    stats = {
        "minutes": "10:00", "fieldGoalsMade": 4, "fieldGoalsAttempted": 8, "threePointersMade": 1,
        "threePointersAttempted": 3, "freeThrowsMade": 2, "freeThrowsAttempted": 2,
        "reboundsOffensive": 0, "reboundsDefensive": 0, "assists": 0, "steals": 0, "blocks": 0,
        "turnovers": 1, "foulsPersonal": 0, "points": 11, "plusMinusPoints": 0,
    }
    assert TraditionalStats.model_validate(stats).pts == 11
    with pytest.raises(ValidationError, match="points"):
        TraditionalStats.model_validate(stats | {"points": 12})


def _bump(df: pd.DataFrame, split: str, stat: str, by: int) -> pd.DataFrame:
    df = df.copy()
    mask = (df["split"] == split) & (df["team_id"] == LAC)
    df.loc[mask, stat] += by
    return df


def test_small_stat_correction_is_reported_not_raised(raw):
    # Real case: 0022500243, the full-game box had one more DREB than its halves.
    gaps = check_split_totals(_bump(team_frame(raw), "FULL", "dreb", 1))

    assert len(gaps) == 1
    assert gaps.iloc[0][["stat", "parts_minus_whole"]].tolist() == ["dreb", -1]
    assert check_split_totals(team_frame(raw)).empty


@pytest.mark.parametrize("stat, by", [("pts", 1), ("dreb", 3), ("tov", -5)])
def test_points_gap_or_large_gap_still_raises(raw, stat, by):
    with pytest.raises(ValueError, match="!= FULL"):
        check_split_totals(_bump(team_frame(raw), "FULL", stat, by))


def test_quarter_checks_skip_games_without_quarters(raw):
    # FULL/H1/H2/OT only (like 2025-26): no quarter rows, so only the half check runs.
    assert check_split_totals(team_frame(raw)).empty


def test_missing_half_is_caught(raw):
    with pytest.raises(ValueError, match="missing"):
        check_split_totals(team_frame(raw, splits=("FULL", "H1", "OT")))


def test_quarter_mismatch_is_caught_when_quarters_exist(raw):
    df = team_frame(raw)
    h1 = df[df["split"] == "H1"]
    zero_q = h1.assign(**{c: 0 for c in ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb",
                                          "ast", "stl", "blk", "tov", "pf", "pts"]})
    h2 = df[df["split"] == "H2"]
    good = pd.concat([df, h1.assign(split="Q1"), zero_q.assign(split="Q2"),
                      h2.assign(split="Q3"), zero_q.assign(split="Q4")])
    assert check_split_totals(good).empty

    doubled = pd.concat([df, h1.assign(split="Q1"), h1.assign(split="Q2"),
                         h2.assign(split="Q3"), zero_q.assign(split="Q4")])
    with pytest.raises(ValueError, match=r"Q1 \+ Q2 != H1"):
        check_split_totals(doubled)
