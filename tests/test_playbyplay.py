"""Play-by-play parsing tests on two real games, trimmed to scoring, turnover, foul and period events.

- 0022600186: LAC (home) vs MIA, 2026-11-04, lost 116-117 in OT after leading 67-52 at half.
- 0022500827: LAC (home) vs ORL, 2026-02-22, lost 109-111; a technical free throw at the
  start of Q3 has the same elapsed time (24:00) as the end of Q2.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from leadcollapse.ingest.playbyplay import check_final_scores, parse_pbp
from leadcollapse.models import elapsed_seconds, parse_clock

FIXTURES = Path(__file__).parent / "fixtures"
LAC = 1610612746


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def mia() -> pd.DataFrame:
    return parse_pbp(load("pbp_0022600186_lac_mia_ot.json"), LAC, is_home=True)


@pytest.fixture(scope="module")
def orl() -> pd.DataFrame:
    return parse_pbp(load("pbp_0022500827_orl_q3_technical.json"), LAC, is_home=True)


@pytest.mark.parametrize(
    "clock, seconds", [("PT12M00.00S", 720.0), ("PT11M43.00S", 703.0), ("PT00M00.40S", 0.4), ("PT05M00.00S", 300.0)]
)
def test_parse_clock(clock, seconds):
    assert parse_clock(clock) == pytest.approx(seconds)


def test_parse_clock_rejects_garbage():
    with pytest.raises(ValueError):
        parse_clock("11:43")


@pytest.mark.parametrize(
    "period, remaining, elapsed",
    [(1, 720, 0), (1, 0, 720), (2, 0, 1440), (4, 0, 2880), (5, 300, 2880), (5, 0, 3180), (6, 0, 3480)],
)
def test_elapsed_seconds_regulation_and_overtime(period, remaining, elapsed):
    assert elapsed_seconds(period, remaining) == elapsed


def test_final_score_from_team_side(mia):
    last = mia.iloc[-1]
    assert (last["team_score"], last["opp_score"], last["margin"]) == (116, 117, -1)
    assert mia["period"].max() == 5
    assert mia["elapsed"].max() == 3180  # 48 + 5 minutes


def test_road_perspective_flips_scores():
    away = parse_pbp(load("pbp_0022600186_lac_mia_ot.json"), 1610612748, is_home=False)
    assert (away.iloc[-1]["team_score"], away.iloc[-1]["opp_score"]) == (117, 116)


def test_score_is_forward_filled_and_starts_at_zero(mia):
    assert mia[["team_score", "opp_score"]].notna().all().all()
    assert (mia.iloc[0]["team_score"], mia.iloc[0]["opp_score"]) == (0, 0)
    assert mia["team_score"].is_monotonic_increasing
    assert mia["opp_score"].is_monotonic_increasing


def test_end_of_second_period_score(mia):
    end_q2 = mia[mia["period"] <= 2].iloc[-1]
    assert (end_q2["team_score"], end_q2["opp_score"]) == (67, 52)


def test_events_in_time_order(mia, orl):
    assert mia["elapsed"].is_monotonic_increasing
    assert orl["elapsed"].is_monotonic_increasing


def test_team_flags_and_neutral_events(mia):
    neutral = mia[mia["action_type"] == "period"]
    assert neutral["team_id"].isna().all()
    assert not (neutral["is_team"] | neutral["is_opp"]).any()
    tov = mia[mia["action_type"] == "Turnover"]
    assert set(tov.loc[tov["is_team"], "team_id"]) == {LAC}
    assert set(tov.loc[tov["is_opp"], "team_id"]) == {1610612748}
    assert (tov["is_team"] | tov["is_opp"]).all()


def test_team_actions_are_attributed_to_the_team(mia):
    # "CLIPPERS Turnover: Shot Clock" arrives with teamId 0 and the team id in personId.
    shot_clock = mia[(mia["action_type"] == "Turnover") & (mia["sub_type"] == "Shot Clock Turnover")]
    assert len(shot_clock) == 2
    assert shot_clock["is_team"].all() and shot_clock["is_team_action"].all()
    assert shot_clock["person_id"].isna().all()
    assert set(shot_clock["team_id"]) == {LAC}


def test_turnover_and_foul_subtypes_kept(mia):
    tov = set(mia.loc[mia["action_type"] == "Turnover", "sub_type"])
    fouls = set(mia.loc[mia["action_type"] == "Foul", "sub_type"])
    assert {"Bad Pass", "Lost Ball"} <= tov
    assert {"Shooting", "Personal"} <= fouls


def test_q3_technical_shares_halftime_timestamp(orl):
    # The reason halftime is cut by period, not by elapsed time.
    at_24 = orl[orl["elapsed"] == 1440]
    assert set(at_24["period"]) == {2, 3}
    tech_ft = at_24[at_24["sub_type"] == "Free Throw Technical"]
    assert tech_ft["period"].tolist() == [3]


def test_check_final_scores(mia):
    games = pd.DataFrame([{"game_id": "0022600186", "team_pts": 116, "opp_pts": 117}])
    check_final_scores(mia, games)
    with pytest.raises(ValueError, match="final score"):
        check_final_scores(mia, games.assign(team_pts=115))
