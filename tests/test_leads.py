"""Margin series and blown-lead classification tests.

Real games (from the trimmed play-by-play fixtures) check the derived
metrics; small synthetic score sequences check classification edge cases.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from leadcollapse.ingest.playbyplay import parse_pbp
from leadcollapse.models import elapsed_seconds
from leadcollapse.transform.leads import check_halftime_margins, classify, game_leads, margin_series

FIXTURES = Path(__file__).parent / "fixtures"
LAC = 1610612746


def real(name: str, game_id: str, result: str, ot: int) -> pd.Series:
    raw = json.loads((FIXTURES / name).read_text(encoding="utf-8"))
    events = parse_pbp(raw, LAC, is_home=True).assign(season="test")
    games = pd.DataFrame([meta(game_id, result, ot)])
    return game_leads(margin_series(events), games, threshold=8, sensitivity=[5, 8, 12]).iloc[0]


def meta(game_id: str, result: str, ot: int = 0) -> dict:
    return {"season": "test", "game_id": game_id, "game_date": pd.Timestamp("2026-01-01"),
            "opponent_abbreviation": "OPP", "is_home": True, "result": result, "overtime_periods": ot}


def synthetic(scores: list[tuple[int, float, int, int]], game_id: str = "G") -> pd.DataFrame:
    """Events from (period, clock_remaining, team_score, opp_score) tuples."""
    rows = [{"season": "test", "game_id": game_id, "period": p, "elapsed": elapsed_seconds(p, c),
             "team_score": t, "opp_score": o, "margin": t - o} for p, c, t, o in scores]
    return pd.DataFrame(rows)


def leads_for(scores, result, ot=0, threshold=8) -> pd.Series:
    ms = margin_series(synthetic(scores))
    return game_leads(ms, pd.DataFrame([meta("G", result, ot)]), threshold=threshold).iloc[0]


# ------------------------------------------------------------------ real games

@pytest.fixture(scope="module")
def mia() -> pd.Series:
    return real("pbp_0022600186_lac_mia_ot.json", "0022600186", "L", ot=1)


def test_mia_lead_metrics(mia):
    assert mia["max_lead"] == 20
    assert mia["halftime_margin"] == 15 and mia["led_at_half"]
    assert mia["q4_start_margin"] == 9
    assert mia["final_margin"] == -1
    assert mia["max_lead_window"] == 20


def test_mia_is_blown_at_every_threshold(mia):
    assert [mia[f"lead_class_{t}"] for t in (5, 8, 12)] == ["blown_lead"] * 3
    assert mia["lead_class"] == "blown_lead"


def test_mia_minutes_include_overtime(mia):
    assert 0 < mia["minutes_led"] + mia["minutes_trailed"] <= 53
    assert mia["minutes_led"] > mia["minutes_trailed"]  # led most of the game and still lost


def test_orl_technical_free_throw_counts_in_second_half():
    orl = real("pbp_0022500827_orl_q3_technical.json", "0022500827", "L", ot=0)
    # Box score: H1 53-57. A time cut at 24:00 would wrongly give 54-57 (-3).
    assert orl["halftime_margin"] == -4
    assert orl["final_margin"] == -2


def test_check_halftime_margins():
    leads = pd.DataFrame({"game_id": ["G"], "halftime_margin": [-4]})
    halves = pd.DataFrame({"game_id": ["G", "G"], "split": ["H1", "H2"], "margin": [-4, 2]})
    check_halftime_margins(leads, halves)
    with pytest.raises(ValueError, match="halftime"):
        check_halftime_margins(leads.assign(halftime_margin=-3), halves)


# ------------------------------------------------------------ margin series

def test_margin_series_starts_at_tipoff_and_keeps_only_changes():
    events = synthetic([(1, 720, 0, 0), (1, 700, 0, 0), (1, 650, 2, 0), (1, 640, 2, 0), (1, 600, 2, 3)])
    ms = margin_series(events)
    assert ms["margin"].tolist() == [0, 2, -1]
    assert ms["elapsed"].iloc[0] == 0


# ------------------------------------------------------- classification edges

def test_lead_exactly_at_threshold_counts():
    s = leads_for([(1, 720, 0, 0), (1, 300, 18, 10), (4, 0, 100, 105)], "L")
    assert s["max_lead_window"] == 8
    assert s["lead_class"] == "blown_lead"


def test_lead_one_below_threshold_does_not():
    s = leads_for([(1, 720, 0, 0), (1, 300, 17, 10), (4, 0, 100, 105)], "L")
    assert s["lead_class"] == "never_led_big"


def test_lead_in_q1_only_is_blown_but_not_led_at_half():
    s = leads_for([(1, 720, 0, 0), (1, 200, 20, 8), (2, 0, 50, 56), (4, 0, 98, 110)], "L")
    assert s["lead_class"] == "blown_lead"
    assert not s["led_at_half"]
    assert s["halftime_margin"] == -6


def test_lead_after_the_30_minute_window_is_ignored():
    # Up 10 with 4 minutes left in Q4 (minute 44), then lost.
    s = leads_for([(1, 720, 0, 0), (3, 0, 70, 72), (4, 240, 95, 85), (4, 0, 99, 101)], "L")
    assert s["max_lead"] == 10
    assert s["max_lead_window"] == 0
    assert s["lead_class"] == "never_led_big"


def test_overtime_win_after_big_lead_is_held():
    s = leads_for([(1, 720, 0, 0), (2, 0, 60, 48), (4, 0, 100, 100), (5, 0, 112, 108)], "W", ot=1)
    assert s["lead_class"] == "held_lead"
    assert s["final_margin"] == 4
    assert s["q4_start_margin"] == 12  # no scoring in Q3 rows, so end of Q2 carries forward


def test_threshold_sensitivity_columns():
    ms = margin_series(synthetic([(1, 720, 0, 0), (2, 100, 40, 30), (4, 0, 90, 99)]))
    s = game_leads(ms, pd.DataFrame([meta("G", "L")]), threshold=8, sensitivity=[5, 8, 12]).iloc[0]
    assert (s["lead_class_5"], s["lead_class_8"], s["lead_class_12"]) == ("blown_lead", "blown_lead", "never_led_big")


@pytest.mark.parametrize("lead, result, expected", [
    (8, "L", "blown_lead"), (8, "W", "held_lead"), (7, "W", "never_led_big"), (0, "L", "never_led_big"),
])
def test_classify(lead, result, expected):
    assert classify(lead, result, 8) == expected


def test_largest_second_half_run_allowed():
    s = leads_for([
        (1, 720, 0, 0), (2, 0, 60, 50),                       # half: +10
        (3, 600, 60, 53), (3, 500, 60, 56), (3, 400, 62, 56),  # 6-0 run, then LAC scores
        (3, 300, 62, 59), (3, 200, 62, 61), (4, 500, 62, 64), (4, 0, 70, 66),  # 8-0 run
    ], "W")
    assert s["largest_2h_run_allowed"] == 8


def test_first_half_run_is_not_counted_in_second_half():
    # 15-0 opponent run in Q1; in the second half only LAC scores.
    s = leads_for([(1, 720, 0, 0), (1, 100, 0, 15), (2, 0, 40, 30), (3, 300, 60, 30), (4, 0, 80, 30)], "W")
    assert s["largest_2h_run_allowed"] == 0
