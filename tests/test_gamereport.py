"""On-court rebuild and the per-game report follow their rules, on made-up games and on the real ones."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from leadcollapse.analysis import gamereport as R
from leadcollapse.analysis.decompose import FACTORS
from leadcollapse.transform import stints

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
needs = pytest.mark.skipif(not (PROCESSED / "segments.parquet").exists(), reason="run leadcollapse build")


# ------------------------------------------------------------------ names

def test_resolve_handles_initials_and_accents():
    roster = {1: "Baba Miller", 2: "Jordan Miller", 3: "Luka Dončić", 4: "Derrick Jones Jr."}
    assert stints.resolve("B. Miller", roster, {}) == {1}
    assert stints.resolve("Doncic", roster, {}) == {3}
    assert stints.resolve("Jones Jr.", roster, {}) == {4}
    assert stints.resolve("Miller", roster, {}) == {1, 2}


# ------------------------------------------------------------------ a tiny made-up game

def _ev(i, period, clock, team, person, name, action, desc="", ts=0, os_=0):
    elapsed = 720 * (period - 1) + (720 - clock)
    return {"game_id": "g", "event_index": i, "period": period, "clock": clock, "elapsed": elapsed, "team_id": team,
            "person_id": person, "player_name": name, "action_type": action, "description": desc,
            "team_score": ts, "opp_score": os_}


def _box(team, ids, split, minutes=12.0):
    return [{"game_id": "g", "team_id": team, "player_id": p, "player_name": f"P{p} Last{p}", "split": split,
             "minutes": minutes} for p in ids]


def test_segments_follow_substitutions_and_score():
    lac, opp = 1, 2
    home = [11, 12, 13, 14, 15]
    away = [21, 22, 23, 24, 25]
    box = pd.DataFrame(_box(lac, home + [16], "Q1") + _box(opp, away, "Q1") + _box(lac, home + [16], "FULL")
                       + _box(opp, away, "FULL"))
    ev = pd.DataFrame([
        _ev(0, 1, 700, lac, 11, "Last11", "Made Shot", ts=2),
        _ev(1, 1, 400, lac, 15, "Last15", "Substitution", "SUB: Last16 FOR Last15", ts=2),
        _ev(2, 1, 300, opp, 21, "Last21", "Made Shot", ts=2, os_=3),
        _ev(3, 1, 0, None, None, "", "period", ts=2, os_=3),
    ])
    seg = stints.game_segments(ev, box, lac)
    assert len(seg) == 2
    first, second = seg.iloc[0], seg.iloc[1]
    assert first["team"] == "11-12-13-14-15" and second["team"] == "11-12-13-14-16"
    assert (first["seconds"], second["seconds"]) == (320.0, 400.0)
    assert (first["pts"], first["opp_pts"], second["pts"], second["opp_pts"]) == (2, 0, 0, 3)
    oc = stints.player_on_court(seg).set_index(["side", "player_id"])
    assert oc.loc[("team", 15), "plus_minus"] == 2 and oc.loc[("team", 16), "plus_minus"] == -3


# ------------------------------------------------------------------ the report's rules

def test_game_score_formula():
    b = pd.DataFrame([{"pts": 20, "fgm": 8, "fga": 15, "fta": 4, "ftm": 3, "oreb": 1, "dreb": 4, "stl": 1, "ast": 5,
                       "blk": 0, "pf": 2, "tov": 3}])
    # 20 + 3.2 - 10.5 - 0.4 + 0.7 + 1.2 + 1 + 3.5 + 0 - 0.8 - 3
    assert R.game_score(b).iloc[0] == pytest.approx(14.9)


def test_runs_are_bursts_within_seven_minutes():
    m = pd.DataFrame({"game_id": "g", "elapsed": [0, 60, 120, 180, 1500, 1560, 1620],
                      "margin": [0, 5, 10, 12, 12, 2, -1], "team_score": [0, 5, 10, 12, 30, 30, 30],
                      "opp_score": [0, 0, 0, 0, 18, 28, 31]})
    against, for_ = R.biggest_runs(m, "g")
    assert against.label == "13-0" and against.start == 1500 and against.against
    assert for_.label == "12-0" and for_.end == 180


def test_verdict_says_whether_normal_would_have_won():
    f = pd.DataFrame({"factor": FACTORS, "name": [R.FACTOR_NAMES[x] for x in FACTORS],
                      "points": [-3.0, 1.0, -6.0, -1.0, 0.0, 0.5, -0.5]})
    f.attrs = {"margin": -5.0, "normal_margin": -1.0}
    lines = pd.DataFrame(columns=["player_name", "minutes", "tov", "pf", "fga", "fgm", "efg"])
    lvs = R.levers(f, lines, pd.DataFrame(columns=["player_name", "minutes", "fga", "fta", "tov", "oreb", "dreb",
                                                   "fgm", "fg3m"]),
                   pd.DataFrame(), None, R.units(pd.DataFrame(), {}))
    assert [lv.factor for lv in lvs] == ["turnovers", "shooting"]  # biggest first; pace/residual never levers
    assert R.verdict(f, lvs).startswith("Lost by 5. A normal night on turnovers, Clippers vs opponents")
    f.attrs = {"margin": 3.0, "normal_margin": -1.0}
    assert R.verdict(f, lvs).startswith("Won by 3")


# ------------------------------------------------------------------ on the real games

@needs
def test_rebuild_matches_the_box_score():
    seg = pd.read_parquet(PROCESSED / "segments.parquet")
    pb = pd.read_parquet(PROCESSED / "player_box.parquet")
    check = stints.check_segments(seg, pb)
    assert check["minutes_match"] >= 0.99 and check["plus_minus_match"] >= 0.99 and check["five_on_five"] >= 0.99


@needs
def test_factors_add_up_to_the_margin_difference():
    gh = pd.read_parquet(PROCESSED / "game_halves.parquet")
    gid = sorted(gh.loc[gh["season"] == gh["season"].max(), "game_id"].unique())[-1]
    f = R.vs_normal(gh, gid)
    assert f["points"].sum() == pytest.approx(f.attrs["margin"] - f.attrs["normal_margin"], abs=1e-6)
