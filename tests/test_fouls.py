"""Foul transform tests: team-foul tags, penalty reconstruction, free-throw sources, late-trailing fouls.

Real data: the Heat OT game play-by-play (box score: LAC 20 FTA / MIA 26 FTA;
by half MIA 6 / 18 / 2 and LAC 5 / 13 / 2 for H1 / H2 / OT).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from courtside.ingest.playbyplay import parse_pbp
from courtside.models import elapsed_seconds
from courtside.transform import fouls as F

FIXTURES = Path(__file__).parent / "fixtures"
LAC = 1610612746


@pytest.fixture(scope="module")
def events() -> pd.DataFrame:
    raw = json.loads((FIXTURES / "pbp_0022600186_lac_mia_ot.json").read_text(encoding="utf-8"))
    return parse_pbp(raw, LAC, is_home=True).assign(season="2026-27")


@pytest.fixture(scope="module")
def fouls(events) -> pd.DataFrame:
    return F.foul_events(events)


def foul(period: float, clock: float, desc: str, sub_type: str = "Personal", team: int = LAC, idx: int = 0,
         margin: int = 0) -> dict:
    return {"game_id": "G", "event_index": idx, "period": period, "clock": clock,
            "elapsed": elapsed_seconds(period, clock), "team_id": team, "is_team": team == LAC,
            "person_id": 1, "action_type": "Foul", "sub_type": sub_type, "description": desc, "margin": margin}


# ------------------------------------------------------------------ tags

@pytest.mark.parametrize("desc, sub_type, counts, penalty", [
    ("Lopez S.FOUL (P1.T1) (S.Mitchell)", "Shooting", True, False),
    ("Hunter L.B.FOUL (P3.PN) (B.Nansel)", "Loose Ball", True, True),
    ("Lopez OFF.Foul (P2) (M.Kallio)", "Offensive", False, False),
    ("Caldwell-Pope T.FOUL (P0.T4) (S.Corbin)", "Technical", False, False),  # echoes the team count
])
def test_team_foul_and_penalty_tags(desc, sub_type, counts, penalty):
    f = F.foul_events(pd.DataFrame([foul(1, 300, desc, sub_type)]))
    assert bool(f["counts_as_team_foul"].iloc[0]) is counts
    assert bool(f["penalty_tag"].iloc[0]) is penalty


def test_late_trailing_flag():
    rows = [
        foul(4, 90, "A P.FOUL (P1.T2)", idx=1, margin=-3),    # LAC behind, last 2:00 of Q4
        foul(4, 90, "B P.FOUL (P1.T3)", idx=2, margin=3),     # LAC ahead
        foul(4, 150, "C P.FOUL (P1.T1)", idx=3, margin=-3),   # 2:30 left
        foul(3, 60, "D P.FOUL (P1.T1)", idx=4, margin=-3),    # Q3
        foul(4, 60, "E P.FOUL (P1.T1)", team=1610612748, idx=5, margin=3),  # MIA fouls, MIA behind
    ]
    f = F.foul_events(pd.DataFrame(rows))
    assert f["late_trailing"].tolist() == [True, False, False, False, True]


# ---------------------------------------------------------------- penalty

def test_penalty_after_fourth_team_foul():
    rows = [foul(1, 700 - 60 * i, f"X P.FOUL (P1.T{i + 1})", idx=i) for i in range(5)]
    starts = F.penalty_starts(F.foul_events(pd.DataFrame(rows)))
    s = starts.iloc[0]
    assert s["penalty_start"] == elapsed_seconds(1, 700 - 60 * 3)  # the 4th foul
    assert s["bonus_secs_conceded"] == 720 - s["penalty_start"]


def test_penalty_after_first_foul_in_last_two_minutes():
    rows = [foul(2, 400, "X P.FOUL (P1.T1)", idx=0), foul(2, 100, "X P.FOUL (P1.T2)", idx=1)]
    s = F.penalty_starts(F.foul_events(pd.DataFrame(rows))).iloc[0]
    assert s["penalty_start"] == elapsed_seconds(2, 100)
    assert s["bonus_secs_conceded"] == pytest.approx(100)


def test_overtime_limit_is_three():
    rows = [foul(5, 290 - 30 * i, f"X P.FOUL (P1.T{i + 1})", idx=i) for i in range(3)]
    s = F.penalty_starts(F.foul_events(pd.DataFrame(rows))).iloc[0]
    assert s["penalty_start"] == elapsed_seconds(5, 230)
    assert s["bonus_secs_conceded"] == pytest.approx(230)


def test_no_penalty_when_under_the_limit():
    rows = [foul(3, 600, "X P.FOUL (P1.T1)", idx=0), foul(3, 300, "X P.FOUL (P1.T2)", idx=1)]
    s = F.penalty_starts(F.foul_events(pd.DataFrame(rows))).iloc[0]
    assert pd.isna(s["penalty_start"]) and s["bonus_secs_conceded"] == 0


def test_reconstruction_matches_official_tags_in_real_game(fouls):
    assert F.check_penalty_tags(fouls, F.penalty_starts(fouls)) == 1.0


# ------------------------------------------------------------ free throws

def test_free_throws_reconcile_with_box_score(events, fouls):
    fts = F.free_throw_sources(events, fouls)
    by_team = fts.groupby("shooting_team_id").size()
    assert by_team[LAC] == 20 and by_team[1610612748] == 26


def test_team_foul_splits_from_clippers_side(events, fouls):
    splits = F.team_foul_splits(fouls, F.penalty_starts(fouls), F.free_throw_sources(events, fouls)).set_index("split")
    # Free throws MIA shot (caused by LAC fouls) by half, matching the box score.
    assert splits.loc[["H1", "H2", "OT"], "opp_fta_pbp"].tolist() == [6, 18, 2]
    assert splits.loc[["H1", "H2", "OT"], "fta_pbp"].tolist() == [5, 13, 2]
    assert splits.loc["FULL", "opp_fta_pbp"] == 26
    # Every MIA free throw has a source foul category or "other".
    assert splits.loc["FULL", "opp_ftm_pbp"] >= splits.loc["FULL", "opp_ftm_late_trailing"]
    assert (splits["bonus_secs_conceded"] >= 0).all()
    assert splits.loc["FULL", "bonus_secs_conceded"] == pytest.approx(
        splits.loc[["H1", "H2", "OT"], "bonus_secs_conceded"].sum())


def test_player_foul_stats(fouls):
    stats = F.player_foul_stats(fouls)
    full = stats[stats["split"] == "FULL"]
    lac_shooting = fouls[fouls["is_team"] & fouls["is_shooting"]]
    assert full["fouls_shooting"].sum() == len(lac_shooting)
