"""Turnover transform tests: TOV% math, categories, live-ball detection, points off turnovers,
player stats and the focus-player baseline.

Real data: the Heat OT game play-by-play and box-score fixtures. The official
BoxScoreMiscV3 counts for that game are MIA 9 / 12 / 2 points off LAC
turnovers (H1 / H2 / OT) and LAC 13 / 0 / 0 off MIA turnovers.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from courtside.ingest.boxscores import parse_box
from courtside.ingest.playbyplay import parse_pbp
from courtside.transform import turnovers as T

FIXTURES = Path(__file__).parent / "fixtures"
LAC, MIA = 1610612746, 1610612748
GARLAND = 1629636


@pytest.fixture(scope="module")
def events() -> pd.DataFrame:
    raw = json.loads((FIXTURES / "pbp_0022600186_lac_mia_ot.json").read_text(encoding="utf-8"))
    return parse_pbp(raw, LAC, is_home=True).assign(season="2026-27")


@pytest.fixture(scope="module")
def tov(events) -> pd.DataFrame:
    return T.turnover_events(events)


@pytest.fixture(scope="module")
def player_box() -> pd.DataFrame:
    raw = json.loads((FIXTURES / "boxscore_0022600186_lac_mia_ot.json").read_text(encoding="utf-8"))
    parts = {}
    for kind in ("traditional", "advanced"):
        rows = []
        for split in ("FULL",) if kind == "advanced" else ("FULL", "H1", "H2", "OT"):
            rows += parse_box(raw[f"{kind}/{split}"], kind, split)[1]
        parts[kind] = pd.DataFrame(rows)
    keys = ["game_id", "split", "team_id", "team_abbreviation", "player_id", "player_name"]
    box = parts["traditional"].merge(parts["advanced"], on=keys, how="left").assign(season="2026-27")
    box["usg_pct"] = box["usg_pct"].fillna(0.2)  # advanced players only in FULL in the fixture
    return box


# ------------------------------------------------------------------ formulas

def test_tov_pct_formula():
    assert T.tov_pct(3, 10, 5) == pytest.approx(3 / (10 + 0.44 * 5 + 3))
    assert np.isnan(T.tov_pct(0, 0, 0))
    np.testing.assert_allclose(T.tov_pct(np.array([1, 0]), np.array([4, 4]), np.array([0, 0])), [0.2, 0.0])


def test_per36():
    assert T.per36(3, 18) == pytest.approx(6.0)
    assert np.isnan(T.per36(1, 0))


@pytest.mark.parametrize("sub_type, category", [
    ("Bad Pass", "bad_pass"), ("Out of Bounds - Bad Pass Turnover", "bad_pass"),
    ("Lost Ball", "lost_ball"), ("Offensive Foul Turnover", "offensive_foul"),
    ("Shot Clock Turnover", "shot_clock"), ("Traveling", "violation"), ("3 Second Violation", "violation"),
])
def test_categorize(sub_type, category):
    assert T.categorize(sub_type) == category


# ------------------------------------------------------------ turnover events

def test_turnover_counts_by_side(tov):
    lac = tov[tov["is_team"]]
    # Box: LAC 18 player turnovers; play-by-play adds 2 shot-clock team turnovers.
    assert len(lac) == 20
    assert lac["person_id"].isna().sum() == 2
    assert set(lac.loc[lac["person_id"].isna(), "category"]) == {"shot_clock"}


def test_live_ball_turnover_detected_from_steal(tov):
    # Q1 1:20 elapsed: "Garland Lost Ball Turnover", then "Mitchell STEAL".
    first = tov[(tov["person_id"] == GARLAND)].iloc[0]
    assert first["sub_type"] == "Lost Ball" and first["live_ball"]
    # An offensive foul is always dead-ball.
    assert not tov.loc[tov["category"] == "offensive_foul", "live_ball"].any()


def test_points_off_turnovers_match_official_count(tov):
    splits = T.team_turnover_splits(tov).set_index("split")
    assert splits.loc[["H1", "H2", "OT"], "opp_pts_off_tov_est"].tolist() == [9, 12, 2]
    assert splits.loc[["H1", "H2", "OT"], "pts_off_tov_est"].tolist() == [13, 0, 0]
    assert splits.loc["FULL", "opp_pts_off_tov_est"] == 23


def test_split_columns_add_up(tov):
    s = T.team_turnover_splits(tov).set_index("split")
    assert s.loc["FULL", "tov_pbp"] == s.loc[["H1", "H2", "OT"], "tov_pbp"].sum()
    assert s.loc["H2", "tov_pbp"] == s.loc[["Q3", "Q4"], "tov_pbp"].sum()
    cats = [f"tov_{c}" for c in T.CATEGORIES]
    assert (s[cats].sum(axis=1) == s["tov_pbp"]).all()
    assert (s["opp_pts_off_live_tov"] + s["opp_pts_off_dead_tov"] == s["opp_pts_off_tov_est"]).all()


def test_shot_clock_violation_is_not_credited_as_opponent_turnover(tov):
    # Regression: before team actions were attributed, this OT shot-clock violation
    # counted as a MIA turnover and credited LAC with 7 points off it.
    ot = T.team_turnover_splits(tov).set_index("split").loc["OT"]
    assert ot["tov_shot_clock"] == 1 and ot["opp_tov_pbp"] == 1
    assert ot["pts_off_tov_est"] == 0


# ------------------------------------------------------------- player stats

def test_player_stats_match_box_and_types(player_box, tov):
    stats = T.player_turnover_stats(player_box, tov, LAC)
    T.check_player_turnovers(stats)
    g = stats[(stats["player_id"] == GARLAND)].set_index("split")
    assert g.loc["FULL", "tov"] == 6
    assert g.loc["H1", "tov"] + g.loc["H2", "tov"] + g.loc["OT", "tov"] == 6
    assert g.loc["FULL", "tov_pct"] == pytest.approx(6 / (14 + 0.44 * 2 + 6))
    assert (g["tov_live"] + g["tov_dead"] == g["tov"]).all()


def test_player_stats_only_include_the_team(player_box, tov):
    # Garland played one 2025-26 game for CLE against LAC: a player row on the other
    # team must not count as a Clippers row.
    imposter = player_box[player_box["player_id"] == GARLAND].assign(team_id=MIA, team_abbreviation="MIA")
    stats = T.player_turnover_stats(pd.concat([player_box, imposter]), tov, LAC)
    assert (stats.groupby(["split", "player_id"]).size() == 1).all()


def test_check_player_turnovers_raises_on_mismatch(player_box, tov):
    stats = T.player_turnover_stats(player_box, tov, LAC)
    stats.loc[stats.index[0], "tov"] += 1
    with pytest.raises(ValueError, match="PBP turnovers"):
        T.check_player_turnovers(stats)


def test_summarize_players_uses_totals():
    rows = pd.DataFrame([
        {"season": "s", "game_id": "A", "split": "H2", "player_id": 1, "minutes": 10, "pts": 0, "fga": 2,
         "fta": 0, "tov": 2, "pf": 0, "ast": 0, "usg_pct": 0.3, "tov_live": 1, "tov_dead": 1,
         "opp_pts_off_tov_est": 2, **{f"tov_{c}": 0 for c in T.CATEGORIES}},
        {"season": "s", "game_id": "B", "split": "H2", "player_id": 1, "minutes": 30, "pts": 0, "fga": 18,
         "fta": 0, "tov": 0, "pf": 0, "ast": 0, "usg_pct": 0.1, "tov_live": 0, "tov_dead": 0,
         "opp_pts_off_tov_est": 0, **{f"tov_{c}": 0 for c in T.CATEGORIES}},
    ])
    s = T.summarize_players(rows, ["player_id", "split"]).iloc[0]
    assert s["tov_pct"] == pytest.approx(2 / 22)          # not mean(0.5, 0.0)
    assert s["tov_per36"] == pytest.approx(36 * 2 / 40)
    assert s["usg_pct"] == pytest.approx((0.3 * 10 + 0.1 * 30) / 40)


# ----------------------------------------------------------- focus baseline

def _baseline_inputs(log_tov: int):
    stats = pd.DataFrame([
        {"season": "2025-26", "game_id": g, "split": sp, "player_id": 7, "minutes": 15, "pts": 0, "fga": 6,
         "fta": 2, "tov": 1, "pf": 0, "ast": 0, "usg_pct": 0.2, "tov_live": 1, "tov_dead": 0,
         "opp_pts_off_tov_est": 2, **{f"tov_{c}": 0 for c in T.CATEGORIES}}
        for g in ("A", "B") for sp in ("H1", "H2")
    ])
    season = pd.DataFrame({"split": ["H1", "H2"], "games": [5, 5], "minutes": [80.0, 70.0],
                           "tov": [6, 7], "fga": [30, 28], "fta": [8, 6], "pts": [0, 0], "pf": [0, 0]})
    log = pd.DataFrame({"team_abbreviation": ["CLE"] * 3 + ["LAC"] * 2,
                        # CLE totals must equal season - LAC: TOV 13-4, FGA 58-24, FTA 14-8.
                        "TOV": [log_tov, 3, 3, 2, 2], "FGA": [12, 11, 11, 12, 12], "FTA": [2, 2, 2, 4, 4]})
    return stats, season, log


def test_focus_player_other_team_is_season_minus_clippers():
    stats, season, log = _baseline_inputs(log_tov=3)
    base = T.focus_player_baselines(stats, season, log, 7, "2025-26", "LAC").set_index(["team", "split"])
    assert base.loc[("CLE", "H1"), "tov"] == 6 - 2 and base.loc[("CLE", "H2"), "tov"] == 7 - 2
    assert base.loc[("CLE", "H1"), "games"] == 3
    assert base.loc[("LAC", "H2"), "tov_pct"] == pytest.approx(2 / (12 + 0.44 * 4 + 2))


def test_focus_player_raises_when_game_log_disagrees():
    stats, season, log = _baseline_inputs(log_tov=4)
    with pytest.raises(ValueError, match="game log"):
        T.focus_player_baselines(stats, season, log, 7, "2025-26", "LAC")
