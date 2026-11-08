"""Half/quarter team-stat transform tests.

Built from the real Heat OT game fixture (LAC led 67-52 at half, lost 116-117
in OT), parsed exactly as ``build`` parses it, plus small synthetic frames for
the aggregation math.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from leadcollapse.ingest.boxscores import parse_box
from leadcollapse.transform.halves import COUNTS, add_rates, build_game_halves, summarize

FIXTURE = Path(__file__).parent / "fixtures" / "boxscore_0022600186_lac_mia_ot.json"
LAC, MIA = 1610612746, 1610612748
SPLITS = ["FULL", "H1", "H2", "OT"]


@pytest.fixture(scope="module")
def team_box() -> pd.DataFrame:
    raw = json.loads(FIXTURE.read_text(encoding="utf-8"))
    parts = {}
    for kind in ("traditional", "advanced", "misc"):
        rows = []
        for split in SPLITS:
            teams, _ = parse_box(raw[f"{kind}/{split}"], kind, split)
            rows += teams
        parts[kind] = pd.DataFrame(rows)
    keys = ["game_id", "split", "team_id", "team_abbreviation"]
    return (
        parts["traditional"]
        .merge(parts["advanced"], on=keys, validate="1:1")
        .merge(parts["misc"], on=keys, validate="1:1")
        .assign(season="2026-27")
    )


@pytest.fixture(scope="module")
def games() -> pd.DataFrame:
    return pd.DataFrame([{
        "season": "2026-27", "game_id": "0022600186", "game_date": pd.Timestamp("2026-11-04"),
        "team_id": LAC, "opponent_abbreviation": "MIA", "is_home": True, "result": "L",
    }])


@pytest.fixture(scope="module")
def gh(team_box, games) -> pd.DataFrame:
    return build_game_halves(team_box, games, LAC).set_index("split")


def test_one_row_per_split_in_order(gh):
    assert list(gh.index.astype(str)) == SPLITS


def test_margins_by_split(gh):
    assert gh.loc["H1", ["pts", "opp_pts", "margin"]].tolist() == [67, 52, 15]
    assert gh.loc["H2", ["pts", "opp_pts", "margin"]].tolist() == [40, 55, -15]
    assert gh.loc["OT", "margin"] == -1
    assert gh.loc["FULL", "margin"] == -1


def test_halves_plus_overtime_equal_full_game(gh):
    count_cols = [c for c in gh.columns if c in COUNTS or c.removeprefix("opp_") in COUNTS]
    parts = gh.loc[["H1", "H2", "OT"], count_cols].sum()
    pd.testing.assert_series_equal(parts, gh.loc["FULL", count_cols], check_names=False, check_dtype=False)


def test_opponent_columns_are_the_opponent(gh, team_box):
    mia_h2 = team_box[(team_box.team_id == MIA) & (team_box.split == "H2")].iloc[0]
    assert gh.loc["H2", "opp_pts"] == mia_h2["pts"]
    assert gh.loc["H2", "opp_fta"] == mia_h2["fta"]
    assert gh.loc["H2", "opp_possessions"] == mia_h2["possessions"]


def test_ratings_match_the_api(gh, team_box):
    api = team_box[team_box.team_id == LAC].set_index("split")
    for split in SPLITS:
        assert gh.loc[split, "ortg"] == pytest.approx(api.loc[split, "off_rating"], abs=0.05)
        assert gh.loc[split, "drtg"] == pytest.approx(api.loc[split, "def_rating"], abs=0.05)
        assert gh.loc[split, "pace"] == pytest.approx(api.loc[split, "pace"], abs=0.05)


def test_shooting_and_rate_formulas(gh):
    h2 = gh.loc["H2"]
    assert h2["efg_pct"] == pytest.approx((h2["fgm"] + 0.5 * h2["fg3m"]) / h2["fga"])
    assert h2["opp_ft_rate"] == pytest.approx(h2["opp_fta"] / h2["opp_fga"])
    assert h2["tov_pct"] == pytest.approx(100 * h2["tov"] / h2["possessions"])
    assert h2["oreb_pct"] == pytest.approx(h2["oreb"] / (h2["oreb"] + h2["opp_dreb"]))
    assert h2["net_rtg"] == pytest.approx(h2["ortg"] - h2["drtg"])


def _synthetic(**overrides) -> dict:
    row = {c: 0 for c in COUNTS} | {f"opp_{c}": 0 for c in COUNTS if c != "minutes"}
    return row | overrides


def test_summary_ratings_come_from_totals_not_averages():
    # Game A: 60 pts on 50 poss (120.0). Game B: 40 pts on 30 poss (133.3).
    # Mean of ratings = 126.7; correct pooled rating = 100 pts / 80 poss = 125.0.
    df = pd.DataFrame([
        _synthetic(game_id="A", split="H2", minutes=120, possessions=50, pts=60, opp_possessions=50, opp_pts=50),
        _synthetic(game_id="B", split="H2", minutes=120, possessions=30, pts=40, opp_possessions=30, opp_pts=30),
    ])
    s = summarize(df, ["split"]).iloc[0]
    assert s["games"] == 2
    assert s["ortg"] == pytest.approx(125.0)
    assert s["drtg"] == pytest.approx(100.0)
    assert s["margin"] == 20


def test_zero_attempts_give_nan_not_errors():
    out = add_rates(pd.DataFrame([_synthetic(minutes=60, possessions=25, opp_possessions=25, fga=10, fgm=5)]))
    assert np.isnan(out.loc[0, "fg3_pct"])
    assert out.loc[0, "fg_pct"] == 0.5


def test_official_points_off_turnovers_by_half(gh):
    # BoxScoreMiscV3: MIA scored 9 off LAC turnovers in H1, 12 in H2, 2 in OT (23 total);
    # LAC scored 13 off MIA turnovers in H1 and none after halftime.
    assert gh.loc[["H1", "H2", "OT", "FULL"], "opp_pts_off_tov"].tolist() == [9, 12, 2, 23]
    assert gh.loc[["H1", "H2", "OT", "FULL"], "pts_off_tov"].tolist() == [13, 0, 0, 13]
    assert gh.loc["H2", "opp_pts_fast_break"] == 14
