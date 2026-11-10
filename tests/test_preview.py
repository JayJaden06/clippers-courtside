"""The next-game odds follow their written formula."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from leadcollapse.analysis import preview as P
from leadcollapse.ingest.schedule import next_game

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def test_win_chance_is_a_normal_curve_around_the_margin():
    assert P.win_chance(0) == pytest.approx(0.5)
    assert P.win_chance(P.MARGIN_SD) == pytest.approx(0.8413, abs=1e-3)  # one standard deviation
    assert P.win_chance(-5) == pytest.approx(1 - P.win_chance(5))


def _periods(team_id, name, pts, opp_pts, poss=100, gp=10, season="2026-27"):
    """Four identical quarters adding up to the given season totals."""
    base = {"season": season, "team_id": team_id, "team_name": name, "gp": gp, "min": 120, "poss": poss / 4,
            "pts": pts / 4, "opp_pts": opp_pts / 4, "fgm": 10, "fga": 22, "fg3m": 3, "fg3a": 9, "ftm": 4, "fta": 5,
            "oreb": 3, "dreb": 8, "tov": 4, "pf": 5, "opp_fgm": 10, "opp_fga": 22, "opp_fg3m": 3, "opp_fg3a": 9,
            "opp_ftm": 4, "opp_fta": 5, "opp_oreb": 3, "opp_dreb": 8, "opp_tov": 4, "opp_pf": 5}
    return [base | {"period": q} for q in range(1, 5)]


def test_ratings_blend_and_home_court():
    lp = pd.DataFrame(_periods(1, "Clippers", 1100, 1000) + _periods(2, "Suns", 1000, 1100)
                      + _periods(1, "Clippers", 1000, 1000, season="2025-26") + _periods(2, "Suns", 1000, 1000,
                                                                                       season="2025-26"))
    t = P.blended(P.team_table(lp, "2026-27"), P.team_table(lp, "2025-26")).set_index("team_id")
    assert t.loc[1, "net"] == pytest.approx(100.0)  # 1100 - 1000 points on 100 possessions
    # 10 games this season, prior weight 10: halfway between this season (+100) and last (0)
    assert t.loc[1, "blend_net"] == pytest.approx(50.0)
    game = pd.Series({"home_id": 1, "away_id": 2, "home": "LAC", "away": "PHX", "neutral": False,
                      "tipoff_utc": pd.Timestamp("2026-11-11", tz="UTC"), "arena": "Intuit Dome"})
    pv = P.preview(game, t.reset_index(), 1)
    # (50 - -50) x 10 possessions a game (100 over 10 games) / 100 + 2 home points
    assert pv.margin == pytest.approx(100 * 10 / 100 + P.HOME_POINTS)
    away = P.preview(pd.Series(game.to_dict() | {"home_id": 2, "away_id": 1, "home": "PHX", "away": "LAC"}),
                     t.reset_index(), 1)
    assert away.margin == pytest.approx(100 * 10 / 100 - P.HOME_POINTS) and not away.home


def test_next_game_skips_finished_games():
    now = pd.Timestamp("2026-11-09 12:00", tz="UTC")
    s = pd.DataFrame([
        {"game_id": "a", "status": 3, "tipoff_utc": pd.Timestamp("2026-11-08 03:00", tz="UTC"), "home_id": 1, "away_id": 2},
        {"game_id": "b", "status": 1, "tipoff_utc": pd.Timestamp("2026-11-11 02:00", tz="UTC"), "home_id": 3, "away_id": 1},
        {"game_id": "c", "status": 1, "tipoff_utc": pd.Timestamp("2026-11-13 03:00", tz="UTC"), "home_id": 1, "away_id": 4},
        {"game_id": "d", "status": 1, "tipoff_utc": pd.Timestamp("2026-11-10 03:00", tz="UTC"), "home_id": 5, "away_id": 6},
    ])
    assert next_game(s, 1, now)["game_id"] == "b"
    assert next_game(s[s["game_id"] == "a"], 1, now) is None


@pytest.mark.skipif(not (PROCESSED / "league_periods.parquet").exists(), reason="run leadcollapse build")
def test_league_table_on_the_real_data():
    lp = pd.read_parquet(PROCESSED / "league_periods.parquet")
    t = P.team_table(lp, lp["season"].max())
    assert len(t) == 30
    assert t["net"].mean() == pytest.approx(0.0, abs=0.6)  # every point scored is a point allowed
    assert t["net_rank"].between(1, 30).all()
