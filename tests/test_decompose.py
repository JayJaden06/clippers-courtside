"""Decomposition and bootstrap tests on hand-built halves."""

from __future__ import annotations

import pandas as pd
import pytest

from leadcollapse.analysis.decompose import FACTORS, bootstrap, decompose, decompose_frame

BASE = {"pts": 0, "fgm": 20, "fga": 45, "fg3m": 6, "ftm": 8, "fta": 10, "oreb": 5, "tov": 7, "tov_team": 0,
        "possessions": 50, "ft_trips": 5}


def half(**changes) -> pd.Series:
    """A half where both sides start from BASE; ``opp_x=`` keys change the opponent."""
    mine = BASE | {k: v for k, v in changes.items() if not k.startswith("opp_")}
    theirs = BASE | {k.removeprefix("opp_"): v for k, v in changes.items() if k.startswith("opp_")}
    for side in (mine, theirs):
        side["pts"] = 2 * side["fgm"] + side["fg3m"] + side["ftm"]
    return pd.Series(mine | {f"opp_{k}": v for k, v in theirs.items()})


def test_factors_sum_exactly_to_margin_change():
    h1 = half()
    h2 = half(fgm=16, fg3m=3, fga=47, tov=11, ftm=6, fta=8, ft_trips=4, oreb=8,
              opp_fgm=22, opp_fga=44, opp_ftm=14, opp_fta=18, opp_ft_trips=9, possessions=49, opp_possessions=49)
    d = decompose(h1, h2)
    assert d["total"] == pytest.approx(d["margin_change"])
    assert sum(d[f] for f in FACTORS) == pytest.approx(d["margin_change"])


def test_shooting_only_change_lands_in_shooting():
    # Same shots, fewer makes: every point of the drop is shooting efficiency.
    d = decompose(half(), half(fgm=16))
    assert d["margin_change"] == -8
    assert d["shooting"] == pytest.approx(-8)
    assert all(d[f] == pytest.approx(0) for f in FACTORS if f != "shooting")


def test_opponent_shooting_sign():
    d = decompose(half(), half(opp_fgm=23))  # opponent makes 3 more shots
    assert d["opp_shooting"] == pytest.approx(-6)


def test_turnovers_cost_shots_at_mean_efficiency():
    # 4 more turnovers -> 4 fewer FGA (possessions unchanged), makes scale with FGA at constant eFG.
    h2 = half(tov=11, fga=41, fgm=20 * 41 / 45, fg3m=6 * 41 / 45)
    d = decompose(half(), h2)
    efg = (20 + 3) / 45
    assert d["turnovers"] == pytest.approx(-2 * efg * 4)
    assert d["shooting"] == pytest.approx(0)
    assert d["residual"] == pytest.approx(0)


def test_free_throw_factor_nets_trips_against_made_fts():
    # Opponent gets 4 more trips (8 more FTs, 6 made) instead of 4 field-goal attempts.
    h2 = half(opp_fta=18, opp_ftm=14, opp_ft_trips=9, opp_fga=41, opp_fgm=20 * 41 / 45, opp_fg3m=6 * 41 / 45)
    d = decompose(half(), h2)
    efg = (20 + 3) / 45
    assert d["free_throws"] == pytest.approx(-(6 - 2 * efg * 4))


def test_decompose_frame_groups():
    rows = []
    for season, fgm2 in (("a", 16), ("b", 20)):
        for split, s in (("H1", half()), ("H2", half(fgm=fgm2))):
            rows.append(s.to_dict() | {"season": season, "game_id": f"{season}1", "split": split})
    out = decompose_frame(pd.DataFrame(rows), ["season"]).set_index("season")
    assert out.loc["a", "margin_change"] == -8 and out.loc["b", "margin_change"] == 0


def test_bootstrap_is_reproducible_and_brackets_the_point():
    rows = []
    for g, drop in enumerate([0, -2, -4, -6, 1, -3, -8, 2]):
        rows += [half().to_dict() | {"game_id": f"g{g}", "split": "H1"},
                 half(fgm=20 + drop).to_dict() | {"game_id": f"g{g}", "split": "H2"}]
    gh = pd.DataFrame(rows)
    stat = lambda a, b: float((b["pts"] - b["opp_pts"]) - (a["pts"] - a["opp_pts"]))  # noqa: E731
    first = bootstrap(gh, stat, n=500, seed=1)
    assert first == bootstrap(gh, stat, n=500, seed=1)
    point, lo, hi = first
    assert point == 2 * sum([0, -2, -4, -6, 1, -3, -8, 2])
    assert lo <= point <= hi
