"""Flip the sign: every generated detail sentence must read right whichever way its numbers move."""

from __future__ import annotations

import pandas as pd
import pytest

from courtside.viz import story as S

BASE = {"efg_pct": 0.53, "fg3_pct": 0.36, "opp_efg_pct": 0.53, "tov_all": 150, "opp_pts_off_tov": 160,
        "oreb_pct": 0.27, "opp_oreb_pct": 0.27, "ft_rate": 0.25, "opp_ft_rate": 0.25, "possessions": 1800}


def _halves(**after) -> pd.DataFrame:
    return pd.DataFrame({"H1": BASE, "H2": {**BASE, **after}}).T


@pytest.mark.parametrize("col", ["efg_pct", "fg3_pct", "opp_efg_pct"])
def test_rates_take_their_verb_from_the_numbers(col):
    factor = "opp_shooting" if col == "opp_efg_pct" else "shooting"
    up = S._detail(factor, _halves(**{col: BASE[col] + 0.05}), 18)
    down = S._detail(factor, _halves(**{col: BASE[col] - 0.05}), 18)
    flat = S._detail(factor, _halves(**{col: BASE[col] + 0.002}), 18)
    assert "rises" in up and "drops" in down and "holds about steady" in flat


def test_rebounding_says_who_gets_the_second_chances():
    them = S._detail("rebounding", _halves(opp_oreb_pct=0.33), 18)
    us = S._detail("rebounding", _halves(oreb_pct=0.33), 18)
    same = S._detail("rebounding", _halves(), 18)
    assert "more second chances for opponents" in them
    assert "more second chances for the Clippers" in us
    assert "about the same second chances" in same


def test_turnovers_only_say_but_when_the_count_did_not_rise():
    assert ", but opponents" in S._detail("turnovers", _halves(tov_all=150), 18)
    assert ", and opponents" in S._detail("turnovers", _halves(tov_all=200), 18)
