"""Game-plan keys appear only when their rule says so, and are worth what the rule says."""

from __future__ import annotations

import pandas as pd
import pytest

from courtside.analysis import gameplan as GP
from courtside.analysis import preview as P


def _team(tid, name, **kw):
    base = dict(team_id=tid, team_name=name, gp=10, poss=1000, pts=1100, tov=150, fga=850, fgm=400, fg3a=350,
                fg3m=125, fta=220, ftm=170, oreb=100, dreb=330, opp_pts=1100, opp_fga=850, opp_fgm=400,
                opp_fg3a=350, opp_fg3m=125, opp_fta=220, opp_ftm=170, opp_oreb=100, opp_dreb=330, opp_tov=150)
    return base | kw


def _teams(lac_kw=None, opp_kw=None, n_others=28):
    rows = [_team(1, "Clippers", **(lac_kw or {})), _team(2, "Rivals", **(opp_kw or {}))]
    rows += [_team(100 + i, f"Team {i}") for i in range(n_others)]
    lp = pd.DataFrame([r | {"season": "2026-27", "period": q, "min": 0} for r in rows for q in (1,)])
    return P.blended(P.team_table(lp, "2026-27"), None)


def _plan(teams, misc=None, players=None):
    game = pd.Series({"home_id": 1, "away_id": 2, "home": "LAC", "away": "RIV", "neutral": False,
                      "tipoff_utc": pd.Timestamp("2026-11-20", tz="UTC"), "arena": "Intuit Dome"})
    pv = P.preview(game, teams, 1)
    misc = misc if misc is not None else pd.DataFrame({"team_id": teams["team_id"], "opp_pts_paint": 48.0})
    players = players if players is not None else pd.DataFrame(columns=[
        "player_name", "player_id", "minutes", "play_share", "ast36", "tov_pct", "reb36", "oreb36", "fg3a",
        "fg3_pct", "paint36", "fta36"])
    return pv, {k.kind: k for k in GP.game_plan(pv, teams, misc, players)}


def test_an_average_matchup_has_no_keys():
    _, keys = _plan(_teams())
    assert keys == {}


def test_threes_only_against_a_bad_three_point_defense():
    _, keys = _plan(_teams(opp_kw={"opp_fg3m": 150}))  # they allow 42.9% from three: worst in the league
    k = keys["threes"]
    # p3: the Clippers' 125/350 and what they allow, 150/350. p2: the Clippers' 275/500 and what they allow,
    # (400 - 150) / (850 - 350) = 250/500, since the same total makes now includes more threes.
    p3, p2 = (125 / 350 + 150 / 350) / 2, (275 / 500 + 250 / 500) / 2
    assert k.points == pytest.approx(GP.SWAPPED_THREES * (3 * p3 - 2 * p2))


def test_paint_counts_half_of_what_they_give_up_beyond_average():
    teams = _teams()
    misc = pd.DataFrame({"team_id": teams["team_id"], "opp_pts_paint": 48.0})
    misc.loc[misc["team_id"] == 2, "opp_pts_paint"] = 58.0
    _, keys = _plan(teams, misc)
    lg = misc["opp_pts_paint"].mean()
    assert keys["paint"].points == pytest.approx(0.5 * (58.0 - lg))


def test_turnover_key_follows_how_much_they_force():
    _, keys = _plan(_teams(opp_kw={"opp_tov": 190}))  # they force 19% vs 15% for everyone else
    assert "turnovers" in keys and keys["turnovers"].chance > 0.5
    pv, keys2 = _plan(_teams(lac_kw={"tov": 190}, opp_kw={"opp_tov": 190}))
    assert keys2["turnovers"].points > keys["turnovers"].points  # worse at home, worse against them
