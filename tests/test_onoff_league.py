"""On/off, lineup and league-rank transform tests on small per-quarter frames."""

from __future__ import annotations

import pandas as pd
import pytest

from leadcollapse.transform import league_context as L
from leadcollapse.transform import onoff as O


def onoff_rows(player: str, status: str, per_period: dict[int, tuple]) -> list[dict]:
    """per_period: period -> (min, pts, plus_minus, poss)."""
    return [
        {"season": "s", "period": p, "player_id": 1 if player != "TEAM" else 0, "player_name": player,
         "court_status": status, "gp": 2, "min": m, "pts": pts, "plus_minus": pm, "poss": poss,
         "fga": 10, "fta": 2, "tov": 3, "pf": 2}
        for p, (m, pts, pm, poss) in per_period.items()
    ]


@pytest.fixture
def onoff() -> pd.DataFrame:
    team = {1: (24, 60, 4, 50), 2: (24, 55, 0, 50), 3: (24, 50, -6, 50), 4: (24, 45, -10, 50)}
    on = {1: (18, 45, 6, 37), 2: (16, 40, 2, 34), 3: (20, 42, -10, 42), 4: (12, 20, -8, 25)}
    off = {p: tuple(t - o for t, o in zip(team[p], on[p])) for p in team}
    return pd.DataFrame(onoff_rows("Garland, Darius", "On", on) + onoff_rows("Garland, Darius", "Off", off)
                        + onoff_rows("TEAM", "Overall", team))


def test_halves_sum_quarters_and_recompute_ratings(onoff):
    h = O.to_halves(onoff[onoff["court_status"] == "On"], ["season", "player_id", "court_status"], O.COUNTS)
    h2 = h[h["split"] == "H2"].iloc[0]
    assert (h2["min"], h2["pts"], h2["poss"]) == (32, 62, 67)
    assert h2["ortg"] == pytest.approx(100 * 62 / 67)
    assert h2["drtg"] == pytest.approx(100 * (62 + 18) / 67)  # opp pts = pts - plus_minus
    reg = h[h["split"] == "REG"].iloc[0]
    assert reg["pts"] == 45 + 40 + 42 + 20


def test_onoff_wide_has_on_off_difference_and_readable_name(onoff):
    w = O.onoff_halves(onoff).set_index("split")
    h2 = w.loc["H2"]
    assert h2["player_name"] == "Darius Garland"
    assert h2["net_on_off"] == pytest.approx(h2["net_rtg_on"] - h2["net_rtg_off"])
    assert h2["min_on"] + h2["min_off"] == 48


def test_onoff_minutes_check(onoff):
    O.check_onoff_minutes(onoff)
    bad = onoff.copy()
    bad.loc[(bad["court_status"] == "On") & (bad["period"] == 3), "min"] += 5  # not whole games
    with pytest.raises(ValueError, match="whole games"):
        O.check_onoff_minutes(bad)


def test_lineup_minutes_check(onoff):
    lineups = pd.DataFrame({"season": "s", "period": [1, 1, 2, 3, 4], "min": [10, 14, 24, 24, 24]})
    O.check_lineup_minutes(lineups, onoff)
    with pytest.raises(ValueError, match="lineup minutes"):
        O.check_lineup_minutes(lineups.assign(min=lineups["min"] - 1), onoff)


def test_top_lineups_by_second_half_minutes():
    rows = []
    for gid, mins in (("A", 10), ("B", 30), ("C", 20)):
        for p in (3, 4):
            rows.append({"season": "s", "period": p, "group_id": gid, "group_name": gid, "min": mins / 2,
                         "pts": 20, "plus_minus": 2, "poss": 20, "fga": 15, "fta": 4, "tov": 3, "pf": 2,
                         "opp_fga": 16, "opp_fta": 8, "opp_fgm": 7, "opp_fg3m": 2})
    top = O.top_lineups(O.lineup_halves(pd.DataFrame(rows)), "s", n=2)
    assert top["group_name"].tolist() == ["B", "C"]
    assert top.loc[0, "opp_ft_rate"] == pytest.approx(16 / 32)
    assert top.loc[0, "tov_pct"] == pytest.approx(100 * 6 / 40)


# ------------------------------------------------------------------ league

def league_rows(team_id: int, h1: tuple, h2: tuple) -> list[dict]:
    """h1/h2: (pts, opp_pts, tov, opp_fta) per half, split evenly over two quarters."""
    rows = []
    for period, (pts, opp_pts, tov, opp_fta) in ((1, h1), (2, h1), (3, h2), (4, h2)):
        rows.append({"season": "s", "period": period, "team_id": team_id, "team_name": f"T{team_id}",
                     "gp": 10, "min": 120, "poss": 250, "pts": pts / 2, "fgm": 45, "fga": 100, "fg3m": 12,
                     "fta": 20, "oreb": 10, "dreb": 30, "tov": tov / 2, "pf": 20,
                     "opp_pts": opp_pts / 2, "opp_fgm": 45, "opp_fga": 100, "opp_fg3m": 12,
                     "opp_fta": opp_fta / 2, "opp_oreb": 10, "opp_dreb": 30, "opp_tov": 15})
    return rows


@pytest.fixture
def league() -> pd.DataFrame:
    return pd.DataFrame(
        league_rows(1, (560, 540, 30, 40), (500, 560, 30, 70))      # good 1H, collapses in 2H
        + league_rows(2, (520, 530, 40, 40), (520, 530, 40, 40))    # steady
        + league_rows(3, (500, 560, 50, 50), (540, 520, 20, 30))    # improves after halftime
    )


def test_half_ratings_from_summed_quarters(league):
    h = L.team_halves(league).set_index(["team_id", "split"])
    assert h.loc[(1, "H1"), "ortg"] == pytest.approx(100 * 560 / 500)
    assert h.loc[(1, "H2"), "net_rtg"] == pytest.approx(100 * (500 - 560) / 500)
    assert h.loc[(1, "H1"), "pace"] == pytest.approx(48 * 500 / 240)


def test_rank_one_is_best_for_both_directions(league):
    r = L.half_ranks(league, team_id=1).set_index(["team_id", "split", "metric"])["rank"]
    assert r[(1, "H1", "net_rtg")] == 1          # higher is better
    assert r[(3, "H1", "tov_pct")] == 3          # most turnovers is worst
    assert r[(1, "H1", "tov_pct")] == 1
    assert r[(3, "H2", "opp_ft_rate")] == 1      # fewest opponent FTs is best


def test_change_rows_rank_the_dropoff(league):
    r = L.half_ranks(league, team_id=1)
    change = r[(r["split"] == "change") & (r["metric"] == "net_rtg")].set_index("team_id")
    assert change.loc[1, "value"] == pytest.approx(-12 - 4)   # 2H net -12, 1H net +4
    assert change.loc[1, "rank"] == 3                          # biggest drop = worst
    assert change.loc[3, "rank"] == 1
    ft = r[(r["split"] == "change") & (r["metric"] == "opp_ft_rate")].set_index("team_id")
    assert ft.loc[1, "rank"] == 3                              # biggest FT-rate increase = worst
    assert r.loc[r["is_team"], "team_id"].eq(1).all()


def test_check_against_box(league):
    gh = pd.DataFrame({"season": "s", "split": ["H1", "H2"], "pts": [560, 500], "possessions": [500, 500]})
    L.check_against_box(league, gh, team_id=1)
    with pytest.raises(ValueError, match="box score"):
        L.check_against_box(league, gh.assign(pts=[560, 501]), team_id=1)
