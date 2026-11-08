"""Where the Clippers rank among all 30 teams in each half, and in the 1H -> 2H change.

Per team and half (regulation only), counts are summed over Q1+Q2 / Q3+Q4 and
rates recomputed. Rank 1 is always the *best* team on that metric (highest
net rating, lowest turnover rate, ...). The ``change`` rows rank the 2H - 1H
difference the same way, so "the Clippers collapse after halftime" is
measured against how much every team's numbers move after halftime.

TOV here includes team turnovers (the league dashboards count them), so
the Clippers' TOV% differs slightly from the box-score-based game_halves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from leadcollapse.transform.onoff import HALF_OF_PERIOD

COUNTS = ["gp", "min", "poss", "pts", "fgm", "fga", "fg3m", "fta", "oreb", "dreb", "tov", "pf",
          "opp_pts", "opp_fgm", "opp_fga", "opp_fg3m", "opp_fta", "opp_oreb", "opp_dreb", "opp_tov"]

# metric -> True if higher is better
METRICS = {
    "net_rtg": True,
    "ortg": True,
    "drtg": False,
    "tov_pct": False,
    "pf_per100": False,
    "opp_ft_rate": False,
    "opp_efg_pct": False,
    "efg_pct": True,
    "oreb_pct": True,
    "opp_oreb_pct": False,
    "pace": None,  # descriptive, no better/worse; ranked high to low
}


def _ratio(a, b):
    return (a / b.replace(0, np.nan)).astype(float)


def team_halves(league: pd.DataFrame) -> pd.DataFrame:
    df = league.assign(split=league["period"].map(HALF_OF_PERIOD))
    agg = {c: "sum" for c in COUNTS if c != "gp"} | {"gp": "max"}
    h = df.groupby(["season", "team_id", "team_name", "split"], as_index=False).agg(agg)
    h["ortg"] = 100 * _ratio(h["pts"], h["poss"])
    h["drtg"] = 100 * _ratio(h["opp_pts"], h["poss"])
    h["net_rtg"] = h["ortg"] - h["drtg"]
    h["tov_pct"] = 100 * _ratio(h["tov"], h["poss"])
    h["pf_per100"] = 100 * _ratio(h["pf"], h["poss"])
    h["opp_ft_rate"] = _ratio(h["opp_fta"], h["opp_fga"])
    h["opp_efg_pct"] = _ratio(h["opp_fgm"] + 0.5 * h["opp_fg3m"], h["opp_fga"])
    h["efg_pct"] = _ratio(h["fgm"] + 0.5 * h["fg3m"], h["fga"])
    h["oreb_pct"] = _ratio(h["oreb"], h["oreb"] + h["opp_dreb"])
    h["opp_oreb_pct"] = _ratio(h["opp_oreb"], h["opp_oreb"] + h["dreb"])
    h["pace"] = 48 * _ratio(h["poss"], h["min"])
    return h


def half_ranks(league: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """Long table: season x team x split (H1, H2, change) x metric with value and rank (1 = best)."""
    h = team_halves(league)
    keys = ["season", "team_id", "team_name"]
    long = h.melt(id_vars=keys + ["split"], value_vars=list(METRICS), var_name="metric")
    wide = long.pivot_table(index=keys + ["metric"], columns="split", values="value").reset_index()
    change = wide.assign(split="change", value=wide["H2"] - wide["H1"])[keys + ["metric", "split", "value"]]
    out = pd.concat([long, change], ignore_index=True)

    # Rank 1 = best: negate higher-is-better metrics (and pace, ranked high to low), then rank ascending.
    sign = out["metric"].map(lambda m: 1 if METRICS[m] is False else -1)
    out["rank"] = (sign * out["value"]).groupby([out["season"], out["split"], out["metric"]]).rank(method="min")
    out["rank"] = out["rank"].astype(int)
    out["is_team"] = out["team_id"] == team_id
    out["league_mean"] = out.groupby(["season", "split", "metric"])["value"].transform("mean")
    return out.sort_values(["season", "metric", "split", "rank"]).reset_index(drop=True)


def check_against_box(league: pd.DataFrame, game_halves: pd.DataFrame, team_id: int) -> None:
    """The team's league-dashboard points and possessions per half must equal our box-score totals."""
    h = team_halves(league)
    mine = h[h["team_id"] == team_id].set_index(["season", "split"])[["pts", "poss"]]
    gh = game_halves[game_halves["split"].astype(str).isin(["H1", "H2"])].assign(split=lambda d: d["split"].astype(str))
    box = gh.groupby(["season", "split"])[["pts", "possessions"]].sum().rename(columns={"possessions": "poss"})
    diff = (mine - box.reindex(mine.index)).abs()
    if (diff > 0).any(axis=None):
        raise ValueError(f"league dashboard != box score totals:\n{diff}")
