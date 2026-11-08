"""On/off and lineup ratings by half, built from per-quarter counts.

Q1+Q2 and Q3+Q4 counts (minutes, points, plus-minus, possessions, FGA, FTA,
TOV, PF) are summed and rates recomputed. Opponent points = points - plus-minus.
Both ratings use the team's possessions (the dashboards report only those);
the two teams' possession counts differ by about one per game, so this
slightly blurs DRtg but not the on/off comparison.

"Off" means the team's minutes **without the player, in games he was on the
roster** (the NBA's definition), not the whole season. For Garland in
2025-26 that is his 33 Clippers games, not 82.

Sample size: an 8-game season gives each player 50-130 second-half minutes.
On/off differences that small are noise-dominated and are reported with
minutes alongside, never alone.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HALF_OF_PERIOD = {1: "H1", 2: "H1", 3: "H2", 4: "H2"}
COUNTS = ["min", "pts", "plus_minus", "poss", "fga", "fta", "tov", "pf"]


def _ratio(a, b):
    return (a / b.replace(0, np.nan)).astype(float)


def _rates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["opp_pts"] = out["pts"] - out["plus_minus"]
    out["ortg"] = 100 * _ratio(out["pts"], out["poss"])
    out["drtg"] = 100 * _ratio(out["opp_pts"], out["poss"])
    out["net_rtg"] = out["ortg"] - out["drtg"]
    out["tov_pct"] = 100 * _ratio(out["tov"], out["poss"])
    out["pf_per100"] = 100 * _ratio(out["pf"], out["poss"])
    if "opp_fta" in out:
        out["opp_ft_rate"] = _ratio(out["opp_fta"], out["opp_fga"])
        out["opp_efg_pct"] = _ratio(out["opp_fgm"] + 0.5 * out["opp_fg3m"], out["opp_fga"])
    return out


def to_halves(df: pd.DataFrame, keys: list[str], counts: list[str]) -> pd.DataFrame:
    """Sum per-period counts into H1/H2 (and FULL = regulation) and recompute rates."""
    df = df.assign(split=df["period"].map(HALF_OF_PERIOD))
    halves = df.groupby(keys + ["split"], as_index=False)[counts].sum()
    full = df.groupby(keys, as_index=False)[counts].sum().assign(split="REG")
    return _rates(pd.concat([halves, full], ignore_index=True))


def check_onoff_minutes(onoff: pd.DataFrame) -> None:
    """For each player and period, on + off minutes must be whole games and at most the team's minutes."""
    players = onoff[onoff["court_status"] != "Overall"]
    total = players.groupby(["season", "period", "player_id"])["min"].sum()
    team = onoff[onoff["court_status"] == "Overall"].set_index(["season", "period"])["min"]
    team = team.reindex(total.index.droplevel("player_id")).to_numpy()
    games = total / 12
    if (total.to_numpy() > team + 0.1).any() or (np.abs(games - games.round()) > 0.01).any():
        raise ValueError("on + off minutes are not whole games within the team total")


def onoff_halves(onoff: pd.DataFrame) -> pd.DataFrame:
    """One row per season x player x split with on-court and off-court ratings side by side."""
    h = to_halves(onoff[onoff["court_status"] != "Overall"],
                  ["season", "player_id", "player_name", "court_status"], COUNTS)
    wide = h.pivot_table(index=["season", "player_id", "player_name", "split"], columns="court_status",
                         values=["min", "poss", "ortg", "drtg", "net_rtg", "tov_pct", "pf_per100"])
    wide.columns = [f"{stat}_{status.lower()}" for stat, status in wide.columns]
    wide = wide.reset_index()
    wide["net_on_off"] = wide["net_rtg_on"] - wide["net_rtg_off"]
    wide["player_name"] = wide["player_name"].map(_first_last)
    return wide


def top_players(onoff_h: pd.DataFrame, season: str, split: str = "H2", n: int = 10) -> pd.DataFrame:
    """The ``n`` players with the most ``split`` minutes in ``season``."""
    s = onoff_h[(onoff_h["season"] == season) & (onoff_h["split"] == split)]
    return s.sort_values("min_on", ascending=False).head(n).reset_index(drop=True)


def lineup_halves(lineups: pd.DataFrame) -> pd.DataFrame:
    counts = COUNTS + ["opp_fga", "opp_fta", "opp_fgm", "opp_fg3m"]
    return to_halves(lineups, ["season", "group_id", "group_name"], counts)


def top_lineups(lineup_h: pd.DataFrame, season: str, split: str = "H2", n: int = 10) -> pd.DataFrame:
    s = lineup_h[(lineup_h["season"] == season) & (lineup_h["split"] == split)]
    cols = ["season", "split", "group_name", "min", "poss", "pts", "opp_pts", "plus_minus", "ortg", "drtg",
            "net_rtg", "tov_pct", "opp_ft_rate", "opp_efg_pct", "pf_per100"]
    return s.sort_values("min", ascending=False).head(n)[cols].reset_index(drop=True)


def _first_last(name: str) -> str:
    """'Garland, Darius' -> 'Darius Garland'."""
    last, _, first = name.partition(", ")
    return f"{first} {last}" if first else name


def check_lineup_minutes(lineups: pd.DataFrame, onoff: pd.DataFrame) -> None:
    """Lineup minutes must add up to the team's minutes in every season and period (all lineups returned)."""
    lu = lineups.groupby(["season", "period"])["min"].sum()
    team = onoff[onoff["court_status"] == "Overall"].set_index(["season", "period"])["min"]
    gap = (lu - team.reindex(lu.index)).abs()
    if (gap > 0.1).any():
        raise ValueError(f"lineup minutes != team minutes:\n{gap[gap > 0.1]}")
