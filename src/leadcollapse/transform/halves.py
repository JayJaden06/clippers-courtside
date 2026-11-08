"""Per-game team stats for each half and quarter, from the Clippers' point of view.

One row per game x split (FULL, H1, H2, Q1-Q4, OT) with the team's counting
stats, the opponent's (``opp_`` prefix), and rates.

Rates are always computed from summed counts, never by averaging per-game
rates: :func:`add_rates` works on a single game or on a ``groupby().sum()``
over many, so "Clippers 2H offensive rating this season" is total points per
100 total possessions, which weights each half by how many possessions it had.

Possessions are the API's own counts (``BoxScoreAdvancedV3``), which are whole
numbers and add up exactly across splits. They are used for ratings and
pace rather than the FGA + 0.44*FTA - OREB + TOV estimate.

Points off turnovers, second-chance, fast-break and paint points are the
NBA's official counts from ``BoxScoreMiscV3`` for every split.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COUNTS = [
    "minutes", "possessions", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta",
    "oreb", "dreb", "ast", "stl", "blk", "tov", "pf",
    # official scoring-context points (BoxScoreMiscV3); opp_pts_off_tov = points the
    # opponent scored off Clippers turnovers
    "pts_off_tov", "pts_2nd_chance", "pts_fast_break", "pts_paint",
]
# Play-by-play count columns that summarize() should also sum.
PBP_COUNT_PREFIXES = (
    "tov_", "opp_tov_", "pts_off_", "opp_pts_off_", "fouls_pbp", "opp_fouls_pbp", "team_fouls", "opp_team_fouls",
    "shooting_fouls", "opp_shooting_fouls", "offensive_fouls", "opp_offensive_fouls", "penalty_fouls",
    "opp_penalty_fouls", "late_trailing_fouls", "opp_late_trailing_fouls", "bonus_secs", "opp_bonus_secs",
    "fta_", "ftm_", "opp_fta_", "opp_ftm_",
)
RATE_SUFFIXES = ("_pct", "_rate", "_rate_ex_late")
GAME_COLUMNS = ["season", "game_id", "game_date", "opponent_abbreviation", "is_home", "result"]
SPLIT_ORDER = ["FULL", "H1", "H2", "Q1", "Q2", "Q3", "Q4", "OT"]


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    return (num / den.replace(0, np.nan)).astype(float)


def add_rates(df: pd.DataFrame) -> pd.DataFrame:
    """Add rate columns computed from the count columns (team and ``opp_``)."""
    out = df.copy()
    out["margin"] = out["pts"] - out["opp_pts"]
    out["ortg"] = 100 * _ratio(out["pts"], out["possessions"])
    out["drtg"] = 100 * _ratio(out["opp_pts"], out["opp_possessions"])
    out["net_rtg"] = out["ortg"] - out["drtg"]
    # Pace: possessions per 48 game minutes. Team minutes are 5x game minutes.
    game_minutes = out["minutes"] / 5
    out["pace"] = 48 * _ratio((out["possessions"] + out["opp_possessions"]) / 2, game_minutes)
    for side in ("", "opp_"):
        out[f"{side}fg_pct"] = _ratio(out[f"{side}fgm"], out[f"{side}fga"])
        out[f"{side}fg3_pct"] = _ratio(out[f"{side}fg3m"], out[f"{side}fg3a"])
        out[f"{side}efg_pct"] = _ratio(out[f"{side}fgm"] + 0.5 * out[f"{side}fg3m"], out[f"{side}fga"])
        out[f"{side}ft_rate"] = _ratio(out[f"{side}fta"], out[f"{side}fga"])
        out[f"{side}tov_pct"] = 100 * _ratio(out[f"{side}tov"], out[f"{side}possessions"])
    if "opp_fta_late_trailing" in out:
        # Opponent FT rate without free throws from late fouls by a trailing Clippers team.
        out["opp_ft_rate_ex_late"] = _ratio(out["opp_fta"] - out["opp_fta_late_trailing"], out["opp_fga"])
    out["oreb_pct"] = _ratio(out["oreb"], out["oreb"] + out["opp_dreb"])
    out["opp_oreb_pct"] = _ratio(out["opp_oreb"], out["opp_oreb"] + out["dreb"])
    return out


def build_game_halves(
    team_box: pd.DataFrame, games: pd.DataFrame, team_id: int, pbp_splits: pd.DataFrame | None = None
) -> pd.DataFrame:
    """One row per game x split with team and opponent stats side by side, plus rates.

    ``pbp_splits``: extra per game x split columns from play-by-play (turnover types,
    fouls, bonus time, free-throw sources), merged in and zero-filled.
    """
    team = team_box[team_box["team_id"] == team_id]
    opp = team_box[team_box["team_id"] != team_id]
    keys = ["game_id", "split"]
    merged = team[keys + COUNTS].merge(
        opp[keys + COUNTS].rename(columns={c: f"opp_{c}" for c in COUNTS}),
        on=keys, how="inner", validate="1:1",
    )
    if len(merged) != len(team):
        raise ValueError("some team splits have no opponent row")
    merged = merged.drop(columns="opp_minutes")

    meta = games[games["team_id"] == team_id][GAME_COLUMNS]
    out = meta.merge(merged, on="game_id", how="inner", validate="1:m")
    if pbp_splits is not None:
        extra = [c for c in pbp_splits.columns if c not in ("game_id", "split")]
        out = out.merge(pbp_splits, on=["game_id", "split"], how="left", validate="1:1")
        out[extra] = out[extra].fillna(0)
    out = add_rates(out)
    out["split"] = pd.Categorical(out["split"], categories=SPLIT_ORDER, ordered=True)
    return out.sort_values(["season", "game_date", "split"]).reset_index(drop=True)


def summarize(game_halves: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Sum counts over games within each group, then recompute rates.

    Example: ``summarize(gh, ["season", "split"])`` gives season totals per half.
    """
    count_cols = [
        c for c in game_halves.columns
        if (c in COUNTS or c.removeprefix("opp_") in COUNTS or c.startswith(PBP_COUNT_PREFIXES))
        and not c.endswith(RATE_SUFFIXES)  # tov_pct starts with "tov_" but is a rate
    ]
    grouped = game_halves.groupby(by, observed=True)
    totals = grouped[count_cols].sum()
    totals.insert(0, "games", grouped["game_id"].nunique())
    return add_rates(totals).reset_index()


def write_game_halves(df: pd.DataFrame, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "game_halves.parquet", index=False)
    df.assign(split=df["split"].astype(str)).to_csv(out_dir / "game_halves.csv", index=False)
