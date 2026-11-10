"""League-wide team stats per quarter, for ranking the Clippers against all 30 teams.

``LeagueDashTeamStats`` takes one ``period`` at a time, so periods 1-4 are
pulled separately (Base, Advanced and Opponent measures) and summed into
halves downstream. Overtime is excluded, as in the other half splits.
"""

from __future__ import annotations

import pandas as pd
from nba_api.stats.endpoints import LeagueDashTeamStats

from courtside.ingest.lineups import PERIODS, result_frame
from courtside.nba_client import NBAClient

BASE_COLS = ["GP", "MIN", "PTS", "FGM", "FGA", "FG3M", "FG3A", "FTM", "FTA", "OREB", "DREB", "TOV", "PF"]
OPP_COLS = ["OPP_PTS", "OPP_FGM", "OPP_FGA", "OPP_FG3M", "OPP_FG3A", "OPP_FTM", "OPP_FTA", "OPP_OREB",
            "OPP_DREB", "OPP_TOV", "OPP_PF"]


def load_league(client: NBAClient, season: str, season_type: str) -> pd.DataFrame:
    """One row per period x team with team counts, opponent counts and possessions."""
    frames = []
    for period in PERIODS:
        parts = {}
        for measure in ("Base", "Advanced", "Opponent"):
            raw = client.get(
                LeagueDashTeamStats, season=season, season_type_all_star=season_type,
                period=period, measure_type_detailed_defense=measure,
            )
            parts[measure] = result_frame(raw, "LeagueDashTeamStats")
        keys = ["TEAM_ID", "TEAM_NAME"]
        df = (
            parts["Base"][keys + BASE_COLS]
            .merge(parts["Advanced"][keys + ["POSS"]], on=keys, validate="1:1")
            .merge(parts["Opponent"][keys + OPP_COLS], on=keys, validate="1:1")
        )
        if len(df) != 30:
            raise ValueError(f"{season} period {period}: expected 30 teams, got {len(df)}")
        frames.append(df.assign(season=season, period=period))
    return pd.concat(frames, ignore_index=True).rename(columns=str.lower)


MISC_COLS = ["PTS_OFF_TOV", "PTS_2ND_CHANCE", "PTS_FB", "PTS_PAINT", "OPP_PTS_OFF_TOV", "OPP_PTS_2ND_CHANCE",
             "OPP_PTS_FB", "OPP_PTS_PAINT"]


def load_team_misc(client: NBAClient, season: str, season_type: str) -> pd.DataFrame:
    """Per team, per game: paint, second-chance, fast-break and off-turnover points, scored and allowed."""
    raw = client.get(LeagueDashTeamStats, season=season, season_type_all_star=season_type,
                     measure_type_detailed_defense="Misc", per_mode_detailed="PerGame")
    df = result_frame(raw, "LeagueDashTeamStats")
    return df[["TEAM_ID", "TEAM_NAME", "GP", *MISC_COLS]].rename(columns=str.lower).assign(season=season)
