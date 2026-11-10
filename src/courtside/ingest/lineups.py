"""On/off and five-man lineup splits for the team, one quarter at a time.

The dashboards take a single ``period``, so every query is made for periods
1-4 separately and the halves are built later by **summing counts** (points,
plus-minus, possessions, minutes, FGA/FTA/TOV) across Q1+Q2 and Q3+Q4 and
recomputing rates. Ratings are never averaged across quarters.

Overtime is not included: on/off and lineup halves are regulation only, the
same as the H1/H2 box-score splits.

- ``TeamPlayerOnOffDetails`` (Base + Advanced): per player, the team's stats
  with him on the court and off it.
- ``LeagueDashLineups`` (Base + Advanced + Opponent, ``group_quantity=5``):
  per lineup, the team's stats and the opponent's (for FT rate).
"""

from __future__ import annotations

import pandas as pd
from nba_api.stats.endpoints import LeagueDashLineups, TeamPlayerOnOffDetails

from courtside.nba_client import NBAClient

PERIODS = (1, 2, 3, 4)


def result_frame(raw: dict, name: str) -> pd.DataFrame:
    rs = next(r for r in raw["resultSets"] if r["name"] == name)
    return pd.DataFrame(rs["rowSet"], columns=rs["headers"])


def _onoff_raw(client: NBAClient, team_id: int, season: str, season_type: str, period: int, measure: str) -> dict:
    return client.get(
        TeamPlayerOnOffDetails, team_id=team_id, season=season, season_type_all_star=season_type,
        period=period, measure_type_detailed_defense=measure,
    )


def load_onoff(client: NBAClient, team_id: int, season: str, season_type: str) -> pd.DataFrame:
    """One row per period x player x court status (On/Off), plus a team ``Overall`` row per period."""
    frames = []
    for period in PERIODS:
        parts = {}
        for measure in ("Base", "Advanced"):
            raw = _onoff_raw(client, team_id, season, season_type, period, measure)
            on = result_frame(raw, "PlayersOnCourtTeamPlayerOnOffDetails")
            off = result_frame(raw, "PlayersOffCourtTeamPlayerOnOffDetails")
            team = result_frame(raw, "OverallTeamPlayerOnOffDetails").assign(
                VS_PLAYER_ID=0, VS_PLAYER_NAME="TEAM", COURT_STATUS="Overall"
            )
            parts[measure] = pd.concat([on, off, team], ignore_index=True)
        keys = ["VS_PLAYER_ID", "VS_PLAYER_NAME", "COURT_STATUS"]
        base = parts["Base"][keys + ["GP", "MIN", "PTS", "PLUS_MINUS", "FGA", "FTA", "TOV", "PF"]]
        adv = parts["Advanced"][keys + ["POSS"]]
        frames.append(base.merge(adv, on=keys, how="left", validate="1:1").assign(season=season, period=period))
    df = pd.concat(frames, ignore_index=True)
    return df.rename(columns={"VS_PLAYER_ID": "player_id", "VS_PLAYER_NAME": "player_name",
                              "COURT_STATUS": "court_status"}).rename(columns=str.lower)


def load_lineups(client: NBAClient, team_id: int, season: str, season_type: str) -> pd.DataFrame:
    """One row per period x five-man lineup with team and opponent counts."""
    frames = []
    for period in PERIODS:
        parts = {}
        for measure in ("Base", "Advanced", "Opponent"):
            raw = client.get(
                LeagueDashLineups, team_id_nullable=team_id, group_quantity=5, season=season,
                season_type_all_star=season_type, period=period, measure_type_detailed_defense=measure,
            )
            parts[measure] = result_frame(raw, "Lineups")
        keys = ["GROUP_ID", "GROUP_NAME"]
        base = parts["Base"][keys + ["GP", "MIN", "PTS", "PLUS_MINUS", "FGA", "FTA", "TOV", "PF"]]
        adv = parts["Advanced"][keys + ["POSS"]]
        opp = parts["Opponent"][keys + ["OPP_FGA", "OPP_FTA", "OPP_FGM", "OPP_FG3M", "OPP_PTS"]]
        df = base.merge(adv, on=keys, how="left", validate="1:1").merge(opp, on=keys, how="left", validate="1:1")
        frames.append(df.assign(season=season, period=period))
    return pd.concat(frames, ignore_index=True).rename(columns=str.lower)
