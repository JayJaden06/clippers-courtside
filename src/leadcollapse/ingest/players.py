"""Player lookups and season-level half splits for players whose history is not all in our box scores.

Darius Garland played 2025-26 for Cleveland (26 games) and then the Clippers
(19 games, plus one game for CLE against LAC). Our box scores cover his LAC
games exactly; for his Cleveland halves we take his full-season splits from
``PlayerDashboardByGameSplits`` (``ByHalfPlayerDashboard``) and subtract the
LAC games. The player id is looked up by name, never hardcoded.
"""

from __future__ import annotations

import pandas as pd
from nba_api.stats.endpoints import PlayerDashboardByGameSplits, PlayerGameLog
from nba_api.stats.static import players

from leadcollapse.nba_client import NBAClient

HALF_LABELS = {"First Half": "H1", "Second Half": "H2", "Overtime": "OT"}


def find_player_id(full_name: str) -> int:
    matches = players.find_players_by_full_name(full_name)
    exact = [m for m in matches if m["full_name"].lower() == full_name.lower()]
    if len(exact) != 1:
        raise ValueError(f"expected one player named {full_name!r}, found {[m['full_name'] for m in matches]}")
    return int(exact[0]["id"])


def _frame(raw: dict, name: str) -> pd.DataFrame:
    rs = next(r for r in raw["resultSets"] if r["name"] == name)
    return pd.DataFrame(rs["rowSet"], columns=rs["headers"])


def player_half_splits(client: NBAClient, player_id: int, season: str, season_type: str) -> pd.DataFrame:
    """Season totals by half (H1, H2, OT) for one player, across every team he played for."""
    raw = client.get(
        PlayerDashboardByGameSplits, player_id=player_id, season=season, season_type_playoffs=season_type
    )
    df = _frame(raw, "ByHalfPlayerDashboard")
    out = pd.DataFrame({
        "split": df["GROUP_VALUE"].map(HALF_LABELS),
        "games": df["GP"], "minutes": df["MIN"], "tov": df["TOV"], "fga": df["FGA"], "fta": df["FTA"],
        "pts": df["PTS"], "pf": df["PF"],
        "fgm": df["FGM"], "fg3m": df["FG3M"], "fg3a": df["FG3A"], "ftm": df["FTM"],
        "oreb": df["OREB"], "dreb": df["DREB"], "ast": df["AST"], "plus_minus": df["PLUS_MINUS"],
    })
    if out["split"].isna().any():
        raise ValueError(f"unexpected half labels: {list(df['GROUP_VALUE'])}")
    return out


def player_game_log(client: NBAClient, player_id: int, season: str, season_type: str) -> pd.DataFrame:
    """One row per game with the team he played for (first three letters of MATCHUP)."""
    raw = client.get(PlayerGameLog, player_id=player_id, season=season, season_type_all_star=season_type)
    df = _frame(raw, "PlayerGameLog")
    df["team_abbreviation"] = df["MATCHUP"].str[:3]
    return df


def player_history(client: NBAClient, names: list[str], seasons: list[str], season_type: str) -> pd.DataFrame:
    """Half-by-half totals for each named player in each season, across every team he played for.

    Seasons with no games (a rookie's earlier years) are skipped, not errors.
    """
    frames = []
    for name in names:
        pid = find_player_id(name)
        for season in seasons:
            df = player_half_splits(client, pid, season, season_type)
            if len(df):
                frames.append(df.assign(player_name=name, player_id=pid, season=season))
    cols = ["player_name", "player_id", "season", "split"]
    if not frames:
        return pd.DataFrame(columns=cols)
    out = pd.concat(frames, ignore_index=True)
    return out[cols + [c for c in out.columns if c not in cols]]
