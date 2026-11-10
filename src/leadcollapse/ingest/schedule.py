"""The season schedule and every player's season-to-date stats, for the next-game preview.

- ``ScheduleLeagueV2`` (one call per season): every game, played or not, with home and away
  teams, tip-off time (UTC) and status (1 = scheduled, 2 = live, 3 = final).
- ``LeagueDashPlayerStats`` (one call per season, per game): who each team's main players are,
  so the preview can say who to watch and who should guard them.

Both change every day during the season, so ``pull`` refreshes them for the current season.
"""

from __future__ import annotations

import pandas as pd
from nba_api.stats.endpoints import LeagueDashPlayerStats, ScheduleLeagueV2

from leadcollapse.nba_client import NBAClient


def load_schedule(client: NBAClient, season: str) -> pd.DataFrame:
    raw = client.get(ScheduleLeagueV2, season=season, league_id="00")
    rows = []
    for day in raw["leagueSchedule"]["gameDates"]:
        for g in day["games"]:
            home, away = g["homeTeam"], g["awayTeam"]
            if not home.get("teamId") or not away.get("teamId"):
                continue  # TBD knockout games
            rows.append({
                "season": season, "game_id": g["gameId"], "status": int(g["gameStatus"]),
                "tipoff_utc": pd.to_datetime(g["gameDateTimeUTC"], utc=True),
                "home_id": int(home["teamId"]), "home": home["teamTricode"],
                "away_id": int(away["teamId"]), "away": away["teamTricode"],
                "home_score": home.get("score"), "away_score": away.get("score"),
                "arena": g.get("arenaName", ""), "neutral": bool(g.get("isNeutral", False)),
                "label": g.get("gameLabel", ""),
            })
    return pd.DataFrame(rows)


def next_game(schedule: pd.DataFrame, team_id: int, now: pd.Timestamp | None = None) -> pd.Series | None:
    """The team's next game that hasn't finished (status 1 or 2), or None at the end of the season."""
    now = now or pd.Timestamp.now(tz="UTC")
    mine = schedule[((schedule["home_id"] == team_id) | (schedule["away_id"] == team_id)) & (schedule["status"] < 3)]
    mine = mine[mine["tipoff_utc"] >= now - pd.Timedelta(hours=4)].sort_values("tipoff_utc")
    return mine.iloc[0] if len(mine) else None


def load_player_season(client: NBAClient, season: str, season_type: str) -> pd.DataFrame:
    raw = client.get(LeagueDashPlayerStats, season=season, season_type_all_star=season_type,
                     per_mode_detailed="PerGame")
    rs = raw["resultSets"][0]
    df = pd.DataFrame(rs["rowSet"], columns=rs["headers"])
    keep = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION", "GP", "MIN", "PTS", "REB", "AST", "TOV",
            "STL", "BLK", "FG3M", "FG3A", "FGA", "FGM", "PLUS_MINUS"]
    return df[keep].rename(columns=str.lower).assign(season=season)
