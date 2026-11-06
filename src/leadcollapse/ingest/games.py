"""Season game logs: which games the team played, against whom, and the result.

Uses the league-wide ``LeagueGameLog`` (one call per season) rather than
``TeamGameLog`` because it returns both teams' rows for every game. That gives
the opponent's points directly and lets each game be cross-checked: the
team's PLUS_MINUS must equal its points minus the opponent's points.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import pandas as pd
from nba_api.stats.endpoints import LeagueGameLog

from leadcollapse.config import Settings
from leadcollapse.models import Game, TeamGameLogRow
from leadcollapse.nba_client import NBAClient

log = logging.getLogger(__name__)


def fetch_league_game_log(
    client: NBAClient, season: str, season_type: str, refresh: bool = False
) -> dict:
    return client.get(
        LeagueGameLog,
        refresh=refresh,
        season=season,
        player_or_team_abbreviation="T",
        season_type_all_star=season_type,
    )


def parse_game_log(raw: dict) -> list[TeamGameLogRow]:
    """Validate every row of a ``LeagueGameLog`` response."""
    result_set = next(rs for rs in raw["resultSets"] if rs["name"] == "LeagueGameLog")
    headers = result_set["headers"]
    return [TeamGameLogRow.model_validate(dict(zip(headers, row))) for row in result_set["rowSet"]]


def team_games(rows: list[TeamGameLogRow], team_id: int, season: str) -> list[Game]:
    """Pair each of the team's rows with the opponent's row; skip unfinished games."""
    by_game: dict[str, list[TeamGameLogRow]] = defaultdict(list)
    for row in rows:
        by_game[row.game_id].append(row)

    games: list[Game] = []
    for game_id, pair in by_game.items():
        team = [r for r in pair if r.team_id == team_id]
        if not team:
            continue
        others = [r for r in pair if r.team_id != team_id]
        if len(pair) != 2 or len(others) != 1:
            raise ValueError(f"{game_id}: expected 2 team rows, got {len(pair)}")
        if not team[0].is_final:
            log.info("skipping %s: not final yet", game_id)
            continue
        games.append(Game.from_rows(team[0], others[0], season))

    return sorted(games, key=lambda g: (g.game_date, g.game_id))


def games_frame(games: list[Game]) -> pd.DataFrame:
    df = pd.DataFrame([g.model_dump() | {"margin": g.margin} for g in games])
    if not df.empty:
        df["game_date"] = pd.to_datetime(df["game_date"])
    return df


def load_games(
    client: NBAClient,
    settings: Settings,
    seasons: list[str] | None = None,
    refresh_current: bool = False,
) -> pd.DataFrame:
    """All completed games for the configured team, one row per game, across seasons.

    ``refresh_current`` re-downloads the current season's log (new games get
    added to it); past seasons are complete and always come from the cache.
    """
    frames = []
    for season in seasons or settings.seasons:
        refresh = refresh_current and season == settings.current_season
        raw = fetch_league_game_log(client, season, settings.season_type, refresh=refresh)
        games = team_games(parse_game_log(raw), settings.team_id, season)
        log.info("%s: %d completed %s games", season, len(games), settings.team_abbreviation)
        frames.append(games_frame(games))
    return pd.concat(frames, ignore_index=True)


def write_games(df: pd.DataFrame, settings: Settings) -> None:
    out = settings.processed_dir
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "games.parquet", index=False)
    df.to_csv(out / "games.csv", index=False)
