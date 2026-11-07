"""Play-by-play: one call per game to ``PlayByPlayV3``, parsed into flat event rows.

Each row gets seconds elapsed since tip-off, the running score from the
team's point of view (``team_score``, ``opp_score``, ``margin``; forward-filled
because the API only fills the score on scoring events), and flags for
whether the action belongs to the team or the opponent. Turnover and foul
subtypes are kept as-is (``sub_type``) for the turnover and foul transforms.
"""

from __future__ import annotations

import logging

import pandas as pd
from nba_api.stats.endpoints import PlayByPlayV3

from leadcollapse.models import PlayEvent
from leadcollapse.nba_client import NBAClient

log = logging.getLogger(__name__)


def fetch_pbp(client: NBAClient, game_id: str) -> dict:
    return client.get(PlayByPlayV3, game_id=game_id)


def parse_pbp(raw: dict, team_id: int, is_home: bool) -> pd.DataFrame:
    """Validated events for one game, in API order, with the running score from ``team_id``'s side."""
    game_id = raw["game"]["gameId"]
    events = [PlayEvent.model_validate(a) for a in raw["game"]["actions"]]
    df = pd.DataFrame([e.model_dump() | {"elapsed": e.elapsed} for e in events])
    df.insert(0, "game_id", game_id)
    df.insert(1, "event_index", range(len(df)))

    home = df["score_home"].ffill().fillna(0).astype(int)
    away = df["score_away"].ffill().fillna(0).astype(int)
    df["team_score"], df["opp_score"] = (home, away) if is_home else (away, home)
    df["margin"] = df["team_score"] - df["opp_score"]
    df["is_team"] = df["team_id"] == team_id
    df["is_opp"] = df["team_id"].notna() & ~df["is_team"]
    return df.drop(columns=["score_home", "score_away"])


def load_pbp(client: NBAClient, games: pd.DataFrame) -> pd.DataFrame:
    """Events for every game in ``games`` (one row per action)."""
    frames = []
    for g in games.itertuples():
        df = parse_pbp(fetch_pbp(client, g.game_id), g.team_id, g.is_home)
        df.insert(1, "season", g.season)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def check_final_scores(events: pd.DataFrame, games: pd.DataFrame) -> None:
    """The last running score of each game must equal the box-score final."""
    last = events.groupby("game_id")[["team_score", "opp_score"]].last()
    final = games.set_index("game_id")[["team_pts", "opp_pts"]]
    joined = final.join(last, how="left")
    bad = joined[(joined["team_pts"] != joined["team_score"]) | (joined["opp_pts"] != joined["opp_score"])]
    if not bad.empty:
        raise ValueError(f"play-by-play final score != game log for:\n{bad}")
