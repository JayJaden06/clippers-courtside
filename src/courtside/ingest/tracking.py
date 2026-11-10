"""Player-tracking numbers the regular box score doesn't have: matchups, hustle and player size.

- ``BoxScoreMatchupsV3`` (one call per game): for every offensive player, each
  defender who guarded him, for how long (partial possessions), and what
  happened: points, shots, turnovers, assists, and switches onto him. Full game
  only; the endpoint has no period filter.
- ``BoxScoreHustleV2`` (one call per game): contested shots, deflections,
  charges drawn, screen assists, loose balls and box-outs. Full game only.
- ``LeagueDashPlayerBioStats`` (one call per season): height, weight, age and
  draft year for every player, used for size in matchups and to flag rookies.

Tracking for a game is usually published a few hours after it ends. Until then
the matchup endpoint returns no players (and ``nba_api``'s parser raises on
that). Such games are skipped with a warning and nothing is cached, so the next
``pull`` picks them up.
"""

from __future__ import annotations

import logging

import pandas as pd
from nba_api.stats.endpoints import BoxScoreHustleV2, BoxScoreMatchupsV3, LeagueDashPlayerBioStats

from courtside.nba_client import CacheMissError, NBAClient

log = logging.getLogger(__name__)

MATCHUP_STATS = {
    "partialPossessions": "partial_poss", "matchupMinutesSort": "matchup_secs", "switchesOn": "switches_on",
    "playerPoints": "pts", "matchupAssists": "ast", "matchupTurnovers": "tov", "matchupBlocks": "blk",
    "matchupFieldGoalsMade": "fgm", "matchupFieldGoalsAttempted": "fga",
    "matchupThreePointersMade": "fg3m", "matchupThreePointersAttempted": "fg3a",
    "matchupFreeThrowsMade": "ftm", "matchupFreeThrowsAttempted": "fta", "shootingFouls": "shooting_fouls",
    "helpBlocks": "help_blk", "helpFieldGoalsMade": "help_fgm", "helpFieldGoalsAttempted": "help_fga",
}
HUSTLE_STATS = {
    "contestedShots": "contested_shots", "contestedShots2pt": "contested_2pt", "contestedShots3pt": "contested_3pt",
    "deflections": "deflections", "chargesDrawn": "charges_drawn", "screenAssists": "screen_assists",
    "screenAssistPoints": "screen_assist_pts", "looseBallsRecoveredTotal": "loose_balls",
    "offensiveBoxOuts": "off_box_outs", "defensiveBoxOuts": "def_box_outs", "boxOuts": "box_outs",
}


def _name(p: dict) -> str:
    return f"{p['firstName']} {p['familyName']}".strip()


def _get(client: NBAClient, endpoint, game_id: str) -> dict | None:
    try:
        return client.get(endpoint, game_id=game_id)
    except CacheMissError:  # offline and never published while pulling
        log.warning("%s for %s not in the cache; skipped", endpoint.__name__, game_id)
        return None
    except (IndexError, KeyError) as e:  # empty tracking payload: not published yet
        log.warning("%s for %s not available yet (%s); skipped", endpoint.__name__, game_id, type(e).__name__)
        return None


def parse_matchups(raw: dict) -> list[dict]:
    """One row per (offensive player, defender) pair in one game."""
    box = raw["boxScoreMatchups"]
    sides = {"homeTeam": "awayTeam", "awayTeam": "homeTeam"}
    rows = []
    for side, other in sides.items():
        off, dfn = box[side], box[other]
        for p in off["players"]:
            for m in p.get("matchups", []):
                s = m["statistics"]
                rows.append({
                    "game_id": box["gameId"],
                    "off_team": off["teamTricode"], "off_player_id": p["personId"], "off_player": _name(p),
                    "def_team": dfn["teamTricode"], "def_player_id": m["personId"], "def_player": _name(m),
                    **{v: float(s.get(k) or 0) for k, v in MATCHUP_STATS.items()},
                })
    return rows


def parse_hustle(raw: dict) -> list[dict]:
    box = raw["boxScoreHustle"]
    rows = []
    for side in ("homeTeam", "awayTeam"):
        t = box[side]
        for p in t["players"]:
            s = p["statistics"]
            mins = s.get("minutes") or "0:00"
            m, _, sec = mins.partition(":")
            rows.append({
                "game_id": box["gameId"], "team_abbreviation": t["teamTricode"],
                "player_id": p["personId"], "player_name": _name(p),
                "minutes": int(m or 0) + int(sec or 0) / 60,
                **{v: float(s.get(k) or 0) for k, v in HUSTLE_STATS.items()},
            })
    return rows


GIVE_UP_DAYS = 7  # tracking usually appears within hours; after a week it isn't coming


def load_tracking(client: NBAClient, games: pd.DataFrame, today: pd.Timestamp | None = None
                  ) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Matchups and hustle for every game. Returns ``(matchups, hustle, game ids skipped)``.

    A game older than ``GIVE_UP_DAYS`` whose tracking was never cached isn't asked for again; it comes back in
    the skipped list as ``"gave-up:<game id>"``.
    """
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    m_rows, h_rows, skipped = [], [], []
    for g in games.itertuples():
        old = (today - pd.Timestamp(g.game_date).normalize()).days > GIVE_UP_DAYS
        if old and not client.cache_path(BoxScoreMatchupsV3.__name__, {"game_id": g.game_id}).exists():
            skipped.append(f"gave-up:{g.game_id}")
            continue
        m = _get(client, BoxScoreMatchupsV3, g.game_id)
        h = _get(client, BoxScoreHustleV2, g.game_id) if m is not None else None
        if m is None or h is None or not h["boxScoreHustle"]["homeTeam"]["players"]:
            skipped.append(g.game_id)
            continue
        m_rows += [r | {"season": g.season} for r in parse_matchups(m)]
        h_rows += [r | {"season": g.season} for r in parse_hustle(h)]
    return pd.DataFrame(m_rows), pd.DataFrame(h_rows), skipped


def load_bio(client: NBAClient, season: str, season_type: str) -> pd.DataFrame:
    raw = client.get(LeagueDashPlayerBioStats, season=season, season_type_all_star=season_type)
    rs = raw["resultSets"][0]
    df = pd.DataFrame(rs["rowSet"], columns=rs["headers"])
    out = pd.DataFrame({
        "season": season, "player_id": df["PLAYER_ID"], "player_name": df["PLAYER_NAME"],
        "team_abbreviation": df["TEAM_ABBREVIATION"], "age": df["AGE"],
        "height_in": pd.to_numeric(df["PLAYER_HEIGHT_INCHES"], errors="coerce"),
        "weight_lb": pd.to_numeric(df["PLAYER_WEIGHT"], errors="coerce"),
        "draft_year": df["DRAFT_YEAR"].astype(str),
    })
    out["rookie"] = out["draft_year"] == season[:4]
    return out


def check_matchups(matchups: pd.DataFrame, player_box: pd.DataFrame) -> float:
    """Matchup points should be at most box-score points; returns the share of box points they cover.

    They are usually lower because points with no assigned defender (transition,
    broken plays, some putbacks) aren't in any matchup, and occasionally 1-3
    higher from free-throw bookkeeping. Raises if a player-game is 4+ over, or
    if coverage is implausibly low.
    """
    got = matchups.groupby(["game_id", "off_player_id"])["pts"].sum()
    box = player_box[player_box["split"].astype(str) == "FULL"].set_index(["game_id", "player_id"])["pts"]
    box = box.reindex(got.index)
    over = (got - box).dropna()
    if (over > 3).any():
        raise ValueError(f"matchup points exceed the box score:\n{over[over > 3]}")
    coverage = got.sum() / box.sum()
    if coverage < 0.7:
        raise ValueError(f"matchups cover only {coverage:.0%} of points")
    return float(coverage)
