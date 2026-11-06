"""Traditional and advanced box scores per game, for the full game, each half and each quarter.

Split stats come from the API, not from estimates: the V3 box-score endpoints
accept a period range. **They need ``range_type=1`` for that.** With the
default ``range_type=0`` the period arguments are silently ignored and the
full-game box comes back, so a "second half" request returns full-game
numbers with no error. :func:`check_split_totals` guards against that by
requiring the halves (plus OT) to add up to the full game.

Splits:

- ``FULL``: whole game, including overtime
- ``H1`` (Q1-Q2), ``H2`` (Q3-Q4): regulation halves; overtime is **not** in H2
- ``Q1``..``Q4``
- ``OT``: all overtime periods combined (only requested for OT games)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from nba_api.stats.endpoints import BoxScoreAdvancedV3, BoxScoreTraditionalV3

from leadcollapse.models import AdvancedStats, TraditionalStats
from leadcollapse.nba_client import NBAClient

log = logging.getLogger(__name__)

MAX_PERIOD = 14  # regulation (4) + up to 10 overtimes; the API ignores periods that did not happen


@dataclass(frozen=True)
class Split:
    name: str
    start_period: int | None = None
    end_period: int | None = None

    def params(self) -> dict[str, int]:
        if self.start_period is None:
            return {}
        return {
            "start_period": self.start_period,
            "end_period": self.end_period,
            "range_type": 1,  # required, see module docstring
            "start_range": 0,
            "end_range": 0,
        }


FULL = Split("FULL")
H1, H2 = Split("H1", 1, 2), Split("H2", 3, 4)
QUARTERS = [Split(f"Q{q}", q, q) for q in range(1, 5)]
OT = Split("OT", 5, MAX_PERIOD)

HALF_SPLITS = [FULL, H1, H2]
ALL_SPLITS = [FULL, H1, H2, *QUARTERS]

ENDPOINTS = {
    "traditional": (BoxScoreTraditionalV3, "boxScoreTraditional", TraditionalStats),
    "advanced": (BoxScoreAdvancedV3, "boxScoreAdvanced", AdvancedStats),
}

COUNTING_STATS = ["fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast", "stl", "blk", "tov", "pf", "pts"]


def splits_for(overtime_periods: int, quarters: bool = True) -> list[Split]:
    splits = list(ALL_SPLITS if quarters else HALF_SPLITS)
    if overtime_periods:
        splits.append(OT)
    return splits


def fetch_box(client: NBAClient, kind: str, game_id: str, split: Split) -> dict:
    endpoint, _, _ = ENDPOINTS[kind]
    return client.get(endpoint, game_id=game_id, **split.params())


def parse_box(raw: dict, kind: str, split: str) -> tuple[list[dict], list[dict]]:
    """Return (team rows, player rows) for one box-score response, validated."""
    _, key, model = ENDPOINTS[kind]
    box = raw[key]
    game_id = box["gameId"]
    teams, players = [], []
    for side in ("homeTeam", "awayTeam"):
        t = box[side]
        base = {"game_id": game_id, "split": split, "team_id": t["teamId"], "team_abbreviation": t["teamTricode"]}
        if kind == "traditional":
            base["is_home"] = side == "homeTeam"
        teams.append(base | model.model_validate(t["statistics"]).model_dump())
        for p in t["players"]:
            row = base | {"player_id": p["personId"], "player_name": f"{p['firstName']} {p['familyName']}".strip()}
            if kind == "traditional":
                row |= {"starter": bool(p.get("position")), "status_comment": p.get("comment") or ""}
            players.append(row | model.model_validate(p["statistics"]).model_dump())
    return teams, players


def load_boxscores(
    client: NBAClient, games: pd.DataFrame, quarter_seasons: list[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Team and player box scores (traditional + advanced merged) for every game and split.

    Returns ``(team_box, player_box)``: one row per game x split x team, and
    one row per game x split x player. Split responses only list players who
    appeared in that split (the FULL box also lists DNPs), so a missing
    player-split row means zero minutes, not missing data.

    ``quarter_seasons``: seasons that also get Q1-Q4; ``None`` means all seasons.
    """
    team_parts: dict[str, list] = {k: [] for k in ENDPOINTS}
    player_parts: dict[str, list] = {k: [] for k in ENDPOINTS}
    for game in games.itertuples():
        quarters = quarter_seasons is None or game.season in quarter_seasons
        for split in splits_for(game.overtime_periods, quarters):
            for kind in ENDPOINTS:
                teams, players = parse_box(fetch_box(client, kind, game.game_id, split), kind, split.name)
                for rows in (teams, players):
                    for r in rows:
                        r["season"] = game.season
                team_parts[kind] += teams
                player_parts[kind] += players

    keys = ["season", "game_id", "split", "team_id", "team_abbreviation"]
    team_box = pd.DataFrame(team_parts["traditional"]).merge(
        pd.DataFrame(team_parts["advanced"]), on=keys, how="left", validate="1:1"
    )
    player_box = pd.DataFrame(player_parts["traditional"]).merge(
        pd.DataFrame(player_parts["advanced"]), on=[*keys, "player_id", "player_name"], how="left", validate="1:1"
    )
    return team_box, player_box


def check_split_totals(team_box: pd.DataFrame) -> None:
    """Raise if H1 + H2 + OT != FULL, or Q1 + Q2 != H1, or Q3 + Q4 != H2, for any counting stat.

    This is what catches a period filter that the API silently ignored.
    """
    idx = ["game_id", "team_id"]
    by_split = {s: g.set_index(idx)[COUNTING_STATS] for s, g in team_box.groupby("split")}

    def total(*names: str) -> pd.DataFrame:
        frames = [by_split[n] for n in names if n in by_split]
        return pd.concat(frames).groupby(level=idx).sum()

    checks = [("FULL", ("H1", "H2", "OT"))]
    if "Q1" in by_split:
        checks += [("H1", ("Q1", "Q2")), ("H2", ("Q3", "Q4"))]
    for whole, parts in checks:
        expected = by_split[whole].sort_index()
        got = total(*parts).reindex(expected.index)
        diff = (got - expected).abs().sum(axis=1)
        bad = diff[diff > 0]
        if not bad.empty:
            raise ValueError(f"{' + '.join(parts)} != {whole} for {list(bad.index)}")


def write_boxscores(team_box: pd.DataFrame, player_box: pd.DataFrame, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in (("team_box", team_box), ("player_box", player_box)):
        df.to_parquet(out_dir / f"{name}.parquet", index=False)
        df.to_csv(out_dir / f"{name}.csv", index=False)
