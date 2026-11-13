"""Who's out tonight, and what it's worth: the injury adjustment to the next-game odds.

The odds come from each team's net rating, blended with last season's (``preview.blended``). Those ratings already
include every player who played. So the question for a player on the injury report is: how much of him is baked
into his team's rating, and how much does the team change without him?

- **How good he is:** his on/off, the team's points per 48 minutes scored minus allowed with him on the floor,
  minus the same with him off it, from his season plus-minus and the team's totals. This season and last are
  pooled by minutes and shrunk toward zero by ``SHRINK_MIN`` minutes, because on/off is noisy in small samples.
- **How much of him is baked in:** his share of his team's minutes, this season and last (on this team only),
  weighted the same way the ratings are (``preview.PRIOR_GAMES``). A player who's been out all season is already
  missing from this season's rating; one who just arrived isn't in last season's.
- **Worth:** baked-in share x on/off, in points a game. A missing player never makes his team better (a bench
  player's negative on/off is mostly noise and who he plays with), and one player is capped at ``PLAYER_CAP``
  points; a team's total at ``TEAM_CAP``.
- **Status:** Out counts fully, Doubtful 75%, Questionable 50%; Probable and Available not at all.

Rules set before looking at any game: these numbers were chosen for being simple and conservative, not tuned.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from courtside.analysis.preview import PRIOR_GAMES

STATUS_WEIGHT = {"Out": 1.0, "Doubtful": 0.75, "Questionable": 0.5}
SHRINK_MIN = 1000.0  # minutes of on/off evidence that count as much as the prior of "no effect"
PLAYER_CAP = 4.0
TEAM_CAP = 8.0


def name_key(name: str) -> str:
    """Accents, punctuation and case removed: "Nikola Jokić" and "Nikola Jokic" match."""
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z]", "", s.lower())


def _team_totals(league_periods: pd.DataFrame) -> pd.DataFrame:
    lp = league_periods.assign(margin=league_periods["pts"] - league_periods["opp_pts"])
    t = lp.groupby(["season", "team_id", "team_name"]).agg(games=("gp", "max"), margin=("margin", "sum")).reset_index()
    t["team_min"] = 48.0 * t["games"]
    return t


def player_values(player_season: pd.DataFrame, league_periods: pd.DataFrame, season: str,
                  prev_season: str | None) -> pd.DataFrame:
    """One row per player on each team this season or last: ``onoff`` (shrunk, points per 48), ``baked`` (his share
    of the team's blended rating), and ``worth`` (points a game the team loses without him)."""
    seasons = [s for s in (season, prev_season) if s]
    ps = player_season[player_season["season"].isin(seasons)].copy()
    if ps.empty or league_periods.empty:
        return pd.DataFrame(columns=["team_id", "team_name", "player_id", "player_name", "onoff", "baked", "worth"])
    tt = _team_totals(league_periods[league_periods["season"].isin(seasons)])
    ps = ps.merge(tt, on=["season", "team_id"], how="inner")
    ps["on_min"] = ps["gp"] * ps["min"]
    ps["on_pm"] = ps["gp"] * ps["plus_minus"]
    off_min = (ps["team_min"] - ps["on_min"]).clip(lower=48.0)
    ps["onoff48"] = 48.0 * (ps["on_pm"] / ps["on_min"].clip(lower=1.0) - (ps["margin"] - ps["on_pm"]) / off_min)
    ps.loc[ps["on_min"] <= 0, "onoff48"] = 0.0
    # how good he is: every team he played for, both seasons, pooled by minutes and shrunk toward zero
    pooled = ps.groupby("player_id").apply(
        lambda d: (d["on_min"] * d["onoff48"]).sum() / (d["on_min"].sum() + SHRINK_MIN), include_groups=False)
    ps["share"] = ps["on_min"] / ps["team_min"]
    games_now = tt[tt["season"] == season].set_index("team_id")["games"]
    rows = []
    for (tid, tname), d in ps[ps["season"] == season].groupby(["team_id", "team_name"]):
        g = float(games_now.get(tid, 0))
        prev = ps[(ps["season"] == prev_season) & (ps["team_id"] == tid)].set_index("player_id")["share"]
        now = d.set_index("player_id")
        ids = set(now.index) | set(prev.index)
        for pid in ids:
            f_now = float(now["share"].get(pid, 0.0))
            f_prev = float(prev.get(pid, 0.0))
            baked = (g * f_now + PRIOR_GAMES * f_prev) / (g + PRIOR_GAMES)
            onoff = float(pooled.get(pid, 0.0))
            name = now["player_name"].get(pid) if pid in now.index else \
                ps.loc[ps["player_id"] == pid, "player_name"].iloc[0]
            rows.append({"team_id": int(tid), "team_name": tname, "player_id": int(pid), "player_name": name,
                         "onoff": onoff, "baked": baked, "worth": min(max(baked * onoff, 0.0), PLAYER_CAP)})
    return pd.DataFrame(rows)


def missing(injuries: pd.DataFrame, values: pd.DataFrame, game_date: pd.Timestamp, matchup: str,
            team_name: str) -> pd.DataFrame:
    """The team's players on the report for this game who may not play, with what each is worth (points a game,
    already weighted by how likely he is to sit). Players who haven't played for the team count zero."""
    cols = ["player_name", "player_id", "status", "reason", "baked", "onoff", "worth", "expected"]
    if injuries is None or injuries.empty:
        return pd.DataFrame(columns=cols)
    day = pd.Timestamp(game_date).normalize()
    r = injuries[(injuries["game_date"].dt.normalize() == day) & (injuries["matchup"] == matchup)
                 & (injuries["team_name"] == team_name) & injuries["status"].isin(STATUS_WEIGHT)]
    v = values[values["team_name"] == team_name].assign(key=lambda d: d["player_name"].map(name_key))
    v = v.drop_duplicates("key").set_index("key")
    out = []
    for x in r.itertuples():
        k = name_key(x.player_name)
        worth = float(v["worth"].get(k, 0.0))
        out.append({"player_name": x.player_name, "player_id": v["player_id"].get(k), "status": x.status,
                    "reason": x.reason, "baked": float(v["baked"].get(k, 0.0)), "onoff": float(v["onoff"].get(k, 0.0)),
                    "worth": worth, "expected": worth * STATUS_WEIGHT[x.status]})
    return pd.DataFrame(out, columns=cols).sort_values("expected", ascending=False).reset_index(drop=True)


def has_report(injuries: pd.DataFrame, game_date: pd.Timestamp, matchup: str) -> bool:
    """Whether the report covers this game yet (the league posts it from the day before)."""
    if injuries is None or injuries.empty:
        return False
    day = pd.Timestamp(game_date).normalize()
    return bool(((injuries["game_date"].dt.normalize() == day) & (injuries["matchup"] == matchup)).any())


def edge(team_out: pd.DataFrame, opp_out: pd.DataFrame) -> float:
    """Points a game the injuries move the expected margin, from the team's side (positive = good for the team)."""
    return min(float(opp_out["expected"].sum()), TEAM_CAP) - min(float(team_out["expected"].sum()), TEAM_CAP)
