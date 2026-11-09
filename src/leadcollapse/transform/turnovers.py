"""Turnovers: who commits them, what kind, whether they are live-ball, and what they cost.

Sources:

- counts, minutes, usage: player box scores (exact per split)
- type, live/dead ball, and points off each turnover: play-by-play

Live-ball turnover: the play-by-play logs a separate STEAL event for the
other team at the same timestamp. That is exact (PBP steals equal box-score
steals in every game), so no guessing from the turnover type. The stealer is
kept (``stealer_id``) so turnovers can be matched to who was guarding the
player (see ``leadcollapse.viz.whatnow.pressure``).

Points off a turnover (play-by-play estimate): the other team's points from
the turnover until the committing team's next possession event (a shot or
free-throw attempt, a turnover, a rebound) or the end of the period. That
covers the possession the turnover gives away, including and-ones, free
throws and offensive rebounds. Team totals are validated against the
official ``BoxScoreMiscV3`` count; the estimate is only used where the
official number does not exist (per player, live vs dead ball).

TOV% = TOV / (FGA + 0.44 * FTA + TOV): the share of a player's own
possession-ending plays that were turnovers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from leadcollapse.transform.periods import count_by_split

CATEGORY = {
    "Bad Pass": "bad_pass",
    "Out of Bounds - Bad Pass Turnover": "bad_pass",
    "Inbound Turnover": "bad_pass",
    "Illegal Assist Turnover": "bad_pass",
    "Lost Ball": "lost_ball",
    "Out of Bounds Lost Ball Turnover": "lost_ball",
    "Offensive Foul Turnover": "offensive_foul",
    "Shot Clock Turnover": "shot_clock",
}
CATEGORIES = ["bad_pass", "lost_ball", "offensive_foul", "violation", "shot_clock"]
# Everything else (traveling, 3 seconds, backcourt, double dribble, step out of bounds,
# palming, 5/8 second, kicked ball, offensive goaltending, ...) is a violation.

POSSESSION_EVENTS = {"Made Shot", "Missed Shot", "Free Throw", "Turnover", "Rebound"}


def categorize(sub_type: str) -> str:
    return CATEGORY.get(sub_type, "violation")


def tov_pct(tov, fga, fta):
    den = fga + 0.44 * fta + tov
    return np.where(den > 0, tov / np.where(den > 0, den, 1), np.nan)


def per36(count, minutes):
    return np.where(minutes > 0, 36 * count / np.where(minutes > 0, minutes, 1), np.nan)


def _points_off(game: pd.DataFrame, i: int, committing_team: int) -> int:
    """Other team's points from turnover ``i`` until the committing team's next possession event."""
    tov = game.loc[i]
    other_score_col = "opp_score" if tov["is_team"] else "team_score"
    start = tov[other_score_col]
    later = game.loc[i + 1 :]
    ends = later[
        (later["period"] != tov["period"])
        | (later["action_type"] == "period")
        | ((later["team_id"] == committing_team) & later["action_type"].isin(POSSESSION_EVENTS))
    ]
    stop = ends.index[0] if len(ends) else game.index[-1] + 1
    window = game.loc[i : stop - 1]
    return int(window[other_score_col].iloc[-1] - start)


def turnover_events(events: pd.DataFrame) -> pd.DataFrame:
    """One row per turnover (both teams) with category, live-ball flag and points off it."""
    rows = []
    for game_id, game in events.groupby("game_id", sort=False):
        game = game.reset_index(drop=True)
        is_steal = game["description"].str.contains(r"\bSTEAL\b", regex=True)
        for i in game.index[game["action_type"] == "Turnover"]:
            t = game.loc[i]
            same_time = game[(game["elapsed"] == t["elapsed"]) & (game.index > i) & (game.index <= i + 3)]
            steals = same_time[is_steal.loc[same_time.index] & (same_time["team_id"] != t["team_id"])]
            live = len(steals) > 0
            person = t["person_id"]
            is_team_tov = bool(t["is_team_action"])
            rows.append({
                "season": t["season"], "game_id": game_id, "event_index": t["event_index"],
                "period": t["period"], "elapsed": t["elapsed"], "clock": t["clock"],
                "team_id": t["team_id"], "is_team": t["is_team"],
                "person_id": None if is_team_tov else int(person),
                "player_name": None if is_team_tov else t["player_name"],
                "sub_type": t["sub_type"], "category": categorize(t["sub_type"]),
                "live_ball": live,
                "stealer_id": int(steals["person_id"].iloc[0]) if live else None,
                "stealer_name": steals["player_name"].iloc[0] if live else None,
                "pts_off": _points_off(game, i, t["team_id"]),
                "margin_before": t["margin"],
            })
    return pd.DataFrame(rows)


def team_turnover_splits(tov_events: pd.DataFrame) -> pd.DataFrame:
    """Per game x split: turnovers, live-ball turnovers and estimated points off them, for both sides.

    Returned from the Clippers' side: ``tov_live`` etc. are Clippers turnovers,
    ``opp_tov_live`` etc. the opponent's. ``opp_pts_off_tov_est`` is points the
    opponent scored off Clippers turnovers (estimate; official is ``opp_pts_off_tov``).
    """
    flags = {
        "tov_pbp": pd.Series(1, index=tov_events.index),
        # shot-clock / 5- and 8-second team turnovers: not in the box-score TOV
        "tov_team": tov_events["person_id"].isna().astype(int),
        "tov_live": tov_events["live_ball"].astype(int),
        "pts_off_live": tov_events["pts_off"].where(tov_events["live_ball"], 0),
        "pts_off_dead": tov_events["pts_off"].where(~tov_events["live_ball"], 0),
        **{f"tov_{c}": (tov_events["category"] == c).astype(int) for c in CATEGORIES},
    }
    counts = count_by_split(tov_events, ["game_id", "is_team"], flags)
    counts["pts_off_est"] = counts["pts_off_live"] + counts["pts_off_dead"]
    mine = counts[counts["is_team"]].drop(columns="is_team")
    theirs = counts[~counts["is_team"]].drop(columns="is_team")
    # Points off *my* turnovers are scored by the opponent, so they get the opp_ prefix.
    mine = mine.rename(columns={"pts_off_live": "opp_pts_off_live_tov", "pts_off_dead": "opp_pts_off_dead_tov",
                                "pts_off_est": "opp_pts_off_tov_est"})
    theirs = theirs.rename(columns={"pts_off_live": "pts_off_live_tov", "pts_off_dead": "pts_off_dead_tov",
                                    "pts_off_est": "pts_off_tov_est"})
    theirs = theirs.rename(columns={c: f"opp_{c}" for c in theirs.columns if c.startswith("tov")})
    return mine.merge(theirs, on=["game_id", "split"], how="outer").fillna(0)


def player_turnover_stats(player_box: pd.DataFrame, tov_events: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """Per Clippers player x game x split: box-score volume plus turnover type breakdown.

    Filters on ``team_id``, not on player: the box scores hold both teams, and a
    player can appear for the opponent (Garland played CLE vs LAC in 2025-26).
    """
    box = player_box[(player_box["team_id"] == team_id) & (player_box["minutes"] > 0)]
    cols = ["season", "game_id", "split", "player_id", "player_name", "starter",
            "minutes", "pts", "fga", "fta", "tov", "pf", "ast", "usg_pct"]
    out = box[cols].copy()

    mine = tov_events[tov_events["is_team"] & tov_events["person_id"].notna()].copy()
    mine["player_id"] = mine["person_id"].astype(int)
    flags = {
        "tov_live": mine["live_ball"].astype(int),
        "opp_pts_off_tov_est": mine["pts_off"],
        **{f"tov_{c}": (mine["category"] == c).astype(int) for c in CATEGORIES},
    }
    counts = count_by_split(mine, ["game_id", "player_id"], flags)
    out = out.merge(counts, on=["game_id", "split", "player_id"], how="left")
    count_cols = list(flags)
    out[count_cols] = out[count_cols].fillna(0).astype(int)
    out["tov_dead"] = out["tov"] - out["tov_live"]
    out["tov_pct"] = tov_pct(out["tov"], out["fga"], out["fta"])
    out["tov_per36"] = per36(out["tov"], out["minutes"])
    return out


def check_player_turnovers(stats: pd.DataFrame) -> None:
    """Play-by-play turnover categories must add up to the box-score TOV for every player-split."""
    cat_total = stats[[f"tov_{c}" for c in CATEGORIES]].sum(axis=1)
    bad = stats[cat_total != stats["tov"]]
    if not bad.empty:
        raise ValueError(f"PBP turnovers != box TOV for {len(bad)} player-splits:\n"
                         f"{bad[['game_id', 'split', 'player_name', 'tov']].head()}")


def summarize_players(stats: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Sum player-split rows over games, then recompute TOV% and per-36 from the totals."""
    count_cols = ["minutes", "pts", "fga", "fta", "tov", "pf", "ast", "tov_live", "tov_dead",
                  "opp_pts_off_tov_est", *[f"tov_{c}" for c in CATEGORIES]]
    g = stats.groupby(by, observed=True)
    out = g[count_cols].sum()
    out.insert(0, "games", g["game_id"].nunique())
    # Usage: minutes-weighted mean of the per-game usage.
    out["usg_pct"] = g.apply(lambda d: np.average(d["usg_pct"], weights=d["minutes"]), include_groups=False)
    out["tov_pct"] = tov_pct(out["tov"], out["fga"], out["fta"])
    out["tov_per36"] = per36(out["tov"], out["minutes"])
    out["live_share"] = np.where(out["tov"] > 0, out["tov_live"] / out["tov"].where(out["tov"] > 0, 1), np.nan)
    return out.reset_index()


def focus_player_baselines(
    player_stats: pd.DataFrame,
    season_splits: pd.DataFrame,
    game_log: pd.DataFrame,
    player_id: int,
    comparison_season: str,
    team_abbreviation: str,
) -> pd.DataFrame:
    """Turnover rates by half for one player: every Clippers season we have, plus his other team.

    ``season_splits`` / ``game_log`` are his ``comparison_season`` dashboard halves and
    game log (all teams). Other-team halves = season totals - Clippers halves from our
    box scores. Raises if the subtraction disagrees with his game log on full-game TOV/FGA/FTA.
    """
    mine = player_stats[(player_stats["player_id"] == player_id) & player_stats["split"].isin(["H1", "H2", "OT"])]
    lac = summarize_players(mine, ["season", "split"]).assign(team=team_abbreviation)

    cols = ["games", "minutes", "tov", "fga", "fta"]
    lac_cmp = lac[lac["season"] == comparison_season].set_index("split")[cols]
    totals = season_splits.set_index("split")[cols]
    other = totals.sub(lac_cmp.reindex(totals.index).fillna(0))
    other["games"] = totals["games"] - lac_cmp["games"].reindex(totals.index).fillna(0)

    others = game_log[game_log["team_abbreviation"] != team_abbreviation]
    other_teams = sorted(others["team_abbreviation"].unique())
    log_totals = others[["TOV", "FGA", "FTA"]].sum()
    derived = other[["tov", "fga", "fta"]].sum()
    if not np.allclose(derived.to_numpy(), log_totals.to_numpy()):
        raise ValueError(f"derived other-team halves {derived.to_dict()} != game log {log_totals.to_dict()}")

    other = other.reset_index().assign(season=comparison_season, team="/".join(other_teams))
    out = pd.concat([lac, other], ignore_index=True)
    out["tov_pct"] = tov_pct(out["tov"], out["fga"], out["fta"])
    out["tov_per36"] = per36(out["tov"], out["minutes"])
    keep = ["season", "team", "split", "games", "minutes", "tov", "fga", "fta", "tov_pct", "tov_per36",
            "tov_live", "live_share"]
    return out.reindex(columns=keep).sort_values(["season", "team", "split"]).reset_index(drop=True)
