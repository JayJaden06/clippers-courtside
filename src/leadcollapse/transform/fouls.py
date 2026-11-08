"""Fouls, the bonus, and where the opponent's free throws come from.

From play-by-play:

- **Team fouls**: a foul counts toward the team total if its description
  carries the team count, ``(P2.T3)``, or the penalty tag, ``(P1.PN)``, and
  it is not a technical (some technicals echo the current team count).
  Offensive fouls carry neither tag.
- **Penalty (bonus)**: a team is over the limit after its 4th team foul in a
  regulation quarter (3rd in an overtime), or after its first team foul in
  the last 2:00 of a period if it has not reached the limit by then.
  ``bonus_secs_conceded`` is the game time the opponent spent in the bonus.
  The reconstruction is checked against the official ``PN`` tags.
- **Late fouls when trailing**: fouls by a team that is behind, in Q4 or OT,
  with 2:00 or less left. Most are intentional (stop the clock); they inflate
  second-half fouls and opponent free throws for any team that falls behind,
  so they are counted separately.
- **Free-throw attribution**: each free throw is tied to the most recent foul
  by the other team (at the same or an earlier time), so opponent free throws
  can be split into shooting, penalty, technical/flagrant, and late-trailing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from leadcollapse.transform.periods import count_by_split

REGULATION_LIMIT = 4
OVERTIME_LIMIT = 3
LAST_TWO_MINUTES = 120.0
FT_TOTALS = {"ft_against_fta", "ft_against_ftm"}
TRIP_STARTS = {"Free Throw 1 of 2", "Free Throw 1 of 3"}
TECHNICAL_LIKE = ("Technical", "Flagrant", "Clear Path", "Defense 3 Second", "Hanging", "Delay", "Away From Play")


def period_end_elapsed(period: int) -> float:
    return period * 720.0 if period <= 4 else 2880.0 + (period - 4) * 300.0


def foul_events(events: pd.DataFrame) -> pd.DataFrame:
    """One row per foul (both teams) with team-foul, penalty and late-trailing flags."""
    f = events[events["action_type"] == "Foul"].copy()
    tag = f["description"].str.extract(r"\(P\d+\.(T\d+|PN)\)")[0]
    f["counts_as_team_foul"] = tag.notna() & ~f["sub_type"].str.contains("Technical")
    f["penalty_tag"] = tag.eq("PN")
    # Official running team-foul count for the period (absent on penalty fouls).
    f["team_foul_number"] = pd.to_numeric(tag.str.removeprefix("T"), errors="coerce")
    f["is_shooting"] = f["sub_type"].eq("Shooting")
    f["is_offensive"] = f["sub_type"].str.startswith("Offensive")
    f["is_technical_like"] = f["sub_type"].str.contains("|".join(TECHNICAL_LIKE))
    # Margin from the fouling team's side, before the foul.
    fouler_margin = np.where(f["is_team"], f["margin"], -f["margin"])
    f["late_trailing"] = (f["period"] >= 4) & (f["clock"] <= LAST_TWO_MINUTES) & (fouler_margin < 0)
    f["is_take"] = f["sub_type"].str.contains("Take")
    return f


def penalty_starts(fouls: pd.DataFrame) -> pd.DataFrame:
    """When each team went over the limit in each period (NaN if it never did).

    One row per game x team x period that had a team foul.
    """
    counted = fouls[fouls["counts_as_team_foul"]].sort_values(["game_id", "event_index"])
    rows = []
    for (game_id, team_id, period), g in counted.groupby(["game_id", "team_id", "period"], sort=False):
        limit = REGULATION_LIMIT if period <= 4 else OVERTIME_LIMIT
        # Use the official count in the description rather than counting rows.
        by_limit = g.loc[g["team_foul_number"] >= limit, "elapsed"].to_numpy()
        by_limit_idx = g.loc[g["team_foul_number"] >= limit, "event_index"].to_numpy()
        late = g[g["clock"] <= LAST_TWO_MINUTES]
        candidates = [(e, i) for e, i in zip(by_limit[:1], by_limit_idx[:1])]
        candidates += list(zip(late["elapsed"].iloc[:1], late["event_index"].iloc[:1]))
        start, start_idx = min(candidates) if candidates else (np.nan, np.nan)
        rows.append({"game_id": game_id, "team_id": team_id, "period": period,
                     "penalty_start": start, "penalty_start_event": start_idx,
                     "bonus_secs_conceded": (period_end_elapsed(period) - start) if candidates else 0.0})
    cols = ["game_id", "team_id", "period", "penalty_start", "penalty_start_event", "bonus_secs_conceded"]
    return pd.DataFrame(rows, columns=cols)


def check_penalty_tags(fouls: pd.DataFrame, starts: pd.DataFrame) -> float:
    """Share of team fouls whose official PN tag agrees with the reconstructed penalty state."""
    counted = fouls[fouls["counts_as_team_foul"]].merge(starts, on=["game_id", "team_id", "period"], how="left")
    # Compare by event order, not time: two fouls can share a timestamp, and only
    # the one after the limit-reaching foul is a penalty foul.
    predicted = counted["event_index"] > counted["penalty_start_event"]
    return float((predicted == counted["penalty_tag"]).mean())


def free_throw_sources(events: pd.DataFrame, fouls: pd.DataFrame) -> pd.DataFrame:
    """Each free throw with the foul that caused it (latest other-team foul at or before it)."""
    ft = events[events["action_type"] == "Free Throw"].copy()
    ft["made"] = ~ft["description"].str.startswith("MISS")
    ft = ft.sort_values(["game_id", "event_index"])
    keep = ["game_id", "event_index", "team_id", "is_shooting", "penalty_tag", "is_technical_like",
            "late_trailing", "is_take"]
    fl = fouls[keep].rename(columns={"team_id": "fouling_team_id", "event_index": "foul_event_index"})
    out = []
    for game_id, g in ft.groupby("game_id", sort=False):
        gf = fl[fl["game_id"] == game_id].sort_values("foul_event_index")
        for _, row in g.iterrows():
            prior = gf[(gf["foul_event_index"] < row["event_index"]) & (gf["fouling_team_id"] != row["team_id"])]
            src = prior.iloc[-1] if len(prior) else None
            out.append({
                "game_id": game_id, "period": row["period"], "event_index": row["event_index"],
                "shooting_team_id": row["team_id"], "is_team": row["is_team"], "made": row["made"],
                # First shot of a 2- or 3-shot trip: the trip ends a possession. And-ones
                # ("1 of 1"), technicals, flagrants and clear-path FTs do not.
                "trip_start": row["sub_type"] in TRIP_STARTS,
                "foul_shooting": bool(src["is_shooting"]) if src is not None else False,
                "foul_penalty": bool(src["penalty_tag"]) if src is not None else False,
                "foul_technical_like": bool(src["is_technical_like"]) if src is not None else False,
                "foul_late_trailing": bool(src["late_trailing"]) if src is not None else False,
            })
    return pd.DataFrame(out)


def team_foul_splits(fouls: pd.DataFrame, starts: pd.DataFrame, fts: pd.DataFrame) -> pd.DataFrame:
    """Per game x split, from the Clippers' side.

    Clippers fouls: ``team_fouls``, ``shooting_fouls``, ``late_trailing_fouls``,
    ``bonus_secs_conceded`` (time the opponent was in the bonus) and the free
    throws they gave up: ``opp_fta_pbp``, ``opp_ftm_pbp``, ``opp_ftm_late_trailing``,
    ``opp_ftm_penalty``, ``opp_ftm_technical``. The same for the opponent's fouls
    carry the reverse prefixes (``opp_team_fouls``, ``fta_pbp``, ``ftm_late_trailing``, ...).
    """
    fflags = {
        "fouls_pbp": pd.Series(1, index=fouls.index),
        "team_fouls": fouls["counts_as_team_foul"].astype(int),
        "shooting_fouls": fouls["is_shooting"].astype(int),
        "offensive_fouls": fouls["is_offensive"].astype(int),
        "penalty_fouls": fouls["penalty_tag"].astype(int),
        "late_trailing_fouls": fouls["late_trailing"].astype(int),
    }
    fc = count_by_split(fouls, ["game_id", "is_team"], fflags)

    # Free throws shot *against* a team come from that team's fouls: key by the fouling side.
    fts = fts.assign(fouler_is_team=~fts["is_team"])
    tflags = {
        "ft_against_fta": pd.Series(1, index=fts.index),
        "ft_against_ftm": fts["made"].astype(int),
        "ft_against_ftm_late_trailing": (fts["made"] & fts["foul_late_trailing"]).astype(int),
        "ft_against_fta_late_trailing": fts["foul_late_trailing"].astype(int),
        "ft_against_ftm_penalty": (fts["made"] & fts["foul_penalty"] & ~fts["foul_shooting"]).astype(int),
        "ft_against_ftm_technical": (fts["made"] & fts["foul_technical_like"]).astype(int),
        "ft_against_ft_trips": fts["trip_start"].astype(int),
    }
    tc = count_by_split(fts.rename(columns={"is_team": "shooter_is_team", "fouler_is_team": "is_team"}),
                        ["game_id", "is_team"], tflags)

    is_team_by_id = fouls.drop_duplicates(["game_id", "team_id"]).set_index(["game_id", "team_id"])["is_team"]
    st = starts.assign(is_team=[is_team_by_id.get((g, t), False) for g, t in zip(starts["game_id"], starts["team_id"])])
    bc = count_by_split(st, ["game_id", "is_team"], {"bonus_secs_conceded": st["bonus_secs_conceded"]})

    both = fc.merge(tc, on=["game_id", "is_team", "split"], how="outer").merge(
        bc, on=["game_id", "is_team", "split"], how="outer").fillna(0)
    mine = both[both["is_team"]].drop(columns="is_team")
    theirs = both[~both["is_team"]].drop(columns="is_team")
    # Free throws shot against the Clippers are the opponent's free throws, and vice versa.
    mine = mine.rename(columns=lambda c: c.replace("ft_against_", "opp_") + ("_pbp" if c in FT_TOTALS else ""))
    theirs = theirs.rename(columns=lambda c: (
        c.replace("ft_against_", "") + ("_pbp" if c in FT_TOTALS else "") if c.startswith("ft_against_")
        else c if c in ("game_id", "split") else f"opp_{c}"
    ))
    return mine.merge(theirs, on=["game_id", "split"], how="outer").fillna(0)


def player_foul_stats(fouls: pd.DataFrame) -> pd.DataFrame:
    """Per Clippers player x game x split: shooting, offensive and late-trailing fouls."""
    mine = fouls[fouls["is_team"] & fouls["person_id"].notna()].copy()
    mine["player_id"] = mine["person_id"].astype(int)
    flags = {
        "fouls_shooting": mine["is_shooting"].astype(int),
        "fouls_offensive": mine["is_offensive"].astype(int),
        "fouls_late_trailing": mine["late_trailing"].astype(int),
    }
    return count_by_split(mine, ["game_id", "player_id"], flags)
