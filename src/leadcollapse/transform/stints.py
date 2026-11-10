"""Who was on the floor: rebuild every five-man unit, for both teams, from the play-by-play.

The play-by-play logs substitutions ("SUB: Santos FOR Podziemski", where ``person_id`` is the
player leaving) but not who starts each period. Period starters come from the per-quarter box
scores: a player who logged minutes in a quarter was on the floor at its start unless his first
substitution in that quarter brings him in. Overtime has one combined box split, so for OT
periods the candidates are narrowed to players who appear in that period's play-by-play.

The result is a table of *segments*: stretches with the same ten players on the floor, with the
points each side scored. It is checked against the box score: every player's minutes and
plus-minus must match (see :func:`check_segments`).
"""

from __future__ import annotations

import logging
import re
import unicodedata

import pandas as pd

log = logging.getLogger(__name__)

SUB = re.compile(r"SUB:\s*(.+?)\s+FOR\s+(.+?)\s*$")
QUARTERS = {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}


def _last_names(full: str) -> list[str]:
    """Ways the play-by-play may write a player's name: everything after the first name, and the last word."""
    parts = full.split()
    out = [" ".join(parts[1:])] if len(parts) > 1 else [full]
    out.append(parts[-1])
    if parts[-1].rstrip(".") in ("Jr", "Sr", "II", "III", "IV") and len(parts) > 2:
        out.append(parts[-2])
    return list(dict.fromkeys(out))


def _norm(name: str) -> str:
    """Lower-case, accents removed ("Dončić" -> "doncic"), so play-by-play and box names compare equal."""
    return unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower().strip()


INITIAL = re.compile(r"^([a-z]+)\.\s+(.+)$")  # "b. miller", "jal. williams": first-name prefix + last name


def resolve(name: str, roster: dict[int, str], pbp_names: dict[int, str]) -> set[int]:
    """Player ids a play-by-play name can refer to, given the team's box-score roster for the game."""
    n = _norm(name)
    m = INITIAL.match(n)
    if m:
        prefix, last = m.groups()
        return {pid for pid, full in roster.items()
                if last in {_norm(x) for x in _last_names(full)} and _norm(full).startswith(prefix)}
    ids = {pid for pid, full in roster.items() if n in {_norm(x) for x in _last_names(full)}}
    return ids | {pid for pid, pn in pbp_names.items() if _norm(pn) == n}


def _period_seconds(period: int) -> float:
    return 720.0 if period <= 4 else 300.0


def _period_start(period: int) -> float:
    return 720.0 * min(period - 1, 4) + 300.0 * max(period - 5, 0)


def game_segments(events: pd.DataFrame, box: pd.DataFrame, team_id: int) -> pd.DataFrame:
    """Segments for one game. ``events``: that game's play-by-play; ``box``: its player box rows (all splits).

    ``team_id`` is the team whose score is ``team_score`` in ``events`` (the Clippers).
    """
    ev = events.sort_values("event_index")
    game_id = ev["game_id"].iloc[0]
    teams = [team_id] + [t for t in box["team_id"].unique() if t != team_id]
    roster: dict[int, dict[int, str]] = {t: {} for t in teams}
    for r in box.drop_duplicates("player_id").itertuples():
        roster[r.team_id][int(r.player_id)] = r.player_name
    pbp_names: dict[int, dict[int, str]] = {t: {} for t in teams}
    for r in ev.dropna(subset=["person_id"]).drop_duplicates("person_id").itertuples():
        if r.team_id in pbp_names and r.action_type != "Substitution":
            pbp_names[int(r.team_id)][int(r.person_id)] = r.player_name

    def played(split: str) -> dict[int, set[int]]:
        b = box[(box["split"].astype(str) == split) & (box["minutes"] > 0)]
        return {t: set(b.loc[b["team_id"] == t, "player_id"].astype(int)) for t in teams}

    rows = []
    ended: dict[int, set[int]] = {}
    for period in sorted(ev["period"].unique()):
        pe = ev[ev["period"] == period]
        subs = pe[pe["action_type"] == "Substitution"]
        cands = played(QUARTERS.get(period, "OT"))
        on: dict[int, set[int]] = {}
        incoming: list[tuple[int, int, int]] = []  # (event_index, team, player in)
        for s in subs.itertuples():
            t = int(s.team_id)
            m = SUB.match(s.description or "")
            pin = None
            if m:
                ids = resolve(m.group(1), roster[t], pbp_names[t])
                pin = next(iter(ids)) if len(ids) == 1 else None
                if pin is None and ids:  # two players share the name: the one who played this period
                    both = ids & cands[t]
                    pin = next(iter(both)) if len(both) == 1 else None
            incoming.append((s.event_index, t, pin))
        for t in teams:
            first: dict[int, str] = {}
            for (idx, tt, pin), s in zip(incoming, subs.itertuples()):
                if tt != t:
                    continue
                first.setdefault(int(s.person_id), "out")
                if pin is not None:
                    first.setdefault(pin, "in")
            start = {p for p in cands[t] if first.get(p) != "in"}
            if period > 4:
                acted = set(pe.loc[pe["team_id"] == t, "person_id"].dropna().astype(int))
                start = {p for p in start if p in acted or first.get(p) == "out"} or start
                if len(start) < 5:  # someone played the whole period without touching the ball:
                    # he's one of the players on the floor when regulation (or the last OT) ended
                    extra = sorted((ended.get(t, set()) & cands[t]) - start)
                    start |= set(extra[: 5 - len(start)])
            if len(start) != 5:
                log.warning("%s period %s team %s: %d starters found", game_id, period, t, len(start))
            on[t] = start
        # walk the period
        t0 = _period_start(period)
        boundary = t0
        before = ev[ev["event_index"] < pe["event_index"].min()]  # the score when the period starts
        prev_score = ((int(before["team_score"].iloc[-1]), int(before["opp_score"].iloc[-1])) if len(before)
                      else (0, 0))
        sub_lookup = {idx: (t, pin) for idx, t, pin in incoming}
        fts = pe[pe["action_type"] == "Free Throw"]
        for r in pe.itertuples():
            if r.event_index in sub_lookup:
                t, pin = sub_lookup[r.event_index]
                # Free throws after a substitution at the same moment belong to the players who were on the
                # floor for the foul (the official plus-minus counts them that way), so the unit changes after them.
                later = fts[(fts["elapsed"] == r.elapsed) & (fts["event_index"] > r.event_index)]
                at = later.iloc[-1] if len(later) else None
                now = ((int(at["team_score"]), int(at["opp_score"])) if at is not None
                       else (int(r.team_score), int(r.opp_score)))
                if r.elapsed > boundary:
                    rows.append(_row(game_id, period, boundary, r.elapsed, on, teams, prev_score, now))
                    prev_score, boundary = now, r.elapsed
                on[t].discard(int(r.person_id))
                if pin is not None:
                    on[t].add(pin)
        end = t0 + _period_seconds(period)
        last = pe.iloc[-1]
        rows.append(_row(game_id, period, boundary, end, on, teams, prev_score,
                         (int(last["team_score"]), int(last["opp_score"]))))
        ended = {t: set(on[t]) for t in teams}
    return pd.DataFrame(rows)


def _row(game_id, period, start, end, on, teams, s0, s1) -> dict:
    t, o = teams
    return {"game_id": game_id, "period": int(period), "start": float(start), "end": float(end),
            "seconds": float(end - start), "team": "-".join(map(str, sorted(on[t]))),
            "opp": "-".join(map(str, sorted(on[o]))), "team_n": len(on[t]), "opp_n": len(on[o]),
            "pts": s1[0] - s0[0], "opp_pts": s1[1] - s0[1]}


def build_segments(events: pd.DataFrame, player_box: pd.DataFrame, games: pd.DataFrame, team_id: int,
                   seasons: list[str]) -> pd.DataFrame:
    """Segments for every game of the given seasons (they need per-quarter box scores)."""
    frames = []
    g = games[games["season"].isin(seasons)]
    for gid in g["game_id"]:
        frames.append(game_segments(events[events["game_id"] == gid], player_box[player_box["game_id"] == gid],
                                    team_id))
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return out.merge(g[["game_id", "season"]], on="game_id")


def player_on_court(segments: pd.DataFrame) -> pd.DataFrame:
    """Per game and player: seconds on the floor and plus-minus, for both teams."""
    rows = []
    for side, pts, opp in (("team", "pts", "opp_pts"), ("opp", "opp_pts", "pts")):
        s = segments.assign(pid=segments[side].str.split("-")).explode("pid")
        s = s[s["pid"] != ""]
        g = s.groupby(["game_id", "pid"]).agg(seconds=("seconds", "sum"), pf=(pts, "sum"), pa=(opp, "sum"))
        rows.append(g.assign(side=side).reset_index())
    out = pd.concat(rows, ignore_index=True)
    out["player_id"] = out["pid"].astype(int)
    out["plus_minus"] = out["pf"] - out["pa"]
    return out.drop(columns="pid")


def check_segments(segments: pd.DataFrame, player_box: pd.DataFrame, tol_seconds: float = 60) -> dict:
    """Compare rebuilt minutes and plus-minus with the box score. Raises if it's clearly off.

    Returns the share of player-games whose minutes match within ``tol_seconds`` and whose
    plus-minus matches exactly, and the share of game time with five players on each side.
    """
    oc = player_on_court(segments)
    box = player_box[(player_box["split"].astype(str) == "FULL") & (player_box["minutes"] > 0)
                     & player_box["game_id"].isin(set(segments["game_id"]))]
    j = box.merge(oc, on=["game_id", "player_id"], how="left")
    j["seconds"] = j["seconds"].fillna(0)
    min_ok = ((j["seconds"] - 60 * j["minutes"]).abs() <= tol_seconds).mean()
    pm_ok = (j["plus_minus_x"] == j["plus_minus_y"]).mean() if "plus_minus_x" in j else float("nan")
    five = segments.loc[(segments["team_n"] == 5) & (segments["opp_n"] == 5), "seconds"].sum() / segments[
        "seconds"].sum()
    if min_ok < 0.9 or five < 0.95:
        raise ValueError(f"on-court rebuild is off: minutes match {min_ok:.0%}, five-a-side {five:.0%} of the time")
    return {"minutes_match": float(min_ok), "plus_minus_match": float(pm_ok), "five_on_five": float(five)}
