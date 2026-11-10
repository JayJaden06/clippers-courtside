"""One game, explained: who played well, what decided it, who guarded whom, the runs, and what needed to happen.

Everything is computed from the processed tables, with written-down rules, so every report
updates as games come in:

- **Best and worst players** by Game Score (John Hollinger's one-number box-score summary:
  PTS + 0.4 FGM - 0.7 FGA - 0.4 missed FT + 0.7 OREB + 0.3 DREB + STL + 0.7 AST + 0.7 BLK
  - 0.4 PF - TOV), with plus-minus alongside. "Worst" only considers players with 12+ minutes.
- **What decided it**: the game is compared with the team's average game this season (every
  other game), using the exact margin decomposition from :mod:`courtside.analysis.decompose`:
  shooting, opponent shooting, turnovers, rebounding, free throws and possessions. The parts add
  up to the difference between this game's margin and an average game's.
- **What needed to happen** (losses): the factors that were worse than normal, biggest first, and
  whether getting them back to normal would have covered the margin. Each comes with suggestions
  that follow simple rules (below) about who should have played more or less.
- **Runs**: the biggest stretch each way, from the running score, with the five Clippers who were
  on the floor most of it (from :mod:`courtside.transform.stints`), the turnovers and missed
  shots in it, and who scored for the opponent.

Suggestion rules:
- turnovers worse than normal: name players with 3+ turnovers, and the regular ball-handlers with
  the lowest turnover rate this season who played under 30 minutes;
- rebounding worse than normal: name the season's best rebounders per 36 who played under 20 minutes;
- shooting worse than normal: name players 15+ points of eFG below their season on 6+ shots;
- opponent shooting worse than normal: name the matchup that gave up the most points and the
  season's stingiest defender who could take it;
- free throws worse than normal: name players with 5+ fouls;
- any five-man unit that was outscored by 6+ in 3+ minutes is flagged, and any that was +6 or better.
These describe the box score and play-by-play; they don't see injuries, foul trouble or coaching plans.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from courtside.analysis.decompose import FACTORS, decompose

WORST_MIN_MINUTES = 12
LEVER_MIN_POINTS = 2.0
UNIT_MIN_SECONDS, UNIT_SWING = 180, 6
RUN_MIN = 8
RUN_MAX_SECONDS = 420  # a "run" is a burst: the biggest swing inside any 7 minutes of game clock
NOT_LEVERS = {"pace", "residual"}  # not something a team can fix
FACTOR_NAMES = {
    "shooting": "Clippers' shooting",
    "opp_shooting": "Opponents' shooting",
    "turnovers": "Turnovers, Clippers vs opponents",
    "rebounding": "Second chances, Clippers vs opponents",
    "free_throws": "Free throws, Clippers vs opponents",
    "pace": "Number of possessions",
    "residual": "Can't tell",
}


def game_score(b: pd.DataFrame) -> pd.Series:
    return (b["pts"] + 0.4 * b["fgm"] - 0.7 * b["fga"] - 0.4 * (b["fta"] - b["ftm"]) + 0.7 * b["oreb"]
            + 0.3 * b["dreb"] + b["stl"] + 0.7 * b["ast"] + 0.7 * b["blk"] - 0.4 * b["pf"] - b["tov"])


def player_lines(player_box: pd.DataFrame, game_id: str, team: str) -> pd.DataFrame:
    """Box score for one team in one game with Game Score, best first. Players who didn't play are left out."""
    b = player_box[(player_box["game_id"] == game_id) & (player_box["team_abbreviation"] == team)
                   & (player_box["split"].astype(str) == "FULL") & (player_box["minutes"] > 0)].copy()
    b["reb"] = b["oreb"] + b["dreb"]
    b["game_score"] = game_score(b)
    b["efg"] = (b["fgm"] + 0.5 * b["fg3m"]) / b["fga"].where(b["fga"] > 0)
    return b.sort_values("game_score", ascending=False).reset_index(drop=True)


def best_and_worst(lines: pd.DataFrame) -> tuple[pd.Series | None, pd.Series | None]:
    if lines.empty:
        return None, None
    regular = lines[lines["minutes"] >= WORST_MIN_MINUTES]
    return lines.iloc[0], (regular.iloc[-1] if len(regular) else lines.iloc[-1])


# ------------------------------------------------------------------ what decided it

def vs_normal(game_halves: pd.DataFrame, game_id: str) -> pd.DataFrame:
    """Each factor's points in this game compared with the team's average other game this season."""
    full = game_halves[game_halves["split"].astype(str) == "FULL"]
    season = full.loc[full["game_id"] == game_id, "season"].iloc[0]
    this = full[full["game_id"] == game_id].iloc[0]
    others = full[(full["season"] == season) & (full["game_id"] != game_id)]
    if others.empty:
        return pd.DataFrame(columns=["factor", "name", "points"])
    avg = others.select_dtypes("number").mean()
    d = decompose(avg, this)
    rows = [{"factor": f, "name": FACTOR_NAMES[f], "points": d[f]} for f in FACTORS]
    out = pd.DataFrame(rows)
    out.attrs["margin"] = float(this["pts"] - this["opp_pts"])
    out.attrs["normal_margin"] = float(avg["pts"] - avg["opp_pts"])
    return out


# ------------------------------------------------------------------ who guarded whom

def matchups_for_game(matchups: pd.DataFrame, game_id: str, team: str, opp_lines: pd.DataFrame,
                      top: int = 3) -> pd.DataFrame:
    """For the opponent's top scorers: who guarded them most, and how many points they scored on each defender."""
    m = matchups[(matchups["game_id"] == game_id) & (matchups["def_team"] == team)]
    if m.empty or opp_lines.empty:
        return pd.DataFrame()
    rows = []
    for r in opp_lines.sort_values("pts", ascending=False).head(top).itertuples():
        mine = m[m["off_player_id"] == r.player_id].sort_values("partial_poss", ascending=False)
        for d in mine.head(3).itertuples():
            rows.append({"scorer": r.player_name, "scorer_pts": r.pts, "defender": d.def_player,
                         "defender_id": d.def_player_id, "poss": d.partial_poss, "pts": d.pts,
                         "fgm": d.fgm, "fga": d.fga})
    return pd.DataFrame(rows)


def defenders_for_game(matchups: pd.DataFrame, game_id: str, team: str) -> pd.DataFrame:
    """Each Clippers defender's night: possessions guarded, points allowed, and the player he guarded most."""
    m = matchups[(matchups["game_id"] == game_id) & (matchups["def_team"] == team)]
    if m.empty:
        return pd.DataFrame()
    g = m.groupby(["def_player_id", "def_player"]).agg(poss=("partial_poss", "sum"), pts=("pts", "sum"),
                                                       fgm=("fgm", "sum"), fga=("fga", "sum")).reset_index()
    main = m.sort_values("partial_poss", ascending=False).drop_duplicates("def_player_id").set_index(
        "def_player_id")["off_player"]
    g["main_assignment"] = g["def_player_id"].map(main)
    g["pts100"] = 100 * g["pts"] / g["poss"].where(g["poss"] > 0)
    return g.sort_values("pts", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ runs

@dataclass
class Run:
    start: float
    end: float
    team_pts: int
    opp_pts: int
    against: bool  # True = the opponent's run
    unit: list[str] = field(default_factory=list)  # Clippers on the floor most of the run
    turnovers: list[str] = field(default_factory=list)
    misses: int = 0
    opp_scorers: list[tuple[str, int]] = field(default_factory=list)
    opp_orebs: int = 0

    @property
    def label(self) -> str:
        return f"{self.opp_pts}-{self.team_pts}" if self.against else f"{self.team_pts}-{self.opp_pts}"


def _clock(elapsed: float) -> str:
    """'Q3 8:10' from seconds since tip-off."""
    if elapsed < 2880:
        q, into = int(elapsed // 720) + 1, elapsed % 720
        left = 720 - into
    else:
        q, into = 5 + int((elapsed - 2880) // 300), (elapsed - 2880) % 300
        left = 300 - into
    name = f"Q{q}" if q <= 4 else ("OT" if q == 5 else f"{q - 4}OT")
    return f"{name} {int(left // 60)}:{int(left % 60):02d}"


def run_window(run: Run) -> str:
    return f"{_clock(run.start)} to {_clock(run.end)}"


def biggest_runs(margins: pd.DataFrame, game_id: str,
                 max_seconds: float = RUN_MAX_SECONDS) -> tuple[Run | None, Run | None]:
    """The biggest swing against the Clippers and in their favor, each within ``max_seconds`` of game clock."""
    m = margins[margins["game_id"] == game_id].sort_values("elapsed")
    if m.empty:
        return None, None
    el, mg = m["elapsed"].to_numpy(), m["margin"].to_numpy()
    ts, os_ = m["team_score"].to_numpy(), m["opp_score"].to_numpy()

    def best(sign: int) -> Run | None:
        best_val, best_ij, i0 = 0, None, 0
        for j in range(len(mg)):
            while el[j] - el[i0] > max_seconds:
                i0 += 1
            window = sign * (mg[j] - mg[i0:j + 1])
            k = int(np.argmax(window))
            if window[k] > best_val:
                best_val, best_ij = window[k], (i0 + k, j)
        if best_ij is None or best_val < RUN_MIN:
            return None
        i, j = best_ij
        return Run(float(el[i]), float(el[j]), int(ts[j] - ts[i]), int(os_[j] - os_[i]), against=sign < 0)

    return best(-1), best(+1)


def describe_run(run: Run, plays: pd.DataFrame, segments: pd.DataFrame, names: dict[int, str]) -> Run:
    """Fill in who was on the floor, the turnovers, misses and opponent scorers during the run."""
    p = plays[(plays["elapsed"] > run.start) & (plays["elapsed"] <= run.end)]
    team = p[p["is_team"]]
    opp = p[~p["is_team"] & p["team_id"].notna()]
    run.turnovers = list(team.loc[team["action_type"] == "Turnover", "player_name"].dropna())
    run.misses = int((team["action_type"] == "Missed Shot").sum())
    made = opp[opp["action_type"].isin(["Made Shot", "Free Throw"]) & ~opp["description"].str.contains("MISS", na=False)]
    pts = made.assign(v=np.where(made["action_type"] == "Free Throw", 1, made["shot_value"].fillna(2)))
    run.opp_scorers = [(n, int(v)) for n, v in pts.groupby("player_name")["v"].sum().sort_values(
        ascending=False).items()][:3]
    shots = p[p["action_type"].isin(["Missed Shot", "Free Throw", "Rebound"])]
    prev_miss_team = None
    for r in shots.itertuples():
        if r.action_type == "Missed Shot" or (r.action_type == "Free Throw" and "MISS" in (r.description or "")):
            prev_miss_team = r.team_id
        elif r.action_type == "Rebound" and not r.is_team and r.team_id == prev_miss_team and pd.notna(r.team_id):
            run.opp_orebs += 1
    seg = segments[(segments["end"] > run.start) & (segments["start"] < run.end)].copy()
    if len(seg):
        seg["overlap"] = np.minimum(seg["end"], run.end) - np.maximum(seg["start"], run.start)
        unit = seg.groupby("team")["overlap"].sum().idxmax()
        run.unit = [names.get(int(x), x) for x in unit.split("-") if x]
    return run


# ------------------------------------------------------------------ five-man units

def units(segments: pd.DataFrame, names: dict[int, str]) -> pd.DataFrame:
    """Each Clippers five-man unit in the game: minutes and plus-minus."""
    if segments.empty:
        return pd.DataFrame(columns=["team", "seconds", "unit", "minutes", "pts", "opp_pts", "plus_minus"])
    g = segments.groupby("team").agg(seconds=("seconds", "sum"), pts=("pts", "sum"), opp_pts=("opp_pts", "sum"))
    g = g.reset_index()
    g["unit"] = g["team"].map(lambda s: [names.get(int(x), x) for x in s.split("-") if x])
    g["minutes"] = g["seconds"] / 60
    g["plus_minus"] = g["pts"] - g["opp_pts"]
    return g.sort_values("seconds", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ what needed to happen

@dataclass
class Lever:
    factor: str
    title: str
    points: float  # points lost against a normal night (positive number)
    suggestions: list[str]


def _short(name: str) -> str:
    return name.split(" ", 1)[-1]


def levers(factors: pd.DataFrame, lines: pd.DataFrame, season_box: pd.DataFrame, game_defenders: pd.DataFrame,
           season_defense: pd.DataFrame | None, game_units: pd.DataFrame) -> list[Lever]:
    """The parts of the game that were worse than normal, biggest first, each with rule-based suggestions.

    ``season_box``: season totals per player for the team (player_name, minutes, fga, fta, tov, oreb, dreb,
    fgm, fg3m, games). ``season_defense``: the ``whatnow.defenders`` table (pts100 by defender), or None.
    """
    out: list[Lever] = []
    sb = season_box.copy()
    sb["plays"] = sb["fga"] + 0.44 * sb["fta"] + sb["tov"]
    sb["tov_pct"] = sb["tov"] / sb["plays"].where(sb["plays"] > 0)
    sb["reb36"] = 36 * (sb["oreb"] + sb["dreb"]) / sb["minutes"].where(sb["minutes"] > 0)
    sb["efg"] = (sb["fgm"] + 0.5 * sb["fg3m"]) / sb["fga"].where(sb["fga"] > 0)
    sb["play_share"] = sb["plays"] / sb["plays"].sum()
    mins = lines.set_index("player_name")["minutes"]
    fixable = factors[~factors["factor"].isin(NOT_LEVERS) & (factors["points"] <= -LEVER_MIN_POINTS)]
    for r in fixable.sort_values("points").itertuples():
        tips: list[str] = []
        if r.factor == "turnovers":
            heavy = lines[lines["tov"] >= 3]
            tips += [f"{_short(x.player_name)} had {int(x.tov)} turnovers" for x in heavy.itertuples()]
            safe = sb[(sb["play_share"] >= 0.08)].sort_values("tov_pct")
            safe = safe[(safe["player_name"].map(mins).fillna(0) < 30)
                        & ~safe["player_name"].isin(set(heavy["player_name"]))].head(2)
            tips += [f"More ball-handling for {_short(x.player_name)} (turns it over on {x.tov_pct:.0%} of plays "
                     f"this season; played {mins.get(x.player_name, 0):.0f} min)" for x in safe.itertuples()]
        elif r.factor == "rebounding":
            reb = sb[(sb["minutes"] >= 40)].sort_values("reb36", ascending=False)
            reb = reb[reb["player_name"].map(mins).fillna(0) < 20].head(2)
            tips += [f"More minutes for {_short(x.player_name)} ({x.reb36:.1f} rebounds per 36 this season; played "
                     f"{mins.get(x.player_name, 0):.0f} min)" for x in reb.itertuples()]
        elif r.factor == "shooting":
            j = lines.merge(sb[["player_name", "efg"]].rename(columns={"efg": "season_efg"}), on="player_name",
                            how="left")
            cold = j[(j["fga"] >= 6) & (j["efg"] <= j["season_efg"] - 0.15)]
            tips += [f"{_short(x.player_name)} shot {int(x.fgm)}/{int(x.fga)} (season eFG {x.season_efg:.0%})"
                     for x in cold.itertuples()]
        elif r.factor == "opp_shooting" and len(game_defenders):
            worst = game_defenders.iloc[0]
            tip = (f"{_short(worst['def_player'])} gave up {int(worst['pts'])} points, mostly guarding "
                   f"{worst['main_assignment']}")
            if season_defense is not None and len(season_defense):
                stingy = season_defense[season_defense["player_name"] != worst["def_player"]].sort_values(
                    "pts100").head(1)
                if len(stingy):
                    tip += (f"; {_short(stingy['player_name'].iloc[0])} allows the fewest points per possession "
                            "this season")
            tips.append(tip)
        elif r.factor == "free_throws":
            fouls = lines[lines["pf"] >= 5]
            tips += [f"{_short(x.player_name)} had {int(x.pf)} fouls" for x in fouls.itertuples()]
        out.append(Lever(r.factor, r.name, -float(r.points), tips))
    bad = game_units[(game_units["seconds"] >= UNIT_MIN_SECONDS) & (game_units["plus_minus"] <= -UNIT_SWING)]
    for u in bad.itertuples():
        out.append(Lever("unit", "A five-man unit that got outscored", float(-u.plus_minus),
                         [f"{', '.join(_short(n) for n in u.unit)}: {u.plus_minus:+d} in {u.minutes:.0f} min"]))
    return out


def _lower_first(name: str) -> str:
    return name if name.startswith("Clippers") else name[0].lower() + name[1:]


def verdict(factors: pd.DataFrame, lvs: list[Lever]) -> str:
    """One sentence: would a normal night on the worst parts have changed the result?"""
    margin = factors.attrs.get("margin", 0.0)
    if margin > 0:
        good = factors[~factors["factor"].isin(NOT_LEVERS) & (factors["points"] >= LEVER_MIN_POINTS)].sort_values(
            "points", ascending=False)
        if good.empty:
            return f"Won by {margin:.0f} on a fairly normal night."
        return (f"Won by {margin:.0f}. What went better than usual: "
                + "; ".join(f"{_lower_first(n)} ({p:+.1f})" for n, p in zip(good["name"], good["points"])) + ".")
    need = -margin + 1
    acc, used = 0.0, []
    for lv in [x for x in lvs if x.factor != "unit"]:
        acc += lv.points
        used.append(lv.title)
        if acc >= need:
            break
    names = " and ".join(_lower_first(u) for u in used)
    if used and acc >= need:
        return (f"Lost by {-margin:.0f}. A normal night on {names} would have been worth about {acc:.0f} points: "
                "enough to win.")
    if used:
        return (f"Lost by {-margin:.0f}. Even a normal night on {names} (about {acc:.0f} points) "
                "wouldn't have covered it.")
    return f"Lost by {-margin:.0f}, with no single part of the game far below normal."
