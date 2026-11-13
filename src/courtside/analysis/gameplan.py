"""How to beat one particular opponent: keys to the game from season statistics, each with its value.

A key appears only when the numbers point to it, either because of a Clippers weakness the opponent
makes worse or because of a weakness of theirs the Clippers can attack. Each key says how many
points a game it's worth, what the win chance becomes (same model as :mod:`courtside.analysis.preview`),
and which Clippers to lean on. The rules:

- **Protect the ball**: the Clippers' expected turnover rate tonight is their own rate, shifted by how
  much more (or less) than average this opponent forces. Getting to league average is the key's
  value; a turnover avoided becomes a normal possession. Lean on the ball-handlers (3+ assists per 36,
  8%+ of the team's plays) with the lowest turnover rate.
- **Finish defense with a rebound**: the opponent's expected offensive-rebound rate tonight is theirs,
  shifted by how much the Clippers give up versus average. Holding them to league average is the
  value; each offensive rebound denied removes a possession. Lean on the best rebounders per 36.
- **Crash the offensive glass**: shows when the opponent is bottom-10 at defensive rebounding; the
  value is the extra offensive rebounds they usually give up, at the Clippers' points per possession.
- **Shoot more threes**: shows when the opponent is bottom-10 at defending the three. Each shot moved
  from two to three is worth 3 x p3 - 2 x p2, where each rate is the average of what the Clippers
  make and what this opponent allows; the key is five such shots. Lean on the best 3-point shooters.
- **Attack the paint**: shows when the opponent gives up more points in the paint than the league
  average. Half of that extra is counted (some paint baskets just replace other baskets). Lean on the
  Clippers who score the most in the paint per 36 minutes.
- **Get to the line**: shows when the opponent fouls more than average (bottom-10 in free throws
  allowed per shot). Half of the extra free throws at the Clippers' FT% is counted.
- **Keep shooting after halftime**: the season-long second-half shooting cost, whenever there is one.

Points from different keys overlap a little, so "the whole plan" (top three added together) is an
optimistic ceiling, not a forecast.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from courtside.analysis.preview import Preview, _ord, win_chance

BOTTOM10 = 21  # rank 21-30 of 30
SWAPPED_THREES = 5
MIN_KEY_POINTS = 0.3


@dataclass
class Key:
    kind: str
    title: str
    why: str
    points: float
    chance: float
    lean_on: list[tuple[str, int, str]] = field(default_factory=list)  # (name, player_id, the stat behind it)


def player_table(player_box: pd.DataFrame, season: str, team: str) -> pd.DataFrame:
    """Season totals and rates per player for one team, from its box scores."""
    b = player_box[(player_box["season"] == season) & (player_box["team_abbreviation"] == team)
                   & (player_box["split"].astype(str) == "FULL") & (player_box["minutes"] > 0)]
    cols = ["minutes", "pts", "fgm", "fga", "fg3m", "fg3a", "ftm", "fta", "oreb", "dreb", "ast", "tov", "pts_paint"]
    t = b.groupby(["player_id", "player_name"])[cols].sum()
    t["games"] = b.groupby(["player_id", "player_name"]).size()
    t = t.reset_index()
    t["plays"] = t["fga"] + 0.44 * t["fta"] + t["tov"]
    t["play_share"] = t["plays"] / t["plays"].sum()
    t["tov_pct"] = t["tov"] / t["plays"].where(t["plays"] > 0)
    per36 = 36 / t["minutes"].where(t["minutes"] > 0)
    t["reb36"] = (t["oreb"] + t["dreb"]) * per36
    t["oreb36"] = t["oreb"] * per36
    t["paint36"] = t["pts_paint"] * per36
    t["fta36"] = t["fta"] * per36
    t["ast36"] = t["ast"] * per36
    t["fg3_pct"] = t["fg3m"] / t["fg3a"].where(t["fg3a"] > 0)
    t["mpg"] = t["minutes"] / t["games"]
    return t


def _top(t: pd.DataFrame, col: str, n: int, fmt, ascending: bool = False) -> list[tuple[str, int, str]]:
    s = t.dropna(subset=[col]).sort_values(col, ascending=ascending).head(n)
    return [(r.player_name, int(r.player_id), fmt(r)) for r in s.itertuples()]


def game_plan(pv: Preview, teams: pd.DataFrame, misc: pd.DataFrame, players: pd.DataFrame,
              shooting_per_game: float | None = None) -> list[Key]:
    """Keys to beating ``pv``'s opponent, biggest first. ``misc``: league team Misc stats for the season."""
    lac, opp = pv.lac, pv.opp
    lg = teams.mean(numeric_only=True)
    pace = (lac["pace"] + opp["pace"]) / 2
    ppp = lac["pts"] / (lac["poss"] - lac["tov"])
    lg_ppp = lg["pts"] / lg["poss"]
    handlers = players[(players["play_share"] >= 0.08) & (players["minutes"] >= 60) & (players["ast36"] >= 3)]
    rotation = players[players["minutes"] >= 60]
    keys: list[Key] = []

    def add(kind: str, title: str, why: str, pts: float, lean: list[tuple[str, int, str]]) -> None:
        if pts >= MIN_KEY_POINTS:
            keys.append(Key(kind, title, why, float(pts), win_chance(pv.margin + pts), lean))

    # protect the ball
    expected = lac["tov_pct"] + (opp["opp_tov_pct"] - lg["opp_tov_pct"])
    add("turnovers", "Protect the ball",
        f"The Clippers turn it over on {lac['tov_pct']:.1%} of possessions ({_ord(lac['tov_pct_rank'])} of 30); "
        f"{pv.opponent} forces turnovers on {opp['opp_tov_pct']:.1%} ({_ord(opp['opp_tov_pct_rank'])}). "
        f"Expect about {expected:.1%} tonight; league average is {lg['tov_pct']:.1%}.",
        (expected - lg["tov_pct"]) * pace * ppp,
        _top(handlers, "tov_pct", 2, lambda r: f"turns it over on {r.tov_pct:.0%} of his plays, "
             f"{r.ast36:.1f} assists per 36", ascending=True))

    # finish defense with a rebound
    exp_oreb = opp["oreb_pct"] + (lac["opp_oreb_pct"] - lg["opp_oreb_pct"])
    opp_misses = (opp["fga"] - opp["fgm"]) / opp["gp"]
    add("defensive_glass", "Finish defense with a rebound",
        f"{pv.opponent} gets {opp['oreb_pct']:.0%} of its misses back ({_ord(opp['oreb_pct_rank'])} of 30) and the "
        f"Clippers allow {lac['opp_oreb_pct']:.0%} ({_ord(lac['opp_oreb_pct_rank'])}). Expect about {exp_oreb:.0%} "
        f"tonight; league average is {lg['oreb_pct']:.0%}.",
        (exp_oreb - lg["oreb_pct"]) * opp_misses * lg_ppp,
        _top(rotation, "reb36", 2, lambda r: f"{r.reb36:.1f} rebounds per 36 minutes"))

    # crash the offensive glass
    if opp["opp_oreb_pct_rank"] >= BOTTOM10:
        lac_misses = (lac["fga"] - lac["fgm"]) / lac["gp"]
        add("offensive_glass", "Crash the offensive glass",
            f"{pv.opponent} lets opponents grab {opp['opp_oreb_pct']:.0%} of their misses "
            f"({_ord(opp['opp_oreb_pct_rank'])} of 30 at defensive rebounding; league average "
            f"{lg['opp_oreb_pct']:.0%}).",
            (opp["opp_oreb_pct"] - lg["opp_oreb_pct"]) * lac_misses * ppp,
            _top(rotation, "oreb36", 2, lambda r: f"{r.oreb36:.1f} offensive rebounds per 36"))

    # shoot more threes
    t = teams.assign(opp_fg3_pct=teams["opp_fg3m"] / teams["opp_fg3a"],
                     opp_fg2_pct=(teams["opp_fgm"] - teams["opp_fg3m"]) / (teams["opp_fga"] - teams["opp_fg3a"]))
    t["opp_fg3_rank"] = t["opp_fg3_pct"].rank(method="min").astype(int)  # 1 = stingiest
    o3 = t.set_index("team_id").loc[int(opp["team_id"]) if "team_id" in opp else opp.name]
    if o3["opp_fg3_rank"] >= BOTTOM10:
        lac3 = lac["fg3m"] / lac["fg3a"]
        lac2 = (lac["fgm"] - lac["fg3m"]) / (lac["fga"] - lac["fg3a"])
        p3, p2 = (lac3 + o3["opp_fg3_pct"]) / 2, (lac2 + o3["opp_fg2_pct"]) / 2
        add("threes", f"Take {SWAPPED_THREES} more threes",
            f"{pv.opponent} allows {o3['opp_fg3_pct']:.1%} from three ({_ord(o3['opp_fg3_rank'])} of 30). A three "
            f"tonight is worth about {3 * p3:.2f} points against {2 * p2:.2f} for a two.",
            SWAPPED_THREES * (3 * p3 - 2 * p2),
            _top(rotation[rotation["fg3a"] >= 15], "fg3_pct", 2, lambda r: f"{r.fg3_pct:.0%} from three"))

    # attack the paint
    m = misc.set_index("team_id")
    oid = int(opp["team_id"]) if "team_id" in opp else int(opp.name)
    if oid in m.index:
        allowed, lg_allowed = m.loc[oid, "opp_pts_paint"], m["opp_pts_paint"].mean()
        rank = int(m["opp_pts_paint"].rank(method="min").loc[oid])  # 1 = gives up the fewest
        add("paint", "Attack the paint",
            f"{pv.opponent} gives up {allowed:.0f} points in the paint a game ({_ord(rank)} of 30; league average "
            f"{lg_allowed:.0f}). Half of the extra is counted.",
            0.5 * (allowed - lg_allowed),
            _top(rotation, "paint36", 2, lambda r: f"{r.paint36:.1f} paint points per 36"))

    # get to the line
    if opp["opp_ft_rate_rank"] >= BOTTOM10:
        extra_fta = (opp["opp_ft_rate"] - lg["opp_ft_rate"]) * lac["fga"] / lac["gp"]
        ft_pct = lac["ftm"] / lac["fta"]
        add("free_throws", "Get to the line",
            f"{pv.opponent} sends opponents to the line {opp['opp_ft_rate']:.2f} times per shot "
            f"({_ord(opp['opp_ft_rate_rank'])} of 30). Half of the extra free throws is counted.",
            0.5 * extra_fta * ft_pct,
            _top(rotation, "fta36", 2, lambda r: f"{r.fta36:.1f} free throws per 36"))

    # keep shooting after halftime
    if shooting_per_game is not None and shooting_per_game < 0:
        add("second_half", "Keep shooting after halftime",
            f"Second-half shooting costs the Clippers {-shooting_per_game:.1f} points a game this season.",
            -shooting_per_game,
            _top(rotation[rotation["fg3a"] >= 15], "fg3_pct", 2, lambda r: f"{r.fg3_pct:.0%} from three"))
    return sorted(keys, key=lambda k: -k.points)


