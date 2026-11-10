"""The next game: odds of winning from season statistics, and what moves them.

The model is deliberately simple and written down:

1. **Team strength** is net rating (points scored minus allowed per 100 possessions) from the
   league's team stats, regulation only. Early in a season a few games swing it a lot, so it is
   blended with last season's: ``(games * this + 10 * last) / (games + 10)``. After 10 games this
   season counts half; after 30, three quarters.
2. **Expected margin** = (Clippers net - opponent net) x expected possessions / 100, plus
   ``HOME_POINTS`` for the home team (none on a neutral floor). ``HOME_POINTS = 2`` is about half of
   the league's home-vs-road gap of ~3.3 points per 100 measured in the clippers-home-court study.
3. **Win chance** = the normal probability of winning by more than zero, with an NBA game's
   spread around the expected margin of ``MARGIN_SD = 12.5`` points.

It knows nothing about injuries, rest or travel, and isn't betting advice. Simple rating models
like this one pick roughly two in three NBA games correctly.

**Levers** ask: if the Clippers were league average at one thing (or as good after halftime as
before it), how many points a game is that worth, and what would the odds become? Each uses a
plain conversion: a turnover avoided becomes a normal possession; an offensive rebound denied
takes away a possession worth the league's points per possession; second-half shooting comes from
the season's halftime breakdown.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import erf, sqrt

import pandas as pd

PRIOR_GAMES = 10
HOME_POINTS = 2.0
MARGIN_SD = 12.5
SIZE_WINGS = (78, 82)  # 6'6" to 6'10": the bigger defenders Garland struggles against
LONG_ROTATION = 4  # an opponent with 4+ of them in its top 8 is a "long" team


def win_chance(margin: float, sd: float = MARGIN_SD) -> float:
    return 0.5 * (1 + erf(margin / (sd * sqrt(2))))


def team_table(league_periods: pd.DataFrame, season: str) -> pd.DataFrame:
    """Per team, regulation totals for the season: ratings and the four factors for both sides."""
    lp = league_periods[league_periods["season"] == season]
    counts = [c for c in lp.columns if c not in ("season", "period", "team_id", "team_name", "gp", "min")]
    t = lp.groupby(["team_id", "team_name"])[counts].sum()
    t["gp"] = lp.groupby(["team_id", "team_name"])["gp"].max()
    t = t.reset_index()
    t["ortg"] = 100 * t["pts"] / t["poss"]
    t["drtg"] = 100 * t["opp_pts"] / t["poss"]
    t["net"] = t["ortg"] - t["drtg"]
    t["pace"] = t["poss"] / t["gp"]
    t["efg"] = (t["fgm"] + 0.5 * t["fg3m"]) / t["fga"]
    t["opp_efg"] = (t["opp_fgm"] + 0.5 * t["opp_fg3m"]) / t["opp_fga"]
    t["tov_pct"] = t["tov"] / t["poss"]
    t["opp_tov_pct"] = t["opp_tov"] / t["poss"]
    t["oreb_pct"] = t["oreb"] / (t["oreb"] + t["opp_dreb"])
    t["opp_oreb_pct"] = t["opp_oreb"] / (t["opp_oreb"] + t["dreb"])
    t["ft_rate"] = t["fta"] / t["fga"]
    t["opp_ft_rate"] = t["opp_fta"] / t["opp_fga"]
    t["fg3a_rate"] = t["fg3a"] / t["fga"]
    for col, best_high in (("net", True), ("ortg", True), ("drtg", False), ("efg", True), ("opp_efg", False),
                           ("tov_pct", False), ("opp_tov_pct", True), ("oreb_pct", True), ("opp_oreb_pct", False),
                           ("ft_rate", True), ("opp_ft_rate", False), ("fg3a_rate", True)):
        t[f"{col}_rank"] = t[col].rank(ascending=not best_high, method="min").astype(int)
    return t


def blended(this: pd.DataFrame, last: pd.DataFrame | None) -> pd.DataFrame:
    """This season's net rating blended with last season's, weighted by games played."""
    out = this.copy()
    prior = last.set_index("team_id")["net"] if last is not None and len(last) else pd.Series(dtype=float)
    out["prior_net"] = out["team_id"].map(prior).fillna(0.0)
    out["blend_net"] = (out["gp"] * out["net"] + PRIOR_GAMES * out["prior_net"]) / (out["gp"] + PRIOR_GAMES)
    return out


@dataclass
class Lever:
    title: str
    points: float  # points a game
    chance: float  # win chance with this lever
    why: str


@dataclass
class Preview:
    opponent: str
    opponent_name: str
    home: bool
    neutral: bool
    tipoff_utc: pd.Timestamp
    arena: str
    margin: float
    chance: float
    lac: pd.Series
    opp: pd.Series
    levers: list[Lever] = field(default_factory=list)


def preview(game: pd.Series, teams: pd.DataFrame, team_id: int, shooting_per_game: float | None = None) -> Preview:
    """Odds for one scheduled game. ``teams``: ``blended(team_table(...))`` for the current season.

    ``shooting_per_game``: the season's second-half shooting cost in points a game (negative), from
    the halftime breakdown; it becomes the "shoot as well after halftime as before it" lever.
    """
    t = teams.set_index("team_id")
    home = int(game["home_id"]) == team_id
    opp_id = int(game["away_id"] if home else game["home_id"])
    lac, opp = t.loc[team_id], t.loc[opp_id]
    pace = (lac["pace"] + opp["pace"]) / 2
    edge = 0.0 if game["neutral"] else (HOME_POINTS if home else -HOME_POINTS)
    margin = (lac["blend_net"] - opp["blend_net"]) * pace / 100 + edge
    p = Preview(game["away"] if home else game["home"], opp.name if isinstance(opp.name, str) else str(opp["team_name"]),
                home, bool(game["neutral"]), game["tipoff_utc"], game.get("arena", ""), float(margin),
                win_chance(margin), lac, opp)
    p.opponent_name = str(opp["team_name"])
    lg = teams.mean(numeric_only=True)
    ppp = lac["pts"] / (lac["poss"] - lac["tov"])  # points per possession that doesn't end in a turnover
    lg_ppp = lg["pts"] / lg["poss"]
    lv: list[Lever] = []
    if lac["tov_pct"] > lg["tov_pct"]:
        pts = (lac["tov_pct"] - lg["tov_pct"]) * pace * ppp
        why = (f"They turn it over on {lac['tov_pct']:.1%} of possessions ({_ord(lac['tov_pct_rank'])} of 30); "
               f"league average is {lg['tov_pct']:.1%}.")
        if opp["opp_tov_pct_rank"] <= 10:
            why += f" {p.opponent} forces turnovers ({_ord(opp['opp_tov_pct_rank'])} in the league), so this matters more."
        lv.append(Lever("Take care of the ball (league-average turnovers)", pts, win_chance(margin + pts), why))
    if lac["opp_oreb_pct"] > lg["opp_oreb_pct"]:
        misses = (opp["fga"] - opp["fgm"]) / opp["gp"]
        pts = (lac["opp_oreb_pct"] - lg["opp_oreb_pct"]) * misses * lg_ppp
        why = (f"Opponents grab {lac['opp_oreb_pct']:.0%} of their misses against them "
               f"({_ord(lac['opp_oreb_pct_rank'])} of 30).")
        if opp["oreb_pct_rank"] <= 10:
            why += f" {p.opponent} crashes the glass ({_ord(opp['oreb_pct_rank'])} in offensive rebounding)."
        lv.append(Lever("Finish defense with a rebound (league-average)", pts, win_chance(margin + pts), why))
    if shooting_per_game is not None and shooting_per_game < 0:
        pts = -shooting_per_game
        lv.append(Lever("Shoot as well after halftime as before it", pts, win_chance(margin + pts),
                        f"Second-half shooting costs them {pts:.1f} points a game this season."))
    p.levers = sorted(lv, key=lambda x: -x.points)
    return p


def _ord(n: float) -> str:
    n = int(n)
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


# ------------------------------------------------------------------ who to watch

def who_to_watch(player_season: pd.DataFrame, bio: pd.DataFrame, season: str, opp: str,
                 defenders: pd.DataFrame | None, top: int = 3) -> pd.DataFrame:
    """The opponent's top scorers this season, their size, and the Clipper best suited to guard each.

    Suggested defender: among Clippers with 100+ tracked possessions on defense, the one allowing the
    fewest points per 100 possessions guarded whose height is within 3 inches of the scorer's (or the
    tallest available if none is). Anyone who would give up 25+ lb is skipped.
    """
    ps = player_season[(player_season["season"] == season) & (player_season["team_abbreviation"] == opp)]
    ps = ps[ps["gp"] >= max(1, ps["gp"].max() // 3)].sort_values("pts", ascending=False).head(top)
    b = bio.sort_values("season").drop_duplicates("player_id", keep="last").set_index("player_id")
    rows = []
    for r in ps.itertuples():
        ht, wt = b["height_in"].get(r.player_id), b["weight_lb"].get(r.player_id)
        pick = None
        if defenders is not None and len(defenders) and pd.notna(ht):
            d = defenders[defenders["poss"] >= 100]
            if pd.notna(wt):
                d = d[(wt - d["own_wt"]) < 25]
            close = d[(d["own_ht"] - ht).abs() <= 3]
            pool = close if len(close) else d.sort_values("own_ht", ascending=False).head(3)
            if len(pool):
                pick = pool.sort_values("pts100").iloc[0]["player_name"]
        rows.append({"player_name": r.player_name, "pts": r.pts, "reb": r.reb, "ast": r.ast, "min": r.min,
                     "height_in": ht, "weight_lb": wt, "guard_with": pick})
    return pd.DataFrame(rows)


def long_rotation(player_season: pd.DataFrame, bio: pd.DataFrame, season: str, opp: str) -> list[str]:
    """Players 6'6"-6'10" in the opponent's top 8 by minutes: the size that bothers Garland."""
    ps = player_season[(player_season["season"] == season) & (player_season["team_abbreviation"] == opp)]
    ps = ps.assign(total=ps["min"] * ps["gp"]).sort_values("total", ascending=False).head(8)
    ht = bio.sort_values("season").drop_duplicates("player_id", keep="last").set_index("player_id")["height_in"]
    lo, hi = SIZE_WINGS
    return [r.player_name for r in ps.itertuples() if lo <= ht.get(r.player_id, 0) <= hi]


def head_to_head(games: pd.DataFrame, season: str, opp: str) -> pd.DataFrame:
    g = games[(games["season"] == season) & (games["opponent_abbreviation"] == opp)]
    return g[["game_date", "is_home", "result", "team_pts", "opp_pts"]].sort_values("game_date")
