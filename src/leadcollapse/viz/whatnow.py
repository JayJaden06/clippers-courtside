"""'What needs to happen now': lineup, ball-handling and returning-player questions.

Every verdict here comes from a simple, written-down rule applied to the
processed tables, so it changes with the data:

- a player "helps the shooting" if his second-half eFG% is at or above the
  league's second-half average (on 20+ second-half shots), and "helps from
  three" at 36%+ on 20+ second-half threes;
- he is "safe with the ball" at a second-half turnover rate of 12% or lower,
  and "helps on the glass" at 9+ rebounds per 36 second-half minutes;
- a sample under 150 second-half minutes is flagged as small;
- the starting five "isn't the problem" if its second-half net rating is at
  least 5 points per 100 better than everyone else's second-half minutes;
- the focus player "should handle less, for now" if his second-half turnover
  rate is at least 1.5x his own rate from last season.

These describe what the numbers point to. They are not a coaching plan, and
they can't account for health, matchups or anything the box score misses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

LEAGUE_SHOOTING_MIN_FGA = 20
THREE_PCT_GOOD, THREE_MIN_ATT = 0.36, 20
SAFE_TOV_PCT = 0.12
GLASS_REB36 = 9.0
SMALL_SAMPLE_MIN = 150
STARTERS_MARGIN = 5.0
LOAD_RATIO = 1.5

COUNTS = ["minutes", "fgm", "fga", "fg3m", "fg3a", "fta", "oreb", "dreb", "tov", "pts", "plus_minus", "ast"]


def _rates(g: pd.DataFrame) -> pd.DataFrame:
    g = g.copy()
    g["efg"] = (g["fgm"] + 0.5 * g["fg3m"]) / g["fga"].where(g["fga"] > 0)
    g["fg3_pct"] = g["fg3m"] / g["fg3a"].where(g["fg3a"] > 0)
    g["plays"] = g["fga"] + 0.44 * g["fta"] + g["tov"]
    g["tov_pct"] = g["tov"] / g["plays"].where(g["plays"] > 0)
    g["reb36"] = 36 * (g["oreb"] + g["dreb"]) / g["minutes"].where(g["minutes"] > 0)
    return g


def box_halves(player_box: pd.DataFrame, season: str, team: str) -> pd.DataFrame:
    """Season totals by player and half from the team's box scores, with rates."""
    b = player_box[(player_box["season"] == season) & (player_box["team_abbreviation"] == team)
                   & player_box["split"].astype(str).isin(["H1", "H2"])]
    g = b.assign(split=b["split"].astype(str)).groupby(["player_name", "split"])[COUNTS].sum()
    return _rates(g)


# ------------------------------------------------------------------ shooting

@dataclass
class ShootingDrop:
    table: pd.DataFrame  # player_name, efg_h1, efg_h2, fga_h1, fga_h2, change
    n_worse: int
    n_players: int
    team_h1: float
    team_h2: float
    league_h2: float


def shooting_drop(player_box: pd.DataFrame, season: str, team: str, league_h2: float,
                  min_minutes: float = 40) -> ShootingDrop:
    h = box_halves(player_box, season, team)
    w = h.reset_index().pivot(index="player_name", columns="split", values=["efg", "fga", "minutes"])
    w.columns = [f"{a}_{b.lower()}" for a, b in w.columns]
    w = w[(w["minutes_h1"] >= min_minutes) & (w["minutes_h2"] >= min_minutes)].copy()
    w["change"] = w["efg_h2"] - w["efg_h1"]
    team_tot = h.groupby(level="split")[["fgm", "fg3m", "fga"]].sum()
    team = (team_tot["fgm"] + 0.5 * team_tot["fg3m"]) / team_tot["fga"]
    return ShootingDrop(w.sort_values("change").reset_index(), int((w["change"] < 0).sum()), len(w),
                        float(team["H1"]), float(team["H2"]), league_h2)


# ------------------------------------------------------------ returning players

@dataclass
class Candidate:
    name: str
    this_season_minutes: float
    sample: str  # e.g. "2025-26, all teams" or "2026-27 with LAC"
    h2_minutes: float
    efg: float
    fg3_pct: float
    fg3a: float
    fga: float
    tov_pct: float
    reb36: float
    helps: list[str] = field(default_factory=list)
    small_sample: bool = False


def _pick_sample(rows: list[tuple[str, pd.Series]]) -> tuple[str, pd.Series] | None:
    """The most recent sample with enough second-half minutes, else the largest one."""
    if not rows:
        return None
    big = [r for r in rows if r[1]["minutes"] >= SMALL_SAMPLE_MIN]
    return big[0] if big else max(rows, key=lambda r: r[1]["minutes"])


def candidates(history: pd.DataFrame, player_box: pd.DataFrame, season: str, team: str, names: list[str],
               league_h2: float) -> list[Candidate]:
    cur = box_halves(player_box, season, team)
    hist = history[history["split"] == "H2"].copy() if len(history) else history
    out = []
    for name in names:
        rows = []
        mins_now = float(cur.loc[name, "minutes"].sum()) if name in cur.index.get_level_values(0) else 0.0
        if (name, "H2") in cur.index:
            rows.append((f"{season} with {team}", cur.loc[(name, "H2")]))
        if len(hist):
            for s, r in hist[hist["player_name"] == name].sort_values("season", ascending=False).set_index(
                    "season").iterrows():
                rows.append((f"{s}, all teams", _rates(r[COUNTS].astype(float).to_frame().T).iloc[0]))
        picked = _pick_sample(rows)
        if picked is None:
            continue
        label, r = picked
        helps = []
        if r["fga"] >= LEAGUE_SHOOTING_MIN_FGA and r["efg"] >= league_h2:
            helps.append("shooting")
        if r["fg3a"] >= THREE_MIN_ATT and r["fg3_pct"] >= THREE_PCT_GOOD:
            helps.append("threes")
        if r["tov_pct"] <= SAFE_TOV_PCT:
            helps.append("ball security")
        if r["reb36"] >= GLASS_REB36:
            helps.append("rebounding")
        out.append(Candidate(name, mins_now, label, float(r["minutes"]), float(r["efg"]), float(r["fg3_pct"]),
                             float(r["fg3a"]), float(r["fga"]), float(r["tov_pct"]), float(r["reb36"]), helps,
                             float(r["minutes"]) < SMALL_SAMPLE_MIN))
    return out


# ------------------------------------------------------------- ball handling

@dataclass
class BallLoad:
    tov_pct_h2: float
    prior_tov_pct_h2: float
    play_share_h2: float
    tov_share_h2: float
    pts_per_game: float
    net_with: float
    net_without: float
    min_with: float
    min_without: float
    safest: pd.DataFrame  # other players with real second-half volume, lowest turnover rate first
    should_handle_less: bool


def ball_load(player_box: pd.DataFrame, lineups: pd.DataFrame, h3_key: dict, season: str, team: str,
              focus: str, n_games: int) -> BallLoad:
    h = box_halves(player_box, season, team).reset_index()
    h2 = h[h["split"] == "H2"].copy()
    h2["play_share"] = h2["plays"] / h2["plays"].sum()
    others = h2[(h2["player_name"] != focus) & (h2["play_share"] >= 0.08) & (h2["minutes"] >= 40)]
    last = focus.split(" ", 1)[-1]
    reg = lineups[(lineups["season"] == season) & (lineups["split"] == "REG")]
    has = reg["group_name"].str.contains(last, regex=False)
    w, wo = reg[has], reg[~has]
    net = lambda d: 100 * (d["pts"].sum() - d["opp_pts"].sum()) / max(d["poss"].sum(), 1)  # noqa: E731
    return BallLoad(
        h3_key["tov_pct_h2"], h3_key["prior_h2_tov_pct"], h3_key["play_share_h2"], h3_key["tov_share_h2"],
        h3_key["excess_pts"] / max(n_games, 1), net(w), net(wo), float(w["min"].sum()), float(wo["min"].sum()),
        others.sort_values("tov_pct")[["player_name", "tov_pct", "play_share", "minutes"]].reset_index(drop=True),
        h3_key["tov_pct_h2"] >= LOAD_RATIO * h3_key["prior_h2_tov_pct"],
    )


# ------------------------------------------------------------- starting five

@dataclass
class StartingFive:
    names: list[str]
    games_together: int
    h1_min: float
    h1_net: float
    h2_min: float
    h2_net: float
    rest_h2_min: float
    rest_h2_net: float
    not_the_problem: bool


def starting_five(player_box: pd.DataFrame, lineups: pd.DataFrame, season: str, team: str) -> StartingFive | None:
    b = player_box[(player_box["season"] == season) & (player_box["team_abbreviation"] == team)
                   & (player_box["split"].astype(str) == "FULL") & player_box["starter"]]
    groups = b.groupby("game_id")["player_name"].apply(lambda s: tuple(sorted(s)))
    if groups.empty:
        return None
    five = groups.value_counts().index[0]
    l = lineups[lineups["season"] == season]
    is5 = l["group_name"].apply(lambda g: all(n.split(" ", 1)[-1] in g for n in five))

    def net(d: pd.DataFrame) -> tuple[float, float]:
        return float(d["min"].sum()), 100 * (d["pts"].sum() - d["opp_pts"].sum()) / max(d["poss"].sum(), 1)

    h1m, h1n = net(l[is5 & (l["split"] == "H1")])
    h2m, h2n = net(l[is5 & (l["split"] == "H2")])
    rm, rn = net(l[~is5 & (l["split"] == "H2")])
    return StartingFive(list(five), int((groups == five).sum()), h1m, h1n, h2m, h2n, rm, rn,
                        h2m > 0 and h2n >= rn + STARTERS_MARGIN)


# ------------------------------------------------------------- the summary

def _names(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def summary_lines(sf: StartingFive | None, bl: BallLoad | None, cands: list[Candidate], sd: ShootingDrop,
                  focus: str) -> list[str]:
    """Plain-English recommendations (HTML), each one tied to the number behind it."""
    out = []
    if sf is not None and sf.h2_min > 0:
        if sf.not_the_problem:
            level = "even" if abs(sf.h2_net) < 0.5 else f"{sf.h2_net:+.1f} per 100 possessions"
            out.append(f"<b>Keep the starting five.</b> They're {level} in their "
                       f"{sf.h2_min:.0f} second-half minutes together; the other {sf.rest_h2_min:.0f} second-half "
                       f"minutes are {sf.rest_h2_net:+.1f}. The damage happens when the rotation changes.")
        else:
            out.append(f"<b>The starting five is part of the problem.</b> They're {sf.h2_net:+.1f} per 100 after "
                       f"halftime, vs {sf.rest_h2_net:+.1f} for everyone else.")
    if bl is not None:
        last = focus.split(" ", 1)[-1]
        if bl.should_handle_less:
            alt = bl.safest[bl.safest["play_share"] >= 0.10].head(2)
            alt_txt = (f" {_names([f'{r.player_name} ({r.tov_pct:.0%})' for r in alt.itertuples()])} "
                       f"{'is the safest other ball-handler' if len(alt) == 1 else 'are the safest other ball-handlers'} "
                       "with real volume."
                       if len(alt) else "")
            out.append(f"<b>Take some of the ball out of {last}'s hands after halftime, for now.</b> "
                       f"{bl.tov_pct_h2:.0%} of his second-half plays end in a turnover, against {bl.prior_tov_pct_h2:.0%} "
                       f"last season, so this looks like a slump rather than who he is.{alt_txt}")
        else:
            out.append(f"<b>{last}'s workload looks fine.</b> His second-half turnover rate ({bl.tov_pct_h2:.0%}) is "
                       f"close to last season's ({bl.prior_tov_pct_h2:.0%}).")
    shooters = [c for c in cands if "threes" in c.helps and not c.small_sample]
    if shooters:
        bits = [f"{c.name} ({c.fg3_pct:.0%} from three after halftime, {c.sample.split(',')[0]})" for c in shooters]
        out.append(f"<b>The returning shooters are the biggest fix on paper.</b> {_names(bits)}. This team shoots "
                   f"{sd.team_h2:.1%} after halftime against a league average of {sd.league_h2:.1%}.")
    back = [c for c in shooters if c.this_season_minutes > 0]
    if back:
        out.append(f"<b>{_names([c.name for c in back])} {'is' if len(back) == 1 else 'are'} already back:</b> "
                   "the quickest way to test that is more second-half minutes now.")
    glass = [c for c in cands if "rebounding" in c.helps]
    if glass:
        bits = [f"{c.name} ({c.reb36:.1f} rebounds per 36{', small sample' if c.small_sample else ''})" for c in glass]
        out.append(f"<b>For the second-chance problem, try the rebounders:</b> {_names(bits)}.")
    return out
