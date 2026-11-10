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

Minutes, defense, matchups and chemistry (tracking data is full-game only):

- a player has "earned a longer look" with a second-half on/off of +10 per 100
  or better on 40+ second-half minutes while averaging under 20 minutes; he is
  "struggling after halftime" at a second-half on-court net of -15 or worse on
  60+ minutes;
- a defender "gets scored on" if opponents shoot 5+ percentage points better
  against him than against the team as a whole (on 30+ shots), and "gives up
  size" if the players he guards average 25+ lb more than he weighs;
- the second-half glass "got worse" if opponents rebound 3+ points more of
  their misses after halftime than they did last season;
- "bigger defenders bother him" if the focus player turns it over at least 1.5x
  as often per possession against defenders 6'6" and taller as against shorter
  ones (on 8+ such turnovers); "teams are trapping him" if a second defender
  (not the one guarding him) gets 10+ points more of his turnovers as steals
  than for his teammates. Public data does not log double teams, so the second
  rule is a proxy: it counts steals by any other defender.

These describe what the numbers point to. They are not a coaching plan, and
they can't account for health, chemistry in the locker room, or anything the
cameras don't track.
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
EARNED_ON_OFF, EARNED_MIN, EARNED_MPG = 10.0, 40, 20
STRUGGLE_NET, STRUGGLE_MIN = -15.0, 60
ATTACKED_FG, ATTACKED_MIN_FGA = 0.05, 30
HEAVIER_BY = 25
GLASS_WORSE = 0.03
LONG_IN = 78  # 6'6"
LENGTH_RATIO, LENGTH_MIN_TOV = 1.5, 8
TRAP_MARGIN = 0.10

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
    team_fg3_h2: float = float("nan")


def shooting_drop(player_box: pd.DataFrame, season: str, team: str, league_h2: float,
                  min_minutes: float = 40) -> ShootingDrop:
    h = box_halves(player_box, season, team)
    w = h.reset_index().pivot(index="player_name", columns="split", values=["efg", "fga", "minutes"])
    w.columns = [f"{a}_{b.lower()}" for a, b in w.columns]
    w = w[(w["minutes_h1"] >= min_minutes) & (w["minutes_h2"] >= min_minutes)].copy()
    w["change"] = w["efg_h2"] - w["efg_h1"]
    team_tot = h.groupby(level="split")[["fgm", "fg3m", "fga", "fg3a"]].sum()
    team = (team_tot["fgm"] + 0.5 * team_tot["fg3m"]) / team_tot["fga"]
    return ShootingDrop(w.sort_values("change").reset_index(), int((w["change"] < 0).sum()), len(w),
                        float(team["H1"]), float(team["H2"]), league_h2,
                        float(team_tot.loc["H2", "fg3m"] / team_tot.loc["H2", "fg3a"]))


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
    mpg: float = float("nan")  # minutes per game in that sample (whole game)
    games: int = 0


def _pick_sample(rows: list[tuple[str, pd.Series]]) -> tuple[str, pd.Series] | None:
    """The most recent sample with enough second-half minutes, else the largest one."""
    if not rows:
        return None
    big = [r for r in rows if r[1]["minutes"] >= SMALL_SAMPLE_MIN]
    return big[0] if big else max(rows, key=lambda r: r[1]["minutes"])


def candidates(history: pd.DataFrame, player_box: pd.DataFrame, season: str, team: str, names: list[str],
               league_h2: float) -> list[Candidate]:
    cur = box_halves(player_box, season, team)
    full = player_box[(player_box["season"] == season) & (player_box["team_abbreviation"] == team)
                      & (player_box["split"].astype(str) == "FULL") & (player_box["minutes"] > 0)]
    hist = history[history["split"] == "H2"].copy() if len(history) else history
    out = []
    for name in names:
        rows = []
        mins_now = float(cur.loc[name, "minutes"].sum()) if name in cur.index.get_level_values(0) else 0.0
        mine = full[full["player_name"] == name]
        if (name, "H2") in cur.index:
            rows.append((f"{season} with {team}", cur.loc[(name, "H2")],
                         (float(mine["minutes"].sum()), len(mine))))
        if len(hist):
            for s, r in hist[hist["player_name"] == name].sort_values("season", ascending=False).set_index(
                    "season").iterrows():
                season_rows = history[(history["player_name"] == name) & (history["season"] == s)]
                gp = int(season_rows["games"].max()) if "games" in season_rows else 0
                rows.append((f"{s}, all teams", _rates(r[COUNTS].astype(float).to_frame().T).iloc[0],
                             (float(season_rows["minutes"].sum()), gp)))
        picked = _pick_sample([(lab, r) for lab, r, _ in rows])
        if picked is None:
            continue
        label, r = picked
        tot_min, gp = next(x for lab, _, x in rows if lab == label)
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
                             float(r["minutes"]) < SMALL_SAMPLE_MIN, tot_min / gp if gp else float("nan"), gp))
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
    h2_pm: float = 0.0  # raw second-half plus-minus, season total
    rest_h2_pm: float = 0.0


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
    pm = lambda d: float(d["pts"].sum() - d["opp_pts"].sum())  # noqa: E731
    return StartingFive(list(five), int((groups == five).sum()), h1m, h1n, h2m, h2n, rm, rn,
                        h2m > 0 and h2n >= rn + STARTERS_MARGIN,
                        pm(l[is5 & (l["split"] == "H2")]), pm(l[~is5 & (l["split"] == "H2")]))


# ------------------------------------------------------------- minutes

def _team_box(player_box: pd.DataFrame, season: str, team: str) -> pd.DataFrame:
    b = player_box[(player_box["season"] == season) & (player_box["team_abbreviation"] == team)]
    return b.assign(split=b["split"].astype(str))


def _bio_by_id(bio: pd.DataFrame, season: str | None = None) -> pd.DataFrame:
    """Latest height/weight per player; ``rookie`` taken from ``season`` when given."""
    latest = bio.sort_values("season").drop_duplicates("player_id", keep="last").set_index("player_id")
    if season is not None:
        this = bio[bio["season"] == season].drop_duplicates("player_id").set_index("player_id")["rookie"]
        latest["rookie"] = this.reindex(latest.index).fillna(False).astype(bool)
    return latest


def rotation(player_box: pd.DataFrame, onoff: pd.DataFrame, bio: pd.DataFrame, season: str, team: str,
             min_minutes: float = 30) -> pd.DataFrame:
    """Who plays how much, when, and how the score moves with them on the floor after halftime."""
    b = _team_box(player_box, season, team)
    played = b[(b["split"] == "FULL") & (b["minutes"] > 0)]
    mins = b.pivot_table(index="player_id", columns="split", values="minutes", aggfunc="sum").fillna(0)
    gp = played.groupby("player_id").size()
    r = pd.DataFrame({"gp": gp})
    r["player_name"] = played.drop_duplicates("player_id").set_index("player_id")["player_name"]
    r["minutes"] = mins["FULL"].reindex(r.index)
    for col, split in (("mpg", "FULL"), ("h2_mpg", "H2"), ("q4_mpg", "Q4")):
        r[col] = (mins[split].reindex(r.index) if split in mins else 0) / r["gp"]
    h2 = b[b["split"] == "H2"].groupby("player_id")[["oreb", "dreb", "minutes"]].sum().reindex(r.index)
    r["h2_reb36"] = 36 * (h2["oreb"] + h2["dreb"]) / h2["minutes"].where(h2["minutes"] > 0)
    o = onoff[(onoff["season"] == season) & (onoff["split"] == "H2")].set_index("player_id")
    r["h2_min_on"] = o["min_on"].reindex(r.index).fillna(0)
    r["h2_net_on"] = o["net_rtg_on"].reindex(r.index)
    r["h2_on_off"] = o["net_on_off"].reindex(r.index)
    # Plus-minus with him on the floor after halftime (from the on-court rating and possessions).
    r["h2_pm"] = (o["net_rtg_on"] * o["poss_on"] / 100).reindex(r.index)
    r["h2_pm_pg"] = r["h2_pm"] / r["gp"]
    r["rookie"] = _bio_by_id(bio, season)["rookie"].reindex(r.index).fillna(False).astype(bool) if len(bio) else False
    r["earned"] = ((r["h2_on_off"] >= EARNED_ON_OFF) & (r["h2_min_on"] >= EARNED_MIN) & (r["mpg"] < EARNED_MPG))
    r["struggling"] = (r["h2_net_on"] <= STRUGGLE_NET) & (r["h2_min_on"] >= STRUGGLE_MIN)
    r["q4_rank"] = r["q4_mpg"].rank(ascending=False, method="first").astype(int)
    r = r[r["minutes"] >= min_minutes]
    return r.sort_values("mpg", ascending=False).reset_index()


# ------------------------------------------------------------- defense

@dataclass
class Defense:
    table: pd.DataFrame  # one row per defender
    team_pts100: float
    team_fg: float
    games: int


def defenders(matchups: pd.DataFrame, hustle: pd.DataFrame, bio: pd.DataFrame, season: str, team: str,
              min_poss: float = 100) -> Defense | None:
    """Who each defender guarded, how big they were, and how much they scored."""
    d = matchups[(matchups["season"] == season) & (matchups["def_team"] == team)].copy()
    if d.empty:
        return None
    b = _bio_by_id(bio, season)
    d["off_wt"] = d["off_player_id"].map(b["weight_lb"])
    d["off_ht"] = d["off_player_id"].map(b["height_in"])

    def one(z: pd.DataFrame) -> pd.Series:
        known = z[z["off_wt"].notna()]
        w = known["partial_poss"].sum()
        return pd.Series({
            "poss": z["partial_poss"].sum(), "pts": z["pts"].sum(), "fgm": z["fgm"].sum(), "fga": z["fga"].sum(),
            "shooting_fouls": z["shooting_fouls"].sum(),
            "opp_wt": (known["off_wt"] * known["partial_poss"]).sum() / w if w else float("nan"),
            "opp_ht": (known["off_ht"] * known["partial_poss"]).sum() / w if w else float("nan"),
        })

    t = d.groupby(["def_player_id", "def_player"]).apply(one, include_groups=False).reset_index()
    t = t.rename(columns={"def_player_id": "player_id", "def_player": "player_name"})
    t["pts100"] = 100 * t["pts"] / t["poss"]
    t["fg_pct"] = t["fgm"] / t["fga"].where(t["fga"] > 0)
    t["own_wt"] = t["player_id"].map(b["weight_lb"])
    t["own_ht"] = t["player_id"].map(b["height_in"])
    t["rookie"] = t["player_id"].map(b["rookie"]).fillna(False).astype(bool)
    hu = hustle[(hustle["season"] == season) & (hustle["team_abbreviation"] == team)]
    hu = hu.groupby("player_id")[["minutes", "contested_shots", "deflections", "def_box_outs"]].sum()
    for col in ("contested_shots", "deflections", "def_box_outs"):
        t[f"{col}_36"] = t["player_id"].map(36 * hu[col] / hu["minutes"].where(hu["minutes"] > 0))
    t["hustle_min"] = t["player_id"].map(hu["minutes"])
    t["def_box_outs"] = t["player_id"].map(hu["def_box_outs"])
    team_pts100 = 100 * d["pts"].sum() / d["partial_poss"].sum()
    team_fg = d["fgm"].sum() / d["fga"].sum()
    t["attacked"] = (t["fg_pct"] >= team_fg + ATTACKED_FG) & (t["fga"] >= ATTACKED_MIN_FGA)
    t["wt_gap"] = t["opp_wt"] - t["own_wt"]
    t["gives_up_size"] = t["wt_gap"] >= HEAVIER_BY
    t = t[t["poss"] >= min_poss].sort_values("poss", ascending=False).reset_index(drop=True)
    return Defense(t, float(team_pts100), float(team_fg), int(d["game_id"].nunique()))


@dataclass
class Glass:
    opp_oreb_h1: float
    opp_oreb_h2: float
    prior_opp_oreb_h2: float
    opp_2nd_h1: float  # opponent second-chance points per game
    opp_2nd_h2: float
    prior_opp_2nd_h2: float
    worse: bool


def glass(game_halves: pd.DataFrame, season: str, prior: str) -> Glass:
    g = game_halves.assign(split=game_halves["split"].astype(str))

    def half(s: str, split: str) -> tuple[float, float]:
        x = g[(g["season"] == s) & (g["split"] == split)]
        return (x["opp_oreb"].sum() / (x["opp_oreb"].sum() + x["dreb"].sum()),
                x["opp_pts_2nd_chance"].sum() / max(x["game_id"].nunique(), 1))

    (o1, p1), (o2, p2), (po2, pp2) = half(season, "H1"), half(season, "H2"), half(prior, "H2")
    return Glass(o1, o2, po2, p1, p2, pp2, o2 >= po2 + GLASS_WORSE)


# ------------------------------------------------------------- pressure on the ball-handler

def help_steals(tov_events: pd.DataFrame, matchups: pd.DataFrame, season: str, team: str,
                team_id: int) -> pd.DataFrame:
    """Per player: turnovers, steals, and steals by a defender other than the one(s) credited with guarding him.

    The matchup data credits each turnover to the defender guarding the player
    at the time; a steal by anyone else (a trap, a double team, a help defender
    jumping a pass) is counted as a ``help`` steal. Per game, a defender's steals
    count as "his own man" up to the turnovers he's credited with.
    """
    mm = matchups[(matchups["season"] == season) & (matchups["off_team"] == team)]
    t = tov_events[(tov_events["season"] == season) & (tov_events["team_id"] == team_id)
                   & tov_events["person_id"].notna() & tov_events["game_id"].isin(set(mm["game_id"]))]
    credited = mm.groupby(["game_id", "off_player_id", "def_player_id"])["tov"].sum()
    rows = []
    for (gid, pid), grp in t.groupby(["game_id", "person_id"]):
        steals = grp.loc[grp["live_ball"], "stealer_id"].dropna().astype(int).value_counts()
        own = sum(min(n, credited.get((gid, int(pid), d), 0)) for d, n in steals.items())
        rows.append({"player_id": int(pid), "tov": len(grp), "steals": int(steals.sum()), "own": int(own)})
    out = pd.DataFrame(rows, columns=["player_id", "tov", "steals", "own"]).groupby("player_id").sum()
    out["help"] = out["steals"] - out["own"]
    return out


@dataclass
class Pressure:
    games: int
    tov100_long: float
    tov100_short: float
    tov_long: int
    tov_short: int
    long_poss_share: float
    prior_tov100_long: float
    prior_tov100_short: float
    mates_tov100_long: float
    mates_tov100_short: float
    tov: int
    steals: int
    help: int
    prior_help_share: float
    mates_help_share: float
    h2_types: dict[str, int]
    forcers: pd.DataFrame  # defenders credited with his turnovers
    length_problem: bool
    trapped: bool

    @property
    def help_share(self) -> float:
        return self.help / self.tov if self.tov else float("nan")

    @property
    def size_ratio(self) -> float:
        """Turnovers per 100 against bigger defenders, as a multiple of the rate against shorter ones."""
        return self.tov100_long / self.tov100_short if self.tov100_short else float("nan")

    @property
    def prior_size_ratio(self) -> float:
        return self.prior_tov100_long / self.prior_tov100_short if self.prior_tov100_short else float("nan")

    @property
    def size_gap_last_season(self) -> bool:
        """Was defender height already a problem last season (by this season's rule)?"""
        return self.prior_size_ratio >= LENGTH_RATIO


def pressure(matchups: pd.DataFrame, tov_events: pd.DataFrame, bio: pd.DataFrame, season: str, prior: str,
             team: str, team_id: int, focus_id: int) -> Pressure | None:
    m = matchups[matchups["off_team"] == team].copy()
    if m[m["season"] == season].empty:
        return None
    b = _bio_by_id(bio)
    m["long"] = m["def_player_id"].map(b["height_in"]) >= LONG_IN
    me = m["off_player_id"] == focus_id

    def rate(z: pd.DataFrame, long: bool) -> tuple[float, int, float]:
        z = z[z["long"] == long]
        return 100 * z["tov"].sum() / max(z["partial_poss"].sum(), 1e-9), int(z["tov"].sum()), z["partial_poss"].sum()

    now, before = m["season"] == season, m["season"] == prior
    (l_now, tl, pl), (s_now, ts, ps) = rate(m[now & me], True), rate(m[now & me], False)
    pl_rate, ps_rate = rate(m[before & me], True)[0], rate(m[before & me], False)[0]
    ml, msr = rate(m[now & ~me], True)[0], rate(m[now & ~me], False)[0]

    hs = help_steals(tov_events, matchups, season, team, team_id)
    hp = help_steals(tov_events, matchups, prior, team, team_id)
    mine = hs.loc[focus_id] if focus_id in hs.index else pd.Series({"tov": 0, "steals": 0, "help": 0})
    mates = hs.drop(index=focus_id, errors="ignore")
    prior_share = (hp.loc[focus_id, "help"] / hp.loc[focus_id, "tov"]) if focus_id in hp.index else float("nan")
    mates_share = mates["help"].sum() / max(mates["tov"].sum(), 1)

    t = tov_events[(tov_events["season"] == season) & (tov_events["person_id"] == focus_id)
                   & tov_events["period"].isin([3, 4])]
    forcers = (m[now & me].groupby(["def_player", "def_team", "def_player_id"])[["tov", "partial_poss"]].sum()
               .reset_index())
    forcers["height_in"] = forcers["def_player_id"].map(b["height_in"])
    forcers = forcers[forcers["tov"] > 0].sort_values(["tov", "partial_poss"], ascending=[False, True])
    share = int(mine["help"]) / int(mine["tov"]) if int(mine["tov"]) else 0.0
    return Pressure(
        int(m.loc[now & me, "game_id"].nunique()), l_now, s_now, tl, ts, pl / max(pl + ps, 1e-9),
        pl_rate, ps_rate, ml, msr, int(mine["tov"]), int(mine["steals"]), int(mine["help"]),
        float(prior_share), float(mates_share), t["category"].value_counts().to_dict(),
        forcers.reset_index(drop=True),
        l_now >= LENGTH_RATIO * s_now and tl >= LENGTH_MIN_TOV,
        share >= mates_share + TRAP_MARGIN,
    )


# ------------------------------------------------------------- chemistry

@dataclass
class Chemistry:
    continuity: float  # share of this season's minutes played by players who were on the team last season
    returning: list[str]
    new: list[str]
    pairs: pd.DataFrame  # a, b, min, net, prior_min, new_pair (second halves this season)


def chemistry(player_box: pd.DataFrame, lineups: pd.DataFrame, season: str, prior: str, team: str,
              min_pair_minutes: float = 25) -> Chemistry:
    """How much this group has played together, and how its two-man combinations do after halftime."""
    import itertools

    now = _team_box(player_box, season, team)
    now = now[(now["split"] == "FULL") & (now["minutes"] > 0)]
    before_ids = set(_team_box(player_box, prior, team).query("minutes > 0")["player_id"])
    mins = now.groupby(["player_id", "player_name"])["minutes"].sum().reset_index()
    mins["was_here"] = mins["player_id"].isin(before_ids)
    names = player_box.drop_duplicates("player_id").set_index("player_id")["player_name"]

    def pair_minutes(df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for gid, mn, pts, opp, poss in df[["group_id", "min", "pts", "opp_pts", "poss"]].itertuples(index=False):
            ids = sorted(int(x) for x in gid.strip("-").split("-") if x)
            rows += [(a, c, mn, pts, opp, poss) for a, c in itertools.combinations(ids, 2)]
        return pd.DataFrame(rows, columns=["a_id", "b_id", "min", "pts", "opp_pts", "poss"]).groupby(
            ["a_id", "b_id"]).sum()

    cur = pair_minutes(lineups[(lineups["season"] == season) & (lineups["split"] == "H2")])
    old = pair_minutes(lineups[(lineups["season"] == prior) & (lineups["split"] == "REG")])
    cur["net"] = 100 * (cur["pts"] - cur["opp_pts"]) / cur["poss"].where(cur["poss"] > 0)
    cur["pm"] = cur["pts"] - cur["opp_pts"]
    cur["prior_min"] = old["min"].reindex(cur.index).fillna(0)
    cur = cur[cur["min"] >= min_pair_minutes].reset_index()
    cur["a"] = cur["a_id"].map(names)
    cur["b"] = cur["b_id"].map(names)
    cur["new_pair"] = cur["prior_min"] == 0
    return Chemistry(
        float(mins.loc[mins["was_here"], "minutes"].sum() / mins["minutes"].sum()),
        list(mins.loc[mins["was_here"]].sort_values("minutes", ascending=False)["player_name"]),
        list(mins.loc[~mins["was_here"]].sort_values("minutes", ascending=False)["player_name"]),
        cur.sort_values("net").reset_index(drop=True),
    )


# ------------------------------------------------------------- the summary

def _names(xs: list[str]) -> str:
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def _ft_in(inches: float) -> str:
    return f"{int(inches) // 12}'{int(inches) % 12}\""


def summary_lines(sf: StartingFive | None, bl: BallLoad | None, cands: list[Candidate], sd: ShootingDrop,
                  focus: str, rot: pd.DataFrame | None = None, defense: Defense | None = None,
                  gl: Glass | None = None, pr: Pressure | None = None) -> list[str]:
    """Plain-English recommendations (HTML), each one tied to the number behind it."""
    out = []
    last = focus.split(" ", 1)[-1]
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
    if pr is not None and (pr.length_problem or pr.trapped):
        how = []
        if pr.length_problem:
            how.append(f"when a bigger defender ({_ft_in(LONG_IN)} or taller) guards him, he turns it over "
                       f"{pr.size_ratio:.1f}x as often as against shorter ones (last season "
                       f"{pr.prior_tov100_long:.1f} vs {pr.prior_tov100_short:.1f} per 100, "
                       f"{'a gap then too' if pr.size_gap_last_season else 'no gap'})")
        if pr.trapped:
            how.append(f"{pr.help_share:.0%} of his turnovers are steals by a second defender, more than his "
                       f"teammates ({pr.mates_help_share:.0%}), which fits traps and double teams")
        else:
            how.append(f"only {pr.help_share:.0%} are steals by a second defender (teammates "
                       f"{pr.mates_help_share:.0%}), so it's mostly the man guarding him, not traps")
        lever = ("Getting him the ball on the move or off a screen, away from a bigger defender, is the lever."
                 if pr.length_problem else "Moving the ball before the second defender arrives is the lever.")
        out.append(f"<b>What's causing {last}'s turnovers:</b> {'; '.join(how)}. {lever}")
    if rot is not None and len(rot):
        bad = rot[rot["struggling"]]
        good = rot[rot["earned"]]
        if len(bad):
            out.append(f"<b>Second-half minutes to rethink:</b> {_names([f'{r.player_name} ({r.h2_net_on:+.0f} per 100 in {r.h2_min_on:.0f} min)' for r in bad.itertuples()])}"
                       " on the floor after halftime.")
        if len(good):
            bits = [f"{r.player_name} ({'rookie, ' if r.rookie else ''}{r.mpg:.0f} min a game; the team is "
                    f"{r.h2_on_off:.0f} points per 100 better with him on after halftime)" for r in good.itertuples()]
            out.append(f"<b>Earned a longer look:</b> {_names(bits)}. Small samples, so a longer look, not a promotion.")
    if defense is not None and rot is not None and len(defense.table):
        closers = set(rot.loc[rot["q4_rank"] <= 3, "player_id"])
        for r in defense.table[defense.table["gives_up_size"]].itertuples():
            role = " and plays the most fourth-quarter minutes" if r.player_id in closers else ""
            box = f", {r.def_box_outs:.0f} defensive box-outs in {r.hustle_min:.0f} minutes" if pd.notna(
                r.def_box_outs) else ""
            out.append(f"<b>{r.player_name} gives up size{' as a rookie' if r.rookie else ''}{role}.</b> At "
                       f"{r.own_wt:.0f} lb he guards players who average {r.opp_wt:.0f} lb; opponents shoot "
                       f"{r.fg_pct:.0%} against him (team {defense.team_fg:.0%}){box}. Pair him with a strong "
                       "rebounder and keep him off the bigger scorers late.")
    if gl is not None and gl.worse:
        out.append(f"<b>Finish defensive possessions:</b> opponents rebound {gl.opp_oreb_h2:.0%} of their misses after "
                   f"halftime (before it {gl.opp_oreb_h1:.0%}, last season {gl.prior_opp_oreb_h2:.0%}), worth "
                   f"{gl.opp_2nd_h2:.1f} second-chance points a game.")
    shooters = [c for c in cands if "threes" in c.helps and not c.small_sample]
    if shooters:
        bits = [f"{c.name} ({c.fg3_pct:.0%} from three after halftime, {c.sample.split(',')[0]}"
                + (f", {c.mpg:.0f} min a game" if c.mpg == c.mpg else "") + ")" for c in shooters]
        out.append(f"<b>The returning shooters are the biggest fix on paper.</b> {_names(bits)}. This team shoots "
                   f"{sd.team_h2:.1%} after halftime against a league average of {sd.league_h2:.1%}."
                   + (" Expect anyone coming back from injury to be eased in on a minutes limit at first."
                      if any(c.this_season_minutes == 0 for c in shooters) else ""))
    now = rot.set_index("player_name") if rot is not None and len(rot) else None
    back = [c for c in shooters if c.this_season_minutes > 0]
    if back:
        detail = ""
        if now is not None and all(c.name in now.index for c in back):
            detail = " (now " + ", ".join(f"{now.loc[c.name, 'h2_mpg']:.0f} second-half minutes a game in "
                                          f"{now.loc[c.name, 'gp']:.0f} games" for c in back) + ")"
        out.append(f"<b>{_names([c.name for c in back])} {'is' if len(back) == 1 else 'are'} already back{detail}:</b> "
                   "the quickest way to test that is more second-half minutes now.")
    glass = [c for c in cands if "rebounding" in c.helps]
    if glass:
        def tag(c: Candidate) -> str:
            bits = [f"{c.reb36:.1f} rebounds per 36"]
            if now is not None and c.name in now.index:
                if bool(now.loc[c.name, "rookie"]):
                    bits.append("rookie")
                bits.append(f"{now.loc[c.name, 'mpg']:.0f} min a game now")
            if c.small_sample:
                bits.append("small sample")
            return f"{c.name} ({', '.join(bits)})"
        out.append(f"<b>For the second-chance problem, try the rebounders:</b> {_names([tag(c) for c in glass])}.")
    return out


# ------------------------------------------------------------- the answer cards

@dataclass
class Verdict:
    """One answer tile: a short headline, one big number, a small picture of it, one line of context.

    ``viz`` (optional) describes the picture; see ``courtside.viz.graphics.mini``:
    ``("dots", k, n)``, ``("pair", before, after, label_before, label_after)``, ``("share", p, mark)``,
    ``("rank", r, n)`` or ``("diverge", (value, ...), (label, ...))``.
    """
    title: str
    big: str
    small: str
    tone: str  # "good", "bad", "warn" or "info"
    anchor: str  # section to jump to
    viz: tuple | None = None
    image: str | None = None  # a photo or logo shown on the tile


def _last(name: str) -> str:
    return name.split(" ", 1)[-1]


def verdicts(sf: StartingFive | None, bl: BallLoad | None, cands: list[Candidate], sd: ShootingDrop, focus: str,
             rot: pd.DataFrame | None = None, defense: Defense | None = None, gl: Glass | None = None,
             pr: Pressure | None = None, n_games: int = 1) -> list[Verdict]:
    """The same rules as ``summary_lines``, as cards: few words, one number each."""
    out: list[Verdict] = []
    last = _last(focus)
    n = max(n_games, 1)
    if sf is not None and sf.h2_min > 0:
        out.append(Verdict(
            "Keep the starting five" if sf.not_the_problem else "Rethink the starting five",
            f"{sf.h2_pm / n:+.1f} vs {sf.rest_h2_pm / n:+.1f}",
            "Points a game after halftime: the five starters on the floor together vs every other lineup",
            "good" if sf.not_the_problem else "bad", "starters",
            ("diverge", (sf.h2_pm / n, sf.rest_h2_pm / n), ("starters", "others"))))
    if bl is not None:
        out.append(Verdict(
            f"{last}: less ball after halftime, for now" if bl.should_handle_less else f"{last}'s workload is fine",
            f"{bl.tov_pct_h2:.0%}", f"of {last}'s plays after halftime end in a turnover. Last season it was "
            f"{bl.prior_tov_pct_h2:.0%}.",
            "bad" if bl.should_handle_less else "good", "garland",
            ("pair", bl.prior_tov_pct_h2, bl.tov_pct_h2, "last year", "now")))
    if pr is not None:
        if pr.trapped:
            out.append(Verdict(f"Teams are trapping {last}", f"{pr.help_share:.0%}",
                               f"of {last}'s turnovers are stolen by a second defender, not the one guarding him "
                               f"(teammates: {pr.mates_help_share:.0%})",
                               "bad", "garland", ("share", pr.help_share, pr.mates_help_share)))
        elif pr.length_problem:
            out.append(Verdict(f"{last} struggles against bigger defenders", f"{pr.size_ratio:.1f}x",
                               f"as many turnovers when the player guarding {last} is {_ft_in(LONG_IN)} or taller. "
                               + ("It was a problem last season too. " if pr.size_gap_last_season else
                                  "Last season the defender's height made no difference. ")
                               + "It's not double teams.",
                               "warn", "garland", ("pair", pr.tov100_short, pr.tov100_long, "shorter",
                                                   f"{_ft_in(LONG_IN)}+")))
    shooters = [c for c in cands if "threes" in c.helps and not c.small_sample]
    if shooters:
        out.append(Verdict("Biggest fix: the returning shooters",
                           " · ".join(f"{c.fg3_pct:.0%}" for c in shooters),
                           f"Three-point shooting after halftime for {_names([_last(c.name) for c in shooters])} "
                           f"(last full season). The Clippers shoot {sd.team_fg3_h2:.0%} now.", "good", "returning",
                           ("pair", sd.team_fg3_h2, max(c.fg3_pct for c in shooters), "LAC now",
                            _last(max(shooters, key=lambda c: c.fg3_pct).name)[:9])))
    if gl is not None and gl.worse:
        out.append(Verdict("Finish possessions with a rebound", f"{gl.opp_oreb_h2:.0%}",
                           f"of their own missed shots the other team rebounds after halftime. Last season it was "
                           f"{gl.prior_opp_oreb_h2:.0%}.",
                           "bad", "defense", ("pair", gl.prior_opp_oreb_h2, gl.opp_oreb_h2, "last year", "now")))
    if defense is not None and rot is not None and len(defense.table):
        q4 = rot.set_index("player_id")["q4_mpg"]
        small = defense.table[defense.table["gives_up_size"]].copy()
        if len(small):
            r = small.assign(q4=small["player_id"].map(q4).fillna(0)).sort_values("q4", ascending=False).iloc[0]
            who = _last(r['player_name'])
            out.append(Verdict(f"{who} gets out-muscled", f"{r['own_wt']:.0f} vs {r['opp_wt']:.0f} lb",
                               f"{who}'s weight vs the average weight of the players {who} guards. They shoot "
                               f"{r['fg_pct']:.0%} against {who}.",
                               "warn", "defense", ("pair", r["own_wt"], r["opp_wt"], who[:9], "guards")))
    if rot is not None and len(rot):
        more = rot[rot["earned"]]
        less = rot[rot["struggling"]]
        if len(more):
            out.append(Verdict("More minutes: " + " & ".join(_last(x) for x in more["player_name"]),
                               " · ".join(f"{v:+.0f}" for v in more["h2_on_off"]),
                               "More points per 100 trips down the floor when they play after halftime than when "
                               "they sit. Small samples.",
                               "good", "minutes", ("diverge", tuple(more["h2_on_off"]),
                                                   tuple(_last(x)[:8] for x in more["player_name"]))))
        if len(less):
            out.append(Verdict("Fewer late minutes: " + ", ".join(_last(x) for x in less["player_name"]),
                               " · ".join(f"{v:+.0f}" for v in less["h2_pm_pg"]),
                               "Points a game the Clippers lose while each of them is on the floor after halftime, "
                               "in that order.",
                               "bad", "minutes", ("diverge", tuple(less["h2_pm_pg"]),
                                                  tuple(_last(x)[:8] for x in less["player_name"]))))
    return out
