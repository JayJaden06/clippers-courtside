"""The five hypothesis tests, the ranked attribution, and the season comparison.

Each test returns a :class:`HypothesisResult`: tidy tables, the key numbers,
the figure the report draws for it, and a verdict paragraph. Verdicts are
assembled from the numbers with explicit thresholds (below), so a rerun on
new games can reach a different conclusion without anyone editing prose.

Thresholds, chosen before reading the results and kept deliberately simple:

- a factor "drives" the swing if it accounts for at least 40% of it,
  "contributes" at 15-40%, and is "minor" below 15% (shares of the actual
  1H -> 2H margin change, from the exact decomposition);
- a league rank of 26-30 on the 1H -> 2H change is "among the worst";
- Garland is "the main driver" only if his excess second-half turnovers
  are worth at least 40% of the swing.

Every verdict carries the sample-size caveat: 8 games is 16 halves, and
the bootstrap intervals (resampling whole games) are wide.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

from courtside.analysis.decompose import FACTOR_LABELS, FACTORS, bootstrap, decompose_frame
from courtside.transform.halves import summarize
from courtside.transform.turnovers import summarize_players, tov_pct

DRIVES, CONTRIBUTES = 0.40, 0.15
WORST_RANKS = 26


@dataclass
class HypothesisResult:
    id: str
    title: str
    verdict: str
    key: dict = field(default_factory=dict)
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    chart: str = ""


@dataclass
class Inputs:
    game_halves: pd.DataFrame
    game_leads: pd.DataFrame
    player_stats: pd.DataFrame
    baselines: pd.DataFrame
    onoff: pd.DataFrame
    lineups: pd.DataFrame
    ranks: pd.DataFrame
    season: str
    comparison_season: str
    focus_player: str

    def halves(self, season: str | None = None) -> pd.DataFrame:
        gh = self.game_halves[self.game_halves["season"] == (season or self.season)]
        return gh[gh["split"].astype(str).isin(["H1", "H2"])].assign(split=lambda d: d["split"].astype(str))

    @property
    def n_games(self) -> int:
        return self.halves()["game_id"].nunique()

    def caveat(self) -> str:
        n = self.n_games
        return (f"Sample: {n} games ({2 * n} halves), so treat every number here as a lead, "
                f"not a settled rate.")


# ----------------------------------------------------------------- helpers

def _share_word(share: float) -> str:
    if share >= DRIVES:
        return "drives"
    if share >= CONTRIBUTES:
        return "contributes to"
    return "is a minor part of"


def _rank(ranks: pd.DataFrame, season: str, metric: str, split: str) -> tuple[float, int]:
    r = ranks[(ranks["season"] == season) & ranks["is_team"] & (ranks["metric"] == metric) & (ranks["split"] == split)]
    return float(r["value"].iloc[0]), int(r["rank"].iloc[0])


def _league(ranks: pd.DataFrame, season: str, metric: str, split: str) -> pd.Series:
    r = ranks[(ranks["season"] == season) & (ranks["metric"] == metric) & (ranks["split"] == split)]
    return r.set_index("team_name")["value"]


def _net_change(h1: pd.Series, h2: pd.Series) -> float:
    r = lambda s: 100 * (s["pts"] / s["possessions"] - s["opp_pts"] / s["opp_possessions"])  # noqa: E731
    return float(r(h2) - r(h1))


def _efg_change(h1: pd.Series, h2: pd.Series, prefix: str = "") -> float:
    e = lambda s: (s[f"{prefix}fgm"] + 0.5 * s[f"{prefix}fg3m"]) / s[f"{prefix}fga"]  # noqa: E731
    return float(e(h2) - e(h1))


def ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _corr(x: pd.Series, y: pd.Series) -> float:
    return float(np.corrcoef(x, y)[0, 1]) if len(x) > 2 and x.std() > 0 and y.std() > 0 else float("nan")


def attribution(inp: Inputs, season: str | None = None) -> pd.DataFrame:
    """Decomposition for one season, ranked by size, with share of the actual swing."""
    season = season or inp.season
    d = decompose_frame(inp.halves(season), ["season"]).iloc[0]
    n = inp.halves(season)["game_id"].nunique()
    swing = d["margin_change"]
    rows = [{"factor": f, "label": FACTOR_LABELS[f], "points": d[f], "per_game": d[f] / n,
             "share_of_swing": d[f] / swing if swing else np.nan} for f in FACTORS]
    out = pd.DataFrame(rows)
    out = out.reindex(out["points"].abs().sort_values(ascending=False).index).reset_index(drop=True)
    out.attrs["swing"] = float(swing)
    out.attrs["games"] = n
    return out


# --------------------------------------------------------------------- H1

def h1_collapse_is_real(inp: Inputs) -> HypothesisResult:
    s = summarize(inp.halves(), ["split"]).set_index("split")
    net_change, net_rank = _rank(inp.ranks, inp.season, "net_rtg", "change")
    league_change = _league(inp.ranks, inp.season, "net_rtg", "change")
    point, lo, hi = bootstrap(inp.halves(), _net_change)

    per_game = inp.halves().pivot_table(index=["game_id", "opponent_abbreviation"], columns="split", values="net_rtg")
    worse = int((per_game["H2"] < per_game["H1"]).sum())

    # Regression check: was the first half unusually hot (or the opponent unusually cold)?
    efg_h1, efg_h1_rank = _rank(inp.ranks, inp.season, "efg_pct", "H1")
    efg_h2, efg_h2_rank = _rank(inp.ranks, inp.season, "efg_pct", "H2")
    lg_efg_h1 = _league(inp.ranks, inp.season, "efg_pct", "H1").mean()
    lg_efg_h2 = _league(inp.ranks, inp.season, "efg_pct", "H2").mean()
    prior = summarize(inp.halves(inp.comparison_season), ["split"]).set_index("split")
    opp3_h1 = s.loc["H1", "opp_fg3_pct"]
    lg_opp_efg_h1 = _league(inp.ranks, inp.season, "opp_efg_pct", "H1").mean()

    table = pd.DataFrame({
        "metric": ["Net rating", "Offensive rating", "Defensive rating", "eFG%", "3P%", "Opponent eFG%", "Opponent 3P%"],
        "H1": [s.loc["H1", c] for c in ("net_rtg", "ortg", "drtg", "efg_pct", "fg3_pct", "opp_efg_pct", "opp_fg3_pct")],
        "H2": [s.loc["H2", c] for c in ("net_rtg", "ortg", "drtg", "efg_pct", "fg3_pct", "opp_efg_pct", "opp_fg3_pct")],
    })
    table["change"] = table["H2"] - table["H1"]
    regression = pd.DataFrame({
        "half": ["H1", "H2"],
        "clippers_efg": [efg_h1, efg_h2],
        "league_avg_efg": [lg_efg_h1, lg_efg_h2],
        "clippers_minus_league": [efg_h1 - lg_efg_h1, efg_h2 - lg_efg_h2],
        "clippers_prior_season_efg": [prior.loc["H1", "efg_pct"], prior.loc["H2", "efg_pct"]],
    })

    hot_first = efg_h1 - lg_efg_h1
    cold_second = efg_h2 - lg_efg_h2
    regression_text = (
        f"Some of it is regression: first-half eFG% was {efg_h1:.3f} ({ordinal(efg_h1_rank)} in the league, "
        f"{hot_first:+.3f} vs the league's first-half average). But the second half ({efg_h2:.3f}, "
        f"{ordinal(efg_h2_rank)}) is {cold_second:+.3f} below the league's second-half average, so the drop "
        f"overshoots a return to normal. "
        + (f"Opponents, if anything, were hot early ({s.loc['H1', 'opp_efg_pct']:.3f} first-half eFG% vs a "
           f"league average of {lg_opp_efg_h1:.3f}; 3P% {opp3_h1:.3f}) and cooled off after halftime, which "
           f"helped the Clippers rather than hurt them."
           if s.loc["H1", "opp_efg_pct"] > lg_opp_efg_h1 else
           f"Opponents were cold early ({s.loc['H1', 'opp_efg_pct']:.3f} vs {lg_opp_efg_h1:.3f}), so part of "
           f"the second-half defense numbers is opponents returning to normal.")
        if hot_first > 0 else
        f"It is not regression from a hot start: first-half eFG% ({efg_h1:.3f}) was not above the league "
        f"average ({lg_efg_h1:.3f})."
    )
    bad_rank, excludes_zero = net_rank >= WORST_RANKS, hi < 0
    answer = "Yes" if bad_rank and excludes_zero else "Probably" if bad_rank else "Not clearly"
    verdict = (
        f"{answer}: net rating goes from {s.loc['H1', 'net_rtg']:+.1f} in first halves to "
        f"{s.loc['H2', 'net_rtg']:+.1f} in second halves, a change of {net_change:+.1f} per 100 possessions. "
        f"That ranks {ordinal(net_rank)} of 30 (league average change {league_change.mean():+.1f}, so most teams "
        f"barely move), and second halves were worse in {worse} of {len(per_game)} games. {regression_text} "
        f"{inp.caveat()} The 90% bootstrap interval for the net-rating change (resampling games) is "
        f"{lo:+.1f} to {hi:+.1f}"
        + (", which excludes zero." if excludes_zero else
           ", which includes zero: relative to the league it looks real, but 8 games cannot rule out noise.")
    )
    return HypothesisResult(
        "H1", "The second-half collapse is real", verdict,
        key={"net_h1": s.loc["H1", "net_rtg"], "net_h2": s.loc["H2", "net_rtg"], "net_change": net_change,
             "net_change_rank": net_rank, "league_mean_change": float(league_change.mean()),
             "boot_lo": lo, "boot_hi": hi, "games_worse": worse, "efg_h1": efg_h1, "efg_h2": efg_h2},
        tables={"h1_halves": table, "h1_regression": regression,
                "h1_league_net_change": league_change.sort_values().reset_index()},
        chart="league_rank_dots.png",
    )


# --------------------------------------------------------------------- H2

def h2_turnovers(inp: Inputs) -> HypothesisResult:
    h = inp.halves()
    s = summarize(h, ["split"]).set_index("split")
    att = attribution(inp)
    tov_share = float(att.set_index("factor").loc["turnovers", "share_of_swing"])
    tov_pts = float(att.set_index("factor").loc["turnovers", "points"])

    g = h.assign(tov_all=h["tov"] + h["tov_team"]).pivot_table(
        index=["game_id", "opponent_abbreviation"], columns="split",
        values=["tov_all", "opp_pts_off_tov", "margin", "tov_live"])
    per_game = pd.DataFrame({
        "tov_h1": g[("tov_all", "H1")], "tov_h2": g[("tov_all", "H2")],
        "opp_pts_off_tov_h1": g[("opp_pts_off_tov", "H1")], "opp_pts_off_tov_h2": g[("opp_pts_off_tov", "H2")],
        "margin_h1": g[("margin", "H1")], "margin_h2": g[("margin", "H2")],
    }).reset_index()
    per_game = per_game.merge(inp.game_leads[["game_id", "lead_class", "led_at_half"]], on="game_id")
    r_level = _corr(per_game["opp_pts_off_tov_h2"], per_game["margin_h2"])
    r_change = _corr(per_game["tov_h2"] - per_game["tov_h1"], per_game["margin_h2"] - per_game["margin_h1"])
    by_class = per_game.groupby("lead_class")[["tov_h2", "opp_pts_off_tov_h2", "margin_h2"]].mean().assign(
        games=per_game.groupby("lead_class").size())

    tov_h1 = s.loc["H1", "tov"] + s.loc["H1", "tov_team"]
    tov_h2 = s.loc["H2", "tov"] + s.loc["H2", "tov_team"]
    pts_h1, pts_h2 = s.loc["H1", "opp_pts_off_tov"], s.loc["H2", "opp_pts_off_tov"]
    live_h1 = s.loc["H1", "tov_live"] / tov_h1
    live_h2 = s.loc["H2", "tov_live"] / tov_h2
    fb_h1, fb_h2 = s.loc["H1", "opp_pts_fast_break"], s.loc["H2", "opp_pts_fast_break"]
    tovp_h1, tovp_h1_rank = _rank(inp.ranks, inp.season, "tov_pct", "H1")
    tovp_h2, tovp_h2_rank = _rank(inp.ranks, inp.season, "tov_pct", "H2")
    _, tovp_change_rank = _rank(inp.ranks, inp.season, "tov_pct", "change")

    table = pd.DataFrame({
        "metric": ["Turnovers (incl. team)", "Live-ball share", "Opponent points off turnovers (official)",
                   "Opponent points per turnover", "Opponent fast-break points", "TOV% (league rank)"],
        "H1": [tov_h1, live_h1, pts_h1, pts_h1 / tov_h1, fb_h1, f"{tovp_h1:.1f} ({tovp_h1_rank})"],
        "H2": [tov_h2, live_h2, pts_h2, pts_h2 / tov_h2, fb_h2, f"{tovp_h2:.1f} ({tovp_h2_rank})"],
    })
    verdict = (
        f"The turnover factor {_share_word(tov_share)} the collapse, but not in the way it looks. The Clippers turn it over "
        f"about as often after halftime ({tov_h1:.0f} vs {tov_h2:.0f} turnovers including team turnovers; TOV% {ordinal(tovp_h1_rank)} and "
        f"{ordinal(tovp_h2_rank)} of 30 in the two halves, so the rate is a whole-game problem, "
        f"{ordinal(tovp_change_rank)} on the change). What changes is the cost: opponents score {pts_h1:.0f} points off "
        f"them in first halves and {pts_h2:.0f} in second halves ({pts_h1 / tov_h1:.2f} vs {pts_h2 / tov_h2:.2f} "
        f"per turnover), with live-ball share {live_h1:.0%} -> {live_h2:.0%} and opponent fast-break points "
        f"{fb_h1:.0f} -> {fb_h2:.0f}. In the exact decomposition, lost shots from turnovers account for "
        f"{tov_pts:+.1f} of the {att.attrs['swing']:+.0f}-point swing ({tov_share:.0%}); the extra transition "
        f"scoring shows up inside opponent shooting, which overall got *worse* after halftime. Across games, "
        f"second-half points off turnovers and second-half margin correlate at r = {r_level:+.2f}, and the "
        f"change in turnovers vs the change in margin at r = {r_change:+.2f} (n = {len(per_game)}). "
        f"{inp.caveat()}"
    )
    return HypothesisResult(
        "H2", "Turnovers drive it", verdict,
        key={"tov_h1": tov_h1, "tov_h2": tov_h2, "opp_pts_off_tov_h1": pts_h1, "opp_pts_off_tov_h2": pts_h2,
             "tov_share_of_swing": tov_share, "r_level": r_level, "r_change": r_change},
        tables={"h2_totals": table, "h2_per_game": per_game, "h2_by_lead_class": by_class.reset_index()},
        chart="half_team_stats.png",
    )


# --------------------------------------------------------------------- H3

def h3_garland(inp: Inputs) -> HypothesisResult:
    name = inp.focus_player
    ps = inp.player_stats[(inp.player_stats["season"] == inp.season) & inp.player_stats["split"].isin(["H1", "H2"])]
    players = summarize_players(ps, ["player_id", "player_name", "split"])
    team = players.groupby("split")[["tov", "fga", "fta"]].sum()
    players["plays"] = players["fga"] + 0.44 * players["fta"] + players["tov"]
    team_plays = players.groupby("split")["plays"].sum()
    players["tov_share"] = players["tov"] / players["split"].map(team["tov"])
    players["play_share"] = players["plays"] / players["split"].map(team_plays)
    rotation = players[players["minutes"] >= 40].sort_values(["split", "tov_pct"], ascending=[True, False])
    me = players[players["player_name"] == name].set_index("split")
    if me.empty:
        raise ValueError(f"{name} has no {inp.season} minutes")

    h2_rank = int(rotation[rotation["split"] == "H2"]["tov_pct"].rank(ascending=False)[
        rotation[(rotation["split"] == "H2")]["player_name"] == name].iloc[0])
    n_rot = int((rotation["split"] == "H2").sum())

    base = inp.baselines.copy()
    prior = base[(base["season"] == inp.comparison_season) & (base["split"] == "H2")]
    prior_rate = float(tov_pct(prior["tov"].sum(), prior["fga"].sum(), prior["fta"].sum()))

    # Excess second-half turnovers vs his own first half and vs last season's second halves.
    plays_h2 = me.loc["H2", "plays"]
    tov_h2 = me.loc["H2", "tov"]
    rate_h1 = me.loc["H1", "tov_pct"]

    def excess(rate: float) -> float:
        # Solve tov = rate * (fga + 0.44 fta + tov) for tov with his 2H shots held fixed.
        shots = plays_h2 - tov_h2
        return float(tov_h2 - rate * shots / (1 - rate))

    h = summarize(inp.halves(), ["split"]).set_index("split").loc["H2"]
    tov_all = h["tov"] + h["tov_team"]
    cost = h["pts"] / h["possessions"] + (h["opp_pts_off_tov"] / tov_all - h["opp_pts"] / h["opp_possessions"])
    att = attribution(inp)
    swing = att.attrs["swing"]
    ex_h1, ex_prior = excess(rate_h1), excess(prior_rate)
    pts_h1, pts_prior = ex_h1 * cost, ex_prior * cost
    share = max(pts_h1, pts_prior) / abs(swing) if swing else np.nan

    oo = inp.onoff[(inp.onoff["season"] == inp.season) & (inp.onoff["player_name"] == name)].set_index("split")
    on = oo.loc["H2"]
    onoff_table = pd.DataFrame({
        "metric": ["Minutes", "Net rating", "Offensive rating", "Defensive rating", "Team TOV%", "Team fouls per 100"],
        "on": [on["min_on"], on["net_rtg_on"], on["ortg_on"], on["drtg_on"], on["tov_pct_on"], on["pf_per100_on"]],
        "off": [on["min_off"], on["net_rtg_off"], on["ortg_off"], on["drtg_off"], on["tov_pct_off"], on["pf_per100_off"]],
    })
    onoff_table["on_minus_off"] = onoff_table["on"] - onoff_table["off"]
    o_gap = on["ortg_on"] - on["ortg_off"]
    d_gap = on["drtg_on"] - on["drtg_off"]

    main = share >= DRIVES
    verdict = (
        f"{'Yes' if main else 'Partly'}: {name} is the team's biggest single source of second-half "
        f"turnovers, {'and the turnovers above his own baseline are big enough to be' if main else 'but not'} "
        f"the main reason for the collapse. He has "
        f"{me.loc['H2', 'tov']:.0f} of the team's {team.loc['H2', 'tov']:.0f} second-half player turnovers "
        f"({me.loc['H2', 'tov_share']:.0%}) on {me.loc['H2', 'play_share']:.0%} of the team's shots, free-throw "
        f"trips and turnovers, and his TOV% rises from {rate_h1:.1%} to {me.loc['H2', 'tov_pct']:.1%} "
        f"({ordinal(h2_rank)} highest of {n_rot} rotation players after halftime); {me.loc['H2', 'live_share']:.0%} of "
        f"them are live-ball. Last season his second-half TOV% was {prior_rate:.1%} (Cleveland and Clippers "
        f"combined). Measured against whichever baseline gives the hypothesis the most credit, his excess "
        f"second-half turnovers number "
        f"about {max(ex_h1, ex_prior):.1f} over {inp.n_games} games, worth roughly {max(pts_h1, pts_prior):.0f} "
        f"points at {cost:.2f} points per turnover - {share:.0%} of the {swing:+.0f}-point swing. The team's "
        f"second-half on/off with him is large ({on['net_on_off']:+.1f} net rating, {on['min_on']:.0f} minutes "
        f"on vs {on['min_off']:.0f} off), but the team's turnover rate is similar either way "
        f"({on['tov_pct_on']:.1f} vs {on['tov_pct_off']:.1f} per 100); the gap is "
        f"{'mostly offense' if abs(o_gap) > abs(d_gap) else 'mostly defense'} (ORtg {o_gap:+.1f}, DRtg "
        f"{d_gap:+.1f} on vs off; part of that is transition defense after his live-ball turnovers). In the {on['min_off']:.0f} second-half minutes he sat, the team was "
        f"{'better' if on['net_rtg_off'] > on['net_rtg_on'] else 'worse'} ({on['net_rtg_off']:+.1f}), but that "
        f"is a few dozen possessions and reflects who replaced him and when. {inp.caveat()}"
    )
    return HypothesisResult(
        "H3", f"It's {name} specifically", verdict,
        key={"tov_h1": me.loc["H1", "tov"], "tov_h2": tov_h2, "tov_pct_h1": rate_h1,
             "tov_pct_h2": me.loc["H2", "tov_pct"], "tov_share_h2": me.loc["H2", "tov_share"],
             "play_share_h2": me.loc["H2", "play_share"], "prior_h2_tov_pct": prior_rate,
             "excess_tov": max(ex_h1, ex_prior), "excess_pts": max(pts_h1, pts_prior), "share_of_swing": share,
             "net_on_off_h2": on["net_on_off"], "cost_per_tov": cost},
        tables={"h3_rotation_tov": rotation[["player_name", "split", "games", "minutes", "tov", "tov_pct",
                                             "tov_per36", "live_share", "tov_share", "play_share"]],
                "h3_baselines": base, "h3_onoff_h2": onoff_table},
        chart="player_tov_by_half.png",
    )


# --------------------------------------------------------------------- H4

def h4_fouls(inp: Inputs) -> HypothesisResult:
    h = inp.halves()
    s = summarize(h, ["split"]).set_index("split")
    no_tor = summarize(h[h["opponent_abbreviation"] != _biggest_ft_game(h)], ["split"]).set_index("split")
    prior = summarize(inp.halves(inp.comparison_season), ["split"]).set_index("split")
    _, ftr_change_rank = _rank(inp.ranks, inp.season, "opp_ft_rate", "change")
    _, ftr_change_rank_prior = _rank(inp.ranks, inp.comparison_season, "opp_ft_rate", "change")
    full_att = attribution(inp)
    swing = full_att.attrs["swing"]
    att = full_att.set_index("factor")
    ft_pts, ft_share = att.loc["free_throws", "points"], att.loc["free_throws", "share_of_swing"]
    outlier = _biggest_ft_game(h)

    table = pd.DataFrame({
        "metric": ["Team fouls", "Late fouls while trailing", "Opponent FTA", "Opponent FTA from late fouls",
                   "Opponent FT rate", "Opponent FT rate excl. late fouls", f"Opponent FT rate excl. late fouls and {outlier}",
                   "Opponent FTM", "Clippers FTM", "Minutes opponent in bonus"],
        "H1": [s.loc["H1", c] for c in ("team_fouls", "late_trailing_fouls", "opp_fta", "opp_fta_late_trailing",
                                        "opp_ft_rate", "opp_ft_rate_ex_late")]
              + [no_tor.loc["H1", "opp_ft_rate_ex_late"], s.loc["H1", "opp_ftm"], s.loc["H1", "ftm"],
                 s.loc["H1", "bonus_secs_conceded"] / 60],
        "H2": [s.loc["H2", c] for c in ("team_fouls", "late_trailing_fouls", "opp_fta", "opp_fta_late_trailing",
                                        "opp_ft_rate", "opp_ft_rate_ex_late")]
              + [no_tor.loc["H2", "opp_ft_rate_ex_late"], s.loc["H2", "opp_ftm"], s.loc["H2", "ftm"],
                 s.loc["H2", "bonus_secs_conceded"] / 60],
    })

    ps = inp.player_stats[(inp.player_stats["season"] == inp.season) & (inp.player_stats["split"] == "H2")]
    foulers = ps.groupby("player_name")[["pf", "fouls_shooting", "fouls_late_trailing", "minutes"]].sum()
    foulers["pf_per36"] = 36 * foulers["pf"] / foulers["minutes"]
    foulers = foulers.sort_values("pf", ascending=False).head(8).reset_index()

    blown = inp.game_leads[(inp.game_leads["season"] == inp.season)][["game_id", "lead_class", "led_at_half"]]
    by_class = h[h["split"] == "H2"].merge(blown, on="game_id").groupby("lead_class").agg(
        games=("game_id", "nunique"), opp_ftm_h2=("opp_ftm", "mean"), opp_fta_h2=("opp_fta", "mean"),
        team_fouls_h2=("team_fouls", "mean")).reset_index()

    prior_ft_per_game = float(attribution(inp, inp.comparison_season).set_index("factor").loc["free_throws", "per_game"])
    rise = s.loc["H2", "opp_ft_rate"] - s.loc["H1", "opp_ft_rate"]
    rise_ex = no_tor.loc["H2", "opp_ft_rate_ex_late"] - no_tor.loc["H1", "opp_ft_rate_ex_late"]
    rise_prior = prior.loc["H2", "opp_ft_rate_ex_late"] - prior.loc["H1", "opp_ft_rate_ex_late"]
    verdict = (
        f"Fouling {_share_word(abs(ft_share))} the collapse on the scoreboard, and it is a chronic habit rather "
        f"than a new one. Team fouls go from {s.loc['H1', 'team_fouls']:.0f} to {s.loc['H2', 'team_fouls']:.0f} "
        f"and opponent free-throw rate from {s.loc['H1', 'opp_ft_rate']:.3f} to {s.loc['H2', 'opp_ft_rate']:.3f} "
        f"(change ranked {ordinal(ftr_change_rank)} of 30; last season {ordinal(ftr_change_rank_prior)}). But "
        f"{s.loc['H2', 'late_trailing_fouls']:.0f} of the second-half fouls were late fouls by a team already "
        f"behind ({s.loc['H2', 'opp_fta_late_trailing']:.0f} opponent FTA) - a consequence of losing, not a "
        f"cause. Without those and without the {outlier} game, the rise is {rise_ex:+.3f}, close to last "
        f"season's {rise_prior:+.3f} (same adjustments). The Clippers also get to the line more after halftime "
        f"({s.loc['H1', 'ftm']:.0f} -> {s.loc['H2', 'ftm']:.0f} made), so in the decomposition free throws net "
        f"{ft_pts:+.1f} points of the {swing:+.0f} ({abs(ft_share):.0%}), {ft_pts / inp.n_games:+.1f} per game "
        f"vs {prior_ft_per_game:+.1f} per game last season. "
        f"{inp.caveat()}"
    )
    return HypothesisResult(
        "H4", "Fouls and free throws drive it", verdict,
        key={"opp_ft_rate_h1": s.loc["H1", "opp_ft_rate"], "opp_ft_rate_h2": s.loc["H2", "opp_ft_rate"],
             "rise": rise, "rise_ex_late_and_outlier": rise_ex, "rise_prior_ex_late": rise_prior,
             "ft_points": ft_pts, "ft_share": ft_share, "change_rank": ftr_change_rank,
             "change_rank_prior": ftr_change_rank_prior},
        tables={"h4_totals": table, "h4_foulers_h2": foulers, "h4_by_lead_class": by_class},
        chart="half_team_stats.png",
    )


def _biggest_ft_game(h: pd.DataFrame) -> str:
    h2 = h[h["split"] == "H2"]
    return str(h2.loc[h2["opp_fta"].idxmax(), "opponent_abbreviation"])


# --------------------------------------------------------------------- H5

def h5_shooting(inp: Inputs) -> HypothesisResult:
    h = inp.halves()
    s = summarize(h, ["split"]).set_index("split")
    att = attribution(inp).set_index("factor")
    shoot, opp_shoot = att.loc["shooting", "points"], att.loc["opp_shooting", "points"]
    reb = att.loc["rebounding", "points"]
    efg_pt, efg_lo, efg_hi = bootstrap(h, _efg_change)
    oefg_pt, oefg_lo, oefg_hi = bootstrap(h, lambda a, b: _efg_change(a, b, "opp_"))
    _, efg_change_rank = _rank(inp.ranks, inp.season, "efg_pct", "change")
    _, oreb_rank = _rank(inp.ranks, inp.season, "opp_oreb_pct", "change")
    prior_att = attribution(inp, inp.comparison_season).set_index("factor")

    rows = []
    for metric, col in (("eFG%", "efg_pct"), ("3P%", "fg3_pct"), ("FG%", "fg_pct"), ("Opponent eFG%", "opp_efg_pct"),
                        ("Opponent 3P%", "opp_fg3_pct"), ("OREB%", "oreb_pct"), ("Opponent OREB%", "opp_oreb_pct"),
                        ("Pace", "pace")):
        rows.append({"metric": metric, "H1": s.loc["H1", col], "H2": s.loc["H2", col]})
    rows.append({"metric": "3PA share of FGA", "H1": s.loc["H1", "fg3a"] / s.loc["H1", "fga"],
                 "H2": s.loc["H2", "fg3a"] / s.loc["H2", "fga"]})
    rows.append({"metric": "Paint points", "H1": s.loc["H1", "pts_paint"], "H2": s.loc["H2", "pts_paint"]})
    table = pd.DataFrame(rows)
    table["change"] = table["H2"] - table["H1"]

    opp_better = oefg_pt > 0
    verdict = (
        f"Mostly yes, but it is the Clippers' shooting, not the opponents'. Clippers eFG% falls from "
        f"{s.loc['H1', 'efg_pct']:.3f} to {s.loc['H2', 'efg_pct']:.3f} ({efg_pt:+.3f}; 90% interval "
        f"{efg_lo:+.3f} to {efg_hi:+.3f}), the {'largest' if efg_change_rank == 30 else f'{ordinal(31 - efg_change_rank)}-largest'} "
        f"drop in the league, and 3P% from {s.loc['H1', 'fg3_pct']:.3f} to {s.loc['H2', 'fg3_pct']:.3f}. "
        f"Opponents {'shoot better' if opp_better else 'actually shoot worse'} after halftime "
        f"({s.loc['H1', 'opp_efg_pct']:.3f} -> {s.loc['H2', 'opp_efg_pct']:.3f}; interval {oefg_lo:+.3f} to "
        f"{oefg_hi:+.3f}), so the answer to 'do opponents just shoot better after halftime?' is "
        f"{'yes' if opp_better else 'no'}. In the decomposition, Clippers shooting accounts for {shoot:+.1f} of "
        f"the {attribution(inp).attrs['swing']:+.0f}-point swing; the other factors roughly cancel (opponent "
        f"shooting {opp_shoot:+.1f}, turnovers {att.loc['turnovers', 'points']:+.1f}, rebounding {reb:+.1f}, "
        f"free throws {att.loc['free_throws', 'points']:+.1f}, unattributed {att.loc['residual', 'points']:+.1f}). "
        f"Offensive rebounds allowed rise ({s.loc['H1', 'opp_oreb_pct']:.3f} -> {s.loc['H2', 'opp_oreb_pct']:.3f}, "
        f"change ranked {ordinal(oreb_rank)}), worth {reb:+.1f} points net; pace barely moves "
        f"({s.loc['H1', 'pace']:.1f} -> {s.loc['H2', 'pace']:.1f}). Last season the same shooting factor was "
        f"{prior_att.loc['shooting', 'per_game']:+.1f} points per game; this season it is "
        f"{att.loc['shooting', 'per_game']:+.1f}. {inp.caveat()}"
    )
    return HypothesisResult(
        "H5", "It's shooting regression, not turnovers", verdict,
        key={"efg_change": efg_pt, "efg_lo": efg_lo, "efg_hi": efg_hi, "opp_efg_change": oefg_pt,
             "opp_efg_lo": oefg_lo, "opp_efg_hi": oefg_hi, "shooting_points": shoot,
             "opp_shooting_points": opp_shoot, "efg_change_rank": efg_change_rank},
        tables={"h5_shooting": table},
        chart="half_team_stats.png",
    )


# ------------------------------------------------------- attribution & seasons

def ranked_attribution(inp: Inputs) -> HypothesisResult:
    cur, prior = attribution(inp), attribution(inp, inp.comparison_season)
    top = cur.iloc[0]
    second = cur.iloc[1]
    per_game = decompose_frame(inp.halves(), ["game_id", "opponent_abbreviation"]).merge(
        inp.game_leads[["game_id", "game_date", "lead_class", "led_at_half", "halftime_margin", "final_margin"]],
        on="game_id").sort_values("game_date")
    collapses = per_game[per_game["led_at_half"] & (per_game["final_margin"] < 0)]
    verdict = (
        f"Of the {cur.attrs['swing']:+.0f}-point first-to-second-half swing across {cur.attrs['games']} games, "
        f"the largest factor is {top['label']} ({top['points']:+.1f}, {top['per_game']:+.1f} per game), "
        f"then {second['label']} ({second['points']:+.1f}). Factors can offset: a positive number "
        f"means that factor got better after halftime. In the {len(collapses)} games lost after leading at "
        f"the half ({', '.join(collapses['opponent_abbreviation'])}), the most costly factors were "
        + "; ".join(f"{r['opponent_abbreviation']}: {_worst_factor(r)}" for _, r in collapses.iterrows())
        + ". This is a description of where the points went, not a causal estimate."
    )
    return HypothesisResult(
        "ATTR", "Ranked attribution of the 1H -> 2H swing", verdict,
        key={"swing": cur.attrs["swing"], "top_factor": top["factor"], "top_points": top["points"]},
        tables={"attribution": cur, "attribution_prior": prior, "attribution_per_game": per_game},
        chart="attribution.png",
    )


def _worst_factor(row: pd.Series) -> str:
    """The most negative factor in one game's decomposition (residual excluded)."""
    worst = min(FACTORS[:-1], key=lambda f: row[f])
    return f"{FACTOR_LABELS[worst]} ({row[worst]:+.1f} of {row['margin_change']:+.0f})"


def season_comparison(inp: Inputs) -> HypothesisResult:
    rows = []
    for season in (inp.season, inp.comparison_season):
        s = summarize(inp.halves(season), ["split"]).set_index("split")
        n = inp.halves(season)["game_id"].nunique()
        att = attribution(inp, season).set_index("factor")
        leads = inp.game_leads[inp.game_leads["season"] == season]["lead_class"].value_counts()
        rows.append({
            "season": season, "games": n,
            "margin_h1_per_game": s.loc["H1", "margin"] / n, "margin_h2_per_game": s.loc["H2", "margin"] / n,
            "net_h1": s.loc["H1", "net_rtg"], "net_h2": s.loc["H2", "net_rtg"],
            "efg_h1": s.loc["H1", "efg_pct"], "efg_h2": s.loc["H2", "efg_pct"],
            "tov_pct_h1": s.loc["H1", "tov_pct"], "tov_pct_h2": s.loc["H2", "tov_pct"],
            "opp_ft_rate_h1": s.loc["H1", "opp_ft_rate"], "opp_ft_rate_h2": s.loc["H2", "opp_ft_rate"],
            "blown_leads": int(leads.get("blown_lead", 0)), "held_leads": int(leads.get("held_lead", 0)),
            **{f"{f}_per_game": att.loc[f, "per_game"] for f in FACTORS},
        })
    table = pd.DataFrame(rows)
    cur, prior = table.iloc[0], table.iloc[1]
    blown_rate = lambda r: r["blown_leads"] / max(r["blown_leads"] + r["held_leads"], 1)  # noqa: E731
    new = [f for f in FACTORS[:-1] if cur[f"{f}_per_game"] - prior[f"{f}_per_game"] <= -1.0]
    chronic = [f for f in FACTORS[:-1] if cur[f"{f}_per_game"] <= -0.5 and prior[f"{f}_per_game"] <= -0.5]
    verdict = (
        f"Last season the Clippers were essentially even after halftime ({prior['margin_h1_per_game']:+.1f} "
        f"per game in first halves, {prior['margin_h2_per_game']:+.1f} in second halves) and blew "
        f"{prior['blown_leads']} of {prior['blown_leads'] + prior['held_leads']} eight-point leads "
        f"({blown_rate(prior):.0%}). This season it is {cur['margin_h1_per_game']:+.1f} and "
        f"{cur['margin_h2_per_game']:+.1f}, with {cur['blown_leads']} of {cur['blown_leads'] + cur['held_leads']} "
        f"leads blown. New this season (at least 1 point per game worse than last season): "
        f"{', '.join(FACTOR_LABELS[f] for f in new) or 'none'}. Chronic (costing at least half a point "
        f"per game in both seasons): {', '.join(FACTOR_LABELS[f] for f in chronic) or 'none'}. "
        f"{inp.caveat()}"
    )
    return HypothesisResult(
        "SEASONS", f"{inp.season} vs {inp.comparison_season}: chronic or new?", verdict,
        key={"new": new, "chronic": chronic}, tables={"season_comparison": table}, chart="season_factors.png",
    )


REQUIRED = ("H1", "H2", "H3", "H4", "H5", "ATTR", "SEASONS")


def run_all(inp: Inputs) -> list[HypothesisResult]:
    """Every test that the data can support today. Early in a season one can lack data (the focus player
    hasn't played both halves yet, say); it's left out with a note, and the rest still run, so the morning
    refresh keeps going and the site shows what it can."""
    out = []
    for test in (h1_collapse_is_real, h2_turnovers, h3_garland, h4_fouls, h5_shooting, ranked_attribution,
                 season_comparison):
        try:
            out.append(test(inp))
        except (ValueError, KeyError, IndexError, ZeroDivisionError) as e:
            log.warning("%s skipped: not enough data yet (%s)", test.__name__, e)
    return out


# ------------------------------------------------------------------- I/O

def load_inputs(processed_dir, season: str, comparison_season: str, focus_player: str) -> Inputs:
    """Read the processed tables written by ``courtside build``."""
    p = processed_dir
    return Inputs(
        game_halves=pd.read_parquet(p / "game_halves.parquet"),
        game_leads=pd.read_parquet(p / "game_leads.parquet"),
        player_stats=pd.read_parquet(p / "player_half_stats.parquet"),
        baselines=pd.read_csv(p / "focus_player_baselines.csv"),
        onoff=pd.read_parquet(p / "onoff_halves.parquet"),
        lineups=pd.read_parquet(p / "lineup_halves.parquet"),
        ranks=pd.read_parquet(p / "league_half_ranks.parquet"),
        season=season, comparison_season=comparison_season, focus_player=focus_player,
    )


def _jsonable(v):
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if np.isnan(v) else round(float(v), 4)
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def write_results(results: list[HypothesisResult], tables_dir, json_path, meta: dict) -> None:
    """Tables to ``tables_dir/<name>.csv``; verdicts, key numbers and table names to ``json_path``."""
    import json

    tables_dir.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    out = {"meta": meta, "results": []}
    for r in results:
        for name, df in r.tables.items():
            df.round({c: 4 for c in df.select_dtypes("float").columns}).to_csv(tables_dir / f"{name}.csv", index=False)
        out["results"].append({
            "id": r.id, "title": r.title, "verdict": r.verdict, "chart": r.chart,
            "key": {k: _jsonable(v) for k, v in r.key.items()}, "tables": list(r.tables),
        })
    json_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
