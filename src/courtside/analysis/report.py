"""Render ``reports/findings.md`` and its figures from the hypothesis results.

Everything here is generated: the executive summary is assembled from the
same key numbers as the verdicts, with the same thresholds, so a rerun after
new games can tell a different story without anyone editing prose. Running
it twice on the same processed data gives the same files.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from courtside.transform.halves import summarize
from courtside.analysis.hypotheses import DRIVES, CONTRIBUTES, HypothesisResult, Inputs, attribution, ordinal
from courtside.viz import charts

FIRST_LOOK = [
    ("attribution.png", "where the second-half points went; the whole answer in one picture"),
    ("player_tov_by_half.png", "the Garland question: his turnover rate next to everyone else's"),
    ("margin_small_multiples.png", "what the collapses actually looked like, game by game"),
]

# Extra figures shown in a section besides the result's own chart.
EXTRA_FIGURES = {"H1": ["margin_small_multiples.png"], "H3": ["lineup_net_2h.png"]}

# Tables shown per section: (table name, columns or None for all, optional row filter).
SECTION_TABLES: dict[str, list[tuple[str, list[str] | None]]] = {
    "H1": [("h1_halves", None), ("h1_regression", None)],
    "H2": [("h2_totals", None), ("h2_by_lead_class", None)],
    "H3": [("h3_baselines", ["season", "team", "split", "games", "minutes", "tov", "tov_pct", "live_share"]),
           ("h3_onoff_h2", None)],
    "H4": [("h4_totals", None), ("h4_foulers_h2", None)],
    "H5": [("h5_shooting", None)],
    "ATTR": [("attribution", ["label", "points", "per_game", "share_of_swing"]),
             ("attribution_per_game", ["game_date", "opponent_abbreviation", "halftime_margin", "final_margin",
                                       "margin_change", "shooting", "opp_shooting", "turnovers", "rebounding",
                                       "free_throws", "pace", "residual"])],
    "SEASONS": [("season_comparison", None)],
}


# ------------------------------------------------------------- figures

def team_halves(inp: Inputs, seasons: list[str]) -> pd.DataFrame:
    """Season totals by half (rates from summed counts), the same numbers the verdicts quote."""
    return pd.concat([summarize(inp.halves(s), ["split"]).assign(season=s) for s in seasons], ignore_index=True)


def draw_figures(inp: Inputs, results: list[HypothesisResult], margins: pd.DataFrame, figures_dir: Path,
                 lead_threshold: int, source: str) -> list[Path]:
    by_id = {r.id: r for r in results}
    seasons = [inp.season, inp.comparison_season]
    rotation = by_id["H3"].tables["h3_rotation_tov"]
    return [
        charts.margin_small_multiples(margins, inp.game_leads, inp.season, lead_threshold,
                                      figures_dir / "margin_small_multiples.png", source),
        charts.half_team_stats(team_halves(inp, seasons), seasons, figures_dir / "half_team_stats.png", source),
        charts.player_tov_by_half(rotation, inp.focus_player, inp.season,
                                  figures_dir / "player_tov_by_half.png", source),
        charts.lineup_net_2h(inp.lineups, inp.season, inp.focus_player, figures_dir / "lineup_net_2h.png", source),
        charts.league_rank_dots(inp.ranks, inp.season, figures_dir / "league_rank_dots.png", source),
        charts.attribution_waterfall(attribution(inp), inp.season, inp.n_games,
                                     figures_dir / "attribution.png", source),
        charts.season_factors(by_id["SEASONS"].tables["season_comparison"], seasons,
                              figures_dir / "season_factors.png", source),
    ]


# ------------------------------------------------------------- markdown

def _digits(values) -> int | None:
    """Decimals for a group of numbers: None = whole numbers, 3 for rates (all below 1.5), else 1."""
    nums = [float(v) for v in values if isinstance(v, (int, float)) and not isinstance(v, bool) and pd.notna(v)]
    if not nums or all(v.is_integer() for v in nums):
        return None
    return 3 if all(abs(v) < 1.5 for v in nums) else 1


def _fmt(v, digits: int | None) -> str:
    if isinstance(v, pd.Timestamp):
        return v.strftime("%Y-%m-%d")
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return f"{v:.0f}" if digits is None else f"{v:.{digits}f}"
    return str(v).replace("|", "\\|")


def md_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    """Markdown table. "metric" tables (one stat per row) are formatted row by row, others column by column."""
    d = (df[columns] if columns else df).reset_index(drop=True)
    if "split" in d.columns:
        d = d[d["split"].astype(str) != "OT"]
    by_row = d.columns[0] == "metric"
    cells = d.astype(object).to_numpy()
    if by_row:
        digits = [[_digits(row[1:])] * len(row) for row in cells]
    else:
        col_digits = [_digits(cells[:, j]) for j in range(cells.shape[1])]
        digits = [col_digits for _ in cells]
    head = "| " + " | ".join(str(c) for c in d.columns) + " |"
    sep = "|" + "|".join("---" for _ in d.columns) + "|"
    rows = ["| " + " | ".join(_fmt(v, dg) for v, dg in zip(row, drow)) + " |" for row, drow in zip(cells, digits)]
    return "\n".join([head, sep, *rows])


def answer(r: HypothesisResult) -> str:
    """A short answer for the summary table, from the key numbers (falls back to the verdict's opening)."""
    k = r.key
    word = lambda s: ("Drives" if s >= DRIVES else "Contributes" if s >= CONTRIBUTES else "Minor")  # noqa: E731
    if r.id == "H2":
        return f"{word(k['tov_share_of_swing'])} ({k['tov_share_of_swing']:.0%} of the swing); costlier, not more frequent"
    if r.id == "H3":
        return f"{r.verdict.split(':')[0]} ({k['share_of_swing']:.0%} of the swing)"
    if r.id == "H4":
        return f"{word(k['ft_share'])} ({k['ft_share']:.0%} of the swing)"
    if r.id == "H5":
        return r.verdict.split(". ")[0].rstrip(".")
    if r.id == "ATTR":
        return (f"Largest factor: {charts.SHORT_LABELS[k['top_factor']]} ({k['top_points']:+.1f} of "
                f"{k['swing']:+.0f})")
    if r.id == "SEASONS":
        return f"New: {_labels(k['new'], False)}. Chronic: {_labels(k['chronic'], False)}"
    head = r.verdict.split(":")[0]
    return head if len(head) < 20 else r.verdict.split(". ")[0].rstrip(".")


def _labels(factors: list[str], capitalize: bool = True) -> str:
    names = [charts.SHORT_LABELS[f] for f in factors]
    names = [n if n.startswith("Clippers") else n.lower() for n in names]
    if not names:
        return "none"
    text = " and ".join(names) if len(names) <= 2 else ", ".join(names[:-1]) + " and " + names[-1]
    return text[0].upper() + text[1:] if capitalize else text


def _share_label(share: float) -> str:
    return "drives" if share >= DRIVES else "contributes to" if share >= CONTRIBUTES else "is a minor part of"


def executive_summary(by_id: dict[str, HypothesisResult], inp: Inputs, threshold: int) -> list[str]:
    h1, h2, h3, h4, h5 = (by_id[k].key for k in ("H1", "H2", "H3", "H4", "H5"))
    seasons = by_id["SEASONS"]
    comp = seasons.tables["season_comparison"].set_index("season")
    cur, prior = comp.loc[inp.season], comp.loc[inp.comparison_season]
    att = attribution(inp).set_index("factor")
    swing = att.attrs["swing"]
    top = att.index[0]
    n = inp.n_games
    out = []

    leads_now = int(cur["blown_leads"] + cur["held_leads"])
    leads_then = int(prior["blown_leads"] + prior["held_leads"])
    out.append(
        f"Through {n} games of {inp.season}, the Clippers are {h1['net_h1']:+.1f} per 100 possessions in first "
        f"halves and {h1['net_h2']:+.1f} in second halves, a drop that ranks {ordinal(h1['net_change_rank'])} of "
        f"30, and they have lost {int(cur['blown_leads'])} of the {leads_now} games in which they led by "
        f"{threshold}+ early (last season: {int(prior['blown_leads'])} of {leads_then})."
    )
    if top == "shooting":
        rank = h5["efg_change_rank"]
        where = "the largest drop in the league" if rank == 30 else f"the {ordinal(31 - rank)}-largest drop"
        out.append(
            f"The biggest factor is their own shooting: eFG% falls from {h1['efg_h1']:.3f} to {h1['efg_h2']:.3f} "
            f"after halftime, {where}. That alone is worth {att.loc['shooting', 'points']:+.0f} points against a "
            f"total swing of {swing:+.0f}, while opponents actually shoot worse after the break."
            if h5["opp_efg_change"] < 0 else
            f"The biggest factor is their own shooting: eFG% falls from {h1['efg_h1']:.3f} to {h1['efg_h2']:.3f} "
            f"after halftime, {where}, worth {att.loc['shooting', 'points']:+.0f} of the {swing:+.0f}-point swing."
        )
    else:
        out.append(f"The biggest factor is {att.loc[top, 'label']}, worth {att.loc[top, 'points']:+.0f} of the "
                   f"{swing:+.0f}-point swing.")

    onoff = by_id["H3"].tables["h3_onoff_h2"].set_index("metric")["on_minus_off"]
    defense_gap = abs(onoff.get("Defensive rating", 0)) > abs(onoff.get("Offensive rating", 0))
    if h3["share_of_swing"] >= DRIVES:
        out.append(
            f"The starting hypothesis holds up: {inp.focus_player}'s excess second-half turnovers are worth about "
            f"{h3['excess_pts']:.0f} points, {h3['share_of_swing']:.0%} of the swing."
        )
    else:
        out.append(
            f"It is not mainly {inp.focus_player}. He is the team's biggest single source of second-half "
            f"turnovers ({h3['tov_share_h2']:.0%} of them on {h3['play_share_h2']:.0%} of the team's plays), but "
            f"even against the most generous baseline his excess turnovers are worth about {h3['excess_pts']:.0f} "
            f"points, {h3['share_of_swing']:.0%} of the swing"
            + ("; the team's large second-half on/off gap with him is mostly on defense." if defense_gap else ".")
        )
    tov_flat = abs(h2["tov_h2"] - h2["tov_h1"]) <= 0.1 * h2["tov_h1"]
    out.append(
        (f"As a team they turn it over about as often after halftime ({h2['tov_h1']:.0f} vs {h2['tov_h2']:.0f}); "
         if tov_flat else f"Team turnovers go from {h2['tov_h1']:.0f} to {h2['tov_h2']:.0f} after halftime; ")
        + f"what changes is the cost ({h2['opp_pts_off_tov_h1']} vs {h2['opp_pts_off_tov_h2']} opponent points off "
        f"them), and the turnover factor {_share_label(h2['tov_share_of_swing'])} the swing "
        f"({h2['tov_share_of_swing']:.0%})."
    )
    chronic, new = seasons.key["chronic"], seasons.key["new"]
    late = h4["rise_ex_late_and_outlier"] < 0.5 * h4["rise"]
    out.append(
        f"Fouling {_share_label(h4['ft_share'])} it ({h4['ft_share']:.0%}) and is "
        + ("a chronic habit, not a new one" if "free_throws" in chronic else
           "new this season" if "free_throws" in new else "about the same as last season")
        + ("; most of the second-half jump in opponent free throws comes from late fouling by a team already "
           "behind and one outlier game." if late else ".")
    )
    out.append(f"Compared with last season, what is new is {_labels(new, False)}; what is chronic is "
               f"{_labels(chronic, False)}.")
    out.append(
        f"{n} games is a small sample: the 90% interval for the net-rating drop ({h1['boot_lo']:+.1f} to "
        f"{h1['boot_hi']:+.1f}) {'includes' if h1['boot_hi'] >= 0 else 'excludes'} zero, so treat these as leads "
        f"to keep checking, not settled rates."
    )
    return out


def methodology(inp: Inputs, meta: dict, tov_agreement: float) -> list[str]:
    return [
        f"**Data.** Every {inp.season} and {inp.comparison_season} Clippers regular-season game from stats.nba.com "
        f"via `nba_api`: game logs, box scores filtered to each half (`BoxScoreTraditionalV3`, `AdvancedV3`, "
        f"`MiscV3`), play-by-play, on/off and lineup dashboards by period, and league-wide team dashboards by period "
        f"for ranks. {inp.season} data through {meta['last_game']}; analyzed on {meta['analyzed_on']}.",
        "**Halves.** Period filters on the V3 box-score endpoints are only honoured with `range_type=1`; without "
        "it they silently return the full game. Every pull checks that H1 + H2 (+ OT) equals the full-game box "
        "score, that play-by-play final scores match, and that the halftime margin from play-by-play matches "
        "the H1 box score. Overtime is kept separate and excluded from 1H vs 2H comparisons.",
        "**Rates.** All rates (eFG%, TOV%, FT rate, ratings) are recomputed from summed counts, never averaged "
        "across games. Possessions follow the NBA's box-score estimate.",
        "**Attribution.** Points = 2 x eFG% x FGA + FTM for each side, and the change from first to second half "
        "is split exactly with the midpoint rule; shot volume is then split with FGA = possessions + OREB - TOV - "
        "free-throw trips. The factors sum exactly to the change in margin. What can't be placed is reported as "
        "\"unattributed\" rather than folded into another factor. It describes where the points went, not why.",
        f"**Garland.** His excess second-half turnovers are measured against whichever baseline gives the "
        f"hypothesis the most credit (his own first halves, his {inp.comparison_season} second halves), and "
        f"valued at what a second-half turnover costs: the Clippers' points per possession given up, plus how much "
        f"more opponents score off a turnover than off an average possession. A test triples his second-half "
        f"turnovers and checks that the verdict flips to \"Yes\".",
        f"**Thresholds** (set before reading the results): a factor *drives* the swing at {DRIVES:.0%} or more, "
        f"*contributes* at {CONTRIBUTES:.0%}-{DRIVES:.0%}, and is *minor* below {CONTRIBUTES:.0%}; a 1H to 2H "
        "change ranked 26th-30th is \"among the worst\"; an early lead is 8+ points in the first 30 minutes.",
        "**Uncertainty.** 90% intervals come from resampling whole games 4,000 times.",
    ], [
        f"**Sample size.** {inp.n_games} games is {2 * inp.n_games} halves. Several verdicts could flip with a "
        "few more games, which is why every number is rerun from scratch on each pull.",
        f"**Points off turnovers.** Totals come from the official per-half `BoxScoreMiscV3` counts. The live-ball "
        f"vs dead-ball split comes from play-by-play, and the play-by-play estimate matches the official count in "
        f"{tov_agreement:.0%} of game-halves.",
        "**Unattributed points.** This season's residual is mostly Clippers second-half shots that the "
        "possession identity can't place, most likely team offensive rebounds that the box score doesn't count.",
        "**On/off and lineups** are small samples dominated by who was on the floor in which games. "
        "They describe, they don't isolate.",
        f"**Garland's {inp.comparison_season}** combines his Cleveland and Clippers games. His Cleveland halves are "
        "his season dashboard minus his Clippers box scores, checked against his game log.",
    ]


def render(results: list[HypothesisResult], inp: Inputs, meta: dict, figures_rel: str = "figures",
           tov_agreement: float = float("nan")) -> str:
    by_id = {r.id: r for r in results}
    method, limits = methodology(inp, meta, tov_agreement)
    lines = [
        f"# Why the Clippers lose leads after halftime ({inp.season})",
        "",
        f"*{inp.n_games} games, data through {meta['last_game']}, compared with {inp.comparison_season}. "
        "Generated by `courtside report` from `data/processed/`. Don't edit by hand: rerun it.*",
        "",
        "## Executive summary",
        "",
        " ".join(executive_summary(by_id, inp, meta["lead_threshold"])),
        "",
        "**Charts to look at first:**",
        "",
        *[f"{i}. [`{name}`]({figures_rel}/{name}): {why}" for i, (name, why) in enumerate(FIRST_LOOK, 1)],
        "",
        "| | Question | Answer |",
        "|---|---|---|",
        *[f"| {r.id} | {r.title} | {answer(r)} |" for r in results],
        "",
    ]
    for r in results:
        lines += [f"## {r.id}. {r.title}", "", f"**Answer: {answer(r)}.**", "", r.verdict, ""]
        for fig in EXTRA_FIGURES.get(r.id, []) + [r.chart]:
            lines += [f"![{fig}]({figures_rel}/{fig})", ""]
        for name, cols in SECTION_TABLES.get(r.id, []):
            if name in r.tables:
                lines += [f"<sub>`{name}.csv`</sub>", "", md_table(r.tables[name], cols), ""]
    lines += ["## Methodology", "", *[f"- {m}" for m in method], "",
              "## Limitations", "", *[f"- {m}" for m in limits], "",
              "Full result tables: `data/processed/analysis/`. Key numbers and verdicts: `reports/analysis.json`.",
              ""]
    return "\n".join(lines)


def tov_estimate_agreement(inp: Inputs) -> float:
    h = inp.halves()
    return float((h["opp_pts_off_tov"] == h["opp_pts_off_tov_est"]).mean())


def write_report(results: list[HypothesisResult], inp: Inputs, meta: dict, margins: pd.DataFrame,
                 reports_dir: Path) -> Path:
    last = pd.to_datetime(inp.game_leads.loc[inp.game_leads["season"] == inp.season, "game_date"]).max()
    meta = {**meta, "last_game": last.strftime("%Y-%m-%d")}
    source = (f"Source: stats.nba.com via nba_api. {inp.season} through {meta['last_game']}. "
              "github.com/JayJaden06/clippers-courtside")
    draw_figures(inp, results, margins, reports_dir / "figures", meta["lead_threshold"], source)
    path = reports_dir / "findings.md"
    text = render(results, inp, meta, tov_agreement=tov_estimate_agreement(inp))
    path.write_text(text, encoding="utf-8")
    return path
