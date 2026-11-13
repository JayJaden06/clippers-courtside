"""Plain-English layer for the dashboard: names, explanations and generated sentences.

Everything a non-basketball reader sees is written here, from the same numbers
the report uses. Sentences are templates filled from data (with the report's
thresholds), so they change with the data rather than being written to an
expected answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from courtside.names import short
from courtside.analysis.decompose import FACTORS
from courtside.analysis.hypotheses import CONTRIBUTES, DRIVES, ordinal
from courtside.viz.whatnow import Verdict

GLOSSARY = {
    "Possession": "One trip down the floor: it ends with a shot that's made or rebounded by the defense, "
                  "a turnover, or free throws. Each team gets about 100 a game.",
    "Net rating": "Points scored minus points allowed per 100 possessions. +5 means a team outscores "
                  "opponents by about 5 points over a full game's worth of trips.",
    "eFG% (shooting accuracy)": "Field-goal percentage that gives 3-pointers 50% extra credit, because they're "
                                "worth more. League average is about 54%.",
    "Turnover": "Losing the ball without getting a shot off: a bad pass, a steal, stepping out of bounds, "
                "an offensive foul.",
    "Live-ball turnover": "A turnover where play keeps going, usually a steal. These are dangerous because the "
                          "other team can run the other way for an easy basket.",
    "TOV% (turnover rate)": "How often a player's plays end in a turnover. 'Plays' here means shots, trips to the "
                            "free-throw line and turnovers.",
    "Offensive rebound": "Grabbing your own team's missed shot, which gives you a second chance to score.",
    "Free-throw rate": "Free throws attempted per shot taken. A high number for opponents means the Clippers "
                       "are fouling a lot.",
    "Blown lead": "Here: the Clippers led by 8 or more points at some point in the first 30 minutes and still "
                  "lost.",
    "Halftime": "Games are four 12-minute quarters. Halftime comes after the 2nd quarter, at minute 24.",
}

# factor -> (plain name, what it means)
FACTOR_PLAIN = {
    "shooting": ("Clippers' shooting", "Making a lower share of their shots after halftime."),
    "opp_shooting": ("Opponents' shooting", "How well opponents shot. Positive = opponents shot worse, "
                     "which helped the Clippers."),
    "turnovers": ("Turnovers, Clippers vs opponents", "Possessions the Clippers lose to turnovers, minus the "
                  "ones opponents lose. Negative = the Clippers come out behind."),
    "rebounding": ("Second chances, Clippers vs opponents", "Extra shots from offensive rebounds: the Clippers' "
                   "minus the opponents'. Negative = opponents get more second chances."),
    "free_throws": ("Free throws, Clippers vs opponents", "Points from the free-throw line: the Clippers' minus "
                    "the opponents'. Negative = opponents gain more at the line."),
    "pace": ("Number of possessions", "Fewer or more trips down the floor."),
    "residual": ("Can't tell", "Points the bookkeeping can't place in any one bucket."),
}

LEAD_TAG = {"blown_lead": "Blown lead", "held_lead": "Held the lead"}

QUESTIONS = {
    "H1": "Do they really fall apart after halftime?",
    "H2": "Is it turnovers?",
    "H3": "Is it {focus}?",  # filled from the analysis meta
    "H4": "Is it fouling?",
    "H5": "Is it just missed shots?",
}


@dataclass
class Card:
    id: str
    question: str
    answer: str  # a few words
    tone: str  # "yes" | "partly" | "no"
    text: str  # plain-English explanation


def _tone_from_share(share: float) -> tuple[str, str]:
    if share >= DRIVES:
        return "Yes, a big part", "yes"
    if share >= CONTRIBUTES:
        return "Partly", "partly"
    return "Only a little", "no"


def answer_cards(analysis: dict) -> list[Card]:
    """One card per hypothesis, in plain English, from ``reports/analysis.json``."""
    res = {r["id"]: r for r in analysis.get("results", [])}
    if not all(k in res for k in QUESTIONS):
        return []
    k1, k2, k3, k4, k5 = (res[i]["key"] for i in ("H1", "H2", "H3", "H4", "H5"))
    focus = analysis.get("meta", {}).get("focus_player", "the point guard")
    chronic = res.get("SEASONS", {}).get("key", {}).get("chronic", [])
    cards = []

    worse_teams = 30 - k1["net_change_rank"]
    real = k1["net_change_rank"] >= 26
    sure = k1["boot_hi"] < 0
    cards.append(Card(
        "H1", QUESTIONS["H1"],
        "Yes" if real and sure else "Probably" if real else "Not clearly",
        "yes" if real and sure else "partly" if real else "no",
        f"Before halftime they're about {_even(k1['net_h1'])} ({k1['net_h1']:+.1f} per 100 possessions). After "
        f"halftime they're {k1['net_h2']:+.1f}. Only {worse_teams} other team{'s' if worse_teams != 1 else ''} "
        f"fall off more after the break."
        + (" With only a handful of games, some of that could still be bad luck." if not sure else ""),
    ))

    ans, tone = _tone_from_share(k2["tov_share_of_swing"])
    flat = abs(k2["tov_h2"] - k2["tov_h1"]) <= 0.1 * k2["tov_h1"]
    cards.append(Card(
        "H2", QUESTIONS["H2"], ans, tone,
        (f"They don't turn it over more after halftime ({k2['tov_h1']:.0f} vs {k2['tov_h2']:.0f}), "
         if flat else f"Turnovers go from {k2['tov_h1']:.0f} to {k2['tov_h2']:.0f} after halftime, ")
        + f"but each one hurts more: opponents scored {k2['opp_pts_off_tov_h2']} points off them after halftime "
        f"vs {k2['opp_pts_off_tov_h1']} before. Turnovers explain about {k2['tov_share_of_swing']:.0%} of the "
        f"second-half drop.",
    ))

    ans, tone = _tone_from_share(k3["share_of_swing"])
    cards.append(Card(
        "H3", QUESTIONS["H3"].format(focus=focus), ans, tone,
        f"After halftime he makes {k3['tov_share_h2']:.0%} of the team's turnovers while handling "
        f"{k3['play_share_h2']:.0%} of its plays"
        + (" (more than his share)" if k3["tov_share_h2"] > k3["play_share_h2"] else "")
        + f". Counting generously, his extra turnovers explain about {k3['share_of_swing']:.0%} of the drop"
        + (f", short of the {DRIVES:.0%} it would take to call him the main cause." if tone != "yes" else "."),
    ))

    ans, tone = _tone_from_share(k4["ft_share"])
    late = k4["rise_ex_late_and_outlier"] < 0.5 * k4["rise"]
    cards.append(Card(
        "H4", QUESTIONS["H4"], ans, tone,
        "Opponents shoot far more free throws after halftime"
        + (", but much of that is the Clippers fouling on purpose late in games they're already losing" if late
           else "")
        + f". Free throws explain about {k4['ft_share']:.0%} of the drop"
        + (", and it was the same story last season." if "free_throws" in chronic else "."),
    ))

    share5 = abs(k5["shooting_points"]) / abs(res["ATTR"]["key"]["swing"]) if res.get("ATTR") else 0
    ans, tone = _tone_from_share(share5)
    worst = k5["efg_change_rank"] == 30
    cards.append(Card(
        "H5", QUESTIONS["H5"], "Yes, this is the big one" if tone == "yes" else ans, tone,
        f"Their shooting accuracy drops from {k1['efg_h1']:.1%} before halftime to {k1['efg_h2']:.1%} after"
        + (", the biggest drop of any team in the league" if worst else "") + ". "
        + ("Opponents actually shoot worse after halftime, so it's not the other team getting hot."
           if k5["opp_efg_change"] < 0 else "Opponents shoot better after halftime too."),
    ))
    return cards


def _last(name: str) -> str:
    """'Darius Garland' -> 'Garland' (the focus player's name as the page uses it)."""
    return short(name) if name else "the point guard"


def _even(net: float) -> str:
    return "even" if abs(net) < 3 else "ahead" if net > 0 else "behind"


def headline(analysis: dict, comparison: pd.DataFrame, games: pd.DataFrame, season: str) -> dict:
    """Numbers for the hero row of the dashboard."""
    res = {r["id"]: r for r in analysis.get("results", [])}
    cur = comparison.set_index("season").loc[season]
    g = games[games["season"] == season]
    return {
        "wins": int((g["result"] == "W").sum()),
        "losses": int((g["result"] == "L").sum()),
        "h1_per_game": float(cur["margin_h1_per_game"]),
        "h2_per_game": float(cur["margin_h2_per_game"]),
        "blown": int(cur["blown_leads"]),
        "leads": int(cur["blown_leads"] + cur["held_leads"]),
        "efg_h1": float(cur["efg_h1"]),
        "efg_h2": float(cur["efg_h2"]),
        "garland_share": float(res["H3"]["key"]["share_of_swing"]) if "H3" in res else float("nan"),
        "swing": float(res["ATTR"]["key"]["swing"]) if "ATTR" in res else float("nan"),
        "games": int(cur["games"]),
        "focus": analysis.get("meta", {}).get("focus_player", "the point guard"),
        "lead_threshold": int(analysis.get("meta", {}).get("lead_threshold", 8)),
    }


def game_story(leads_row: pd.Series, halves: pd.DataFrame, attribution: pd.DataFrame) -> str:
    """A short paragraph about one game, from its halves and its decomposition."""
    h = halves.assign(split=halves["split"].astype(str)).set_index("split")
    hm, final, mx = int(leads_row["halftime_margin"]), int(leads_row["final_margin"]), int(leads_row["max_lead"])
    opp = leads_row["opponent_abbreviation"]
    lead = f"They led by as many as {mx}" if mx > 0 else "They never led"
    at_half = "tied at halftime" if hm == 0 else f"{'up' if hm > 0 else 'down'} {abs(hm)} at halftime"
    pts2, opp2 = int(h.loc["H2", "pts"]), int(h.loc["H2", "opp_pts"])
    second = (f"were outscored {opp2}-{pts2}" if pts2 < opp2 else f"won the second half {pts2}-{opp2}" if pts2 > opp2
              else f"played the second half even at {pts2}-{opp2}")
    ot = int(leads_row["overtime_periods"])
    result = "won" if leads_row["result"] == "W" else "lost"
    out = (f"{lead} against {opp} and were {at_half}. After the break they {second}, and {result} by "
           f"{abs(final)}{' in overtime' if ot else ''}.")
    a = attribution[attribution["factor"] != "residual"].set_index("factor")["points"]
    change = float(attribution["margin_change"].iloc[0])
    if change < 0:
        f = a.idxmin()
        out += f" Biggest second-half problem: {_lower(FACTOR_PLAIN[f][0])} ({a[f]:+.0f} points vs the first half)."
    elif change > 0:
        f = a.idxmax()
        out += f" Biggest second-half improvement: {_lower(FACTOR_PLAIN[f][0])} ({a[f]:+.0f} points)."
    return out


def _lower(name: str) -> str:
    return name if name.startswith("Clippers") else name[0].lower() + name[1:]


def turnover_markers(turnovers: pd.DataFrame, margins: pd.DataFrame, game_id: str,
                     focus_last_name: str) -> pd.DataFrame:
    """Clippers turnovers placed on the margin line, with hover text."""
    t = turnovers[(turnovers["game_id"] == game_id) & turnovers["is_team"]].sort_values("elapsed").copy()
    t["minute"] = t["elapsed"] / 60
    t["margin"] = t["margin_before"]
    t["who"] = t["player_name"].fillna("Team")
    t["is_focus"] = t["who"] == focus_last_name
    t["kind"] = t["live_ball"].map({True: "steal / live ball", False: "dead ball"})
    return t[["minute", "margin", "who", "sub_type", "kind", "pts_off", "is_focus", "period"]].reset_index(drop=True)


def replay_frames(margin: pd.DataFrame, step_seconds: int = 30) -> list[float]:
    """Frame end-times (in minutes) for the game replay animation."""
    end = float(margin["elapsed"].max()) if len(margin) else 48 * 60
    stops = list(range(step_seconds, int(end) + step_seconds, step_seconds))
    return [s / 60 for s in stops]


def factor_order() -> list[str]:
    return list(FACTORS)


# ------------------------------------------------------------ the full answer

NEW, SAME, BETTER, SIMILAR = "New this season", "Same as last season", "Better than last season", "Similar to last season"


@dataclass
class Reason:
    factor: str
    name: str
    per_game: float  # this season, points per game after halftime (negative = costs points)
    prev_per_game: float  # last season
    share: float  # share of this season's swing
    status: str
    detail: str


def _status(cur: float, prev: float) -> str:
    """Same rules as the report's season comparison: 1 point per game worse is new, -0.5 in both is chronic."""
    if cur - prev <= -1.0:
        return NEW
    if cur - prev >= 1.0:
        return BETTER
    if cur <= -0.5 and prev <= -0.5:
        return SAME
    return SIMILAR


def _detail(factor: str, h: pd.DataFrame, n_games: int) -> str:
    a, b = h.loc["H1"], h.loc["H2"]
    if factor == "shooting":
        return (f"Shooting accuracy falls from {a['efg_pct']:.1%} before halftime to {b['efg_pct']:.1%} after; "
                f"3-point shooting from {a['fg3_pct']:.1%} to {b['fg3_pct']:.1%}.")
    if factor == "opp_shooting":
        return f"Opponents' shooting accuracy goes from {a['opp_efg_pct']:.1%} to {b['opp_efg_pct']:.1%}."
    if factor == "turnovers":
        return (f"{a['tov_all']:.0f} turnovers before halftime and {b['tov_all']:.0f} after, but opponents scored "
                f"{a['opp_pts_off_tov']:.0f} points off them before and {b['opp_pts_off_tov']:.0f} after.")
    if factor == "rebounding":
        return (f"Opponents grab {a['opp_oreb_pct']:.0%} of their own misses before halftime and "
                f"{b['opp_oreb_pct']:.0%} after, so they get more second chances.")
    if factor == "free_throws":
        return (f"Free throws per shot before halftime → after: Clippers {a['ft_rate']:.2f} → {b['ft_rate']:.2f}, "
                f"opponents {a['opp_ft_rate']:.2f} → {b['opp_ft_rate']:.2f}.")
    if factor == "pace":
        return (f"{a['possessions'] / n_games:.0f} possessions per game before halftime, "
                f"{b['possessions'] / n_games:.0f} after.")
    return ""


def ranked_reasons(att_cur: pd.DataFrame, att_prev: pd.DataFrame, halves: pd.DataFrame) -> list[Reason]:
    """Every factor except the residual, worst first, with last season alongside.

    ``halves`` is this season's ``season_halves`` slice (index H1/H2).
    """
    cur, prev = att_cur.set_index("factor"), att_prev.set_index("factor")
    swing = float(cur["points"].sum())
    n_games = int(round(cur["points"].iloc[0] / cur["per_game"].iloc[0])) if cur["per_game"].iloc[0] else 1
    out = [
        Reason(f, FACTOR_PLAIN[f][0], float(cur.loc[f, "per_game"]), float(prev.loc[f, "per_game"]),
               float(cur.loc[f, "points"]) / swing if swing else float("nan"),
               _status(float(cur.loc[f, "per_game"]), float(prev.loc[f, "per_game"])), _detail(f, halves, n_games))
        for f in FACTORS if f != "residual"
    ]
    return sorted(out, key=lambda r: r.per_game)


@dataclass
class ScoreRow:
    stat: str
    prev_h1: str
    prev_h2: str
    cur_h1: str
    cur_h2: str
    verdict: str


SCORECARD = [
    # label, column, format, higher is better, per-game count, tolerance for "about the same"
    ("Score margin per game", "margin", "{:+.1f}", True, True, 1.0),
    ("Net rating (per 100 possessions)", "net_rtg", "{:+.1f}", True, False, 2.0),
    ("Shooting accuracy", "efg_pct", "{:.1%}", True, False, 0.01),
    ("3-point shooting", "fg3_pct", "{:.1%}", True, False, 0.015),
    ("Opponents' shooting accuracy", "opp_efg_pct", "{:.1%}", False, False, 0.01),
    ("Turnovers per game", "tov_all", "{:.1f}", False, True, 0.5),
    ("Opponent points off turnovers per game", "opp_pts_off_tov", "{:.1f}", False, True, 1.0),
    ("Second chances given up (opp. offensive rebound %)", "opp_oreb_pct", "{:.1%}", False, False, 0.01),
    ("Opponent free throws per shot", "opp_ft_rate", "{:.2f}", False, False, 0.02),
]


def scorecard(sh: pd.DataFrame, cur: str, prev: str) -> list[ScoreRow]:
    """1st vs 2nd half for both seasons, and whether this season's second-half drop-off is new."""
    rows = []
    for label, col, fmt, higher_better, per_game, tol in SCORECARD:
        vals = {}
        for season in (prev, cur):
            n = sh.loc[(season, "H1"), "games"] if per_game else 1
            vals[season] = (sh.loc[(season, "H1"), col] / n, sh.loc[(season, "H2"), col] / n)
        drop = {s: (v[1] - v[0]) * (1 if higher_better else -1) for s, v in vals.items()}  # + = better after half
        gap = drop[cur] - drop[prev]
        verdict = ("Worse than last season" if gap <= -tol else "Better than last season" if gap >= tol
                   else "About the same")
        rows.append(ScoreRow(label, fmt.format(vals[prev][0]), fmt.format(vals[prev][1]), fmt.format(vals[cur][0]),
                             fmt.format(vals[cur][1]), verdict))
    return rows


def what_if(h2_per_game: float, reasons: list[Reason], fixed: set[str], garland_pts_per_game: float = 0.0,
            fix_garland: bool = False) -> float:
    """Second-half margin per game if the chosen factors went back to last season's level.

    The decomposition is additive, so swapping one factor's per-game value for
    last season's is exact bookkeeping. It is not a forecast: it ignores how
    the factors interact.
    """
    out = h2_per_game
    for r in reasons:
        if r.factor in fixed and r.prev_per_game > r.per_game:
            out += r.prev_per_game - r.per_game
    if fix_garland and "turnovers" not in fixed:
        out += garland_pts_per_game
    return out


def sixty_seconds(reasons: list[Reason], results: dict, hl: dict, prev_season: str) -> list[str]:
    """Three or four one-line takeaways (HTML) for the top of the answer page."""
    if not reasons:
        return []
    out = []
    r0 = reasons[0]
    out.append(f"<b>The main reason is {_lower(r0.name)}.</b> It costs {-r0.per_game:.1f} points a game after "
               f"halftime" + (", and it's new this season." if r0.status == NEW else "."))
    if "H3" in results:
        share = results["H3"]["key"]["share_of_swing"]
        out.append(f"<b>{_last(hl.get('focus', ''))}'s turnovers are "
                   f"{'a main cause' if share >= DRIVES else 'part of it, not the main cause'}"
                   f"</b>: about {share:.0%} of the second-half drop.")
    new = [_lower(r.name) for r in reasons if r.status == NEW]
    same = [_lower(r.name) for r in reasons if r.status == SAME]
    if new or same:
        out.append(f"<b>New since {prev_season}:</b> {', '.join(new) or 'nothing'}. "
                   f"<b>Same as before:</b> {', '.join(same) or 'nothing'}.")
    if r0.prev_per_game > r0.per_game:
        fixed = what_if(hl["h2_per_game"], reasons, {r0.factor})
        out.append(f"<b>If {_lower(r0.name)} went back to last season's level</b>, their second halves would go from "
                   f"{hl['h2_per_game']:+.1f} to {fixed:+.1f} points a game.")
    return out


def worst_collapse(game_leads: pd.DataFrame, season: str) -> dict | None:
    """The blown lead with the biggest fall from the largest lead to the final score."""
    g = game_leads[(game_leads["season"] == season) & (game_leads["lead_class"] == "blown_lead")]
    if g.empty:
        return None
    row = g.assign(fall=g["max_lead"] - g["final_margin"]).sort_values(["fall", "max_lead"]).iloc[-1]
    date = pd.Timestamp(row["game_date"])
    where = "vs" if row["is_home"] else "at"
    return {"game_id": row["game_id"], "label": f"{date:%b} {date.day} {where} {row['opponent_abbreviation']}"}


# ------------------------------------------------------------- answer cards for each page
# Each card: a short headline, one big number, one line that says what the number is.

THEORY_TONE = {"yes": "bad", "partly": "warn", "no": "info"}


def _first_sentence(text: str) -> str:
    end = text.find(". ")
    return text if end < 0 else text[: end + 1]


def answer_page_cards(hl: dict, reasons: list[Reason], results: dict, prev_season: str) -> list[Verdict]:
    out = [
        Verdict("Fine before halftime, behind after it", f"{hl['h1_per_game']:+.1f} → {hl['h2_per_game']:+.1f}",
                "Average score margin per game: first halves, then second halves.",
                "bad" if hl["h2_per_game"] < hl["h1_per_game"] else "good", "reasons",
                ("diverge", (hl["h1_per_game"], hl["h2_per_game"]), ("1st half", "2nd half"))),
        Verdict("Early leads blown", f"{hl['blown']} of {hl['leads']}",
                f"Games where they led by {hl.get('lead_threshold', 8)} or more early on and still lost.", "bad" if hl["blown"] else "good",
                "changes", ("dots", hl["blown"], hl["leads"])),
        Verdict("Shots stop falling after halftime", f"{hl['efg_h1']:.0%} → {hl['efg_h2']:.0%}",
                "Shooting accuracy (threes count extra), first half then second half.",
                "bad" if hl["efg_h2"] < hl["efg_h1"] else "good", "reasons",
                ("pair", hl["efg_h1"], hl["efg_h2"], "1st half", "2nd half")),
    ]
    hurts = [r for r in reasons if r.per_game < 0]
    if hurts:
        top = hurts[0]
        out.append(Verdict(f"Biggest reason: {_lower(top.name)}", f"{top.per_game:+.1f}",
                           f"Points a game it costs them after halftime ({top.share:.0%} of the drop).", "bad",
                           "reasons", ("share", top.share, None)))
    if "H3" in results:
        share = results["H3"]["key"]["share_of_swing"]
        label = "the main cause" if share >= DRIVES else "part of it" if share >= CONTRIBUTES else "a small part"
        last = _last(hl.get("focus", ""))
        out.append(Verdict(f"{last}'s turnovers: {label}", f"{share:.0%}",
                           f"of the second-half drop comes from {last}'s extra turnovers. The line marks "
                           f"{DRIVES:.0%}, where it would count as the main cause.", "warn", "theories",
                           ("share", share, DRIVES)))
    new = [r.name for r in reasons if r.status == NEW]
    out.append(Verdict("New problems this season", str(len(new)),
                       (f"Not a problem in {prev_season}: " + ", ".join(_lower(n) for n in new) + ".") if new
                       else f"Everything that hurts now also hurt in {prev_season}.",
                       "warn" if new else "info", "changes"))
    return out


def theory_cards(cards: list[Card]) -> list[Verdict]:
    return [Verdict(c.question, c.answer, _first_sentence(c.text), THEORY_TONE.get(c.tone, "info"), "theories")
            for c in cards]


def game_cards(row: pd.Series, halves: pd.DataFrame, turnovers: pd.DataFrame, focus_last: str) -> list[Verdict]:
    """Cards for one game: halftime, final, the second-half swing, and turnovers after halftime."""
    h = halves.assign(split=halves["split"].astype(str)).set_index("split")
    h2 = float(h.loc["H2", "margin"]) if "H2" in h.index else float("nan")
    ht, fin = int(row["halftime_margin"]), int(row["final_margin"])
    t = turnovers[(turnovers["game_id"] == row.name) & turnovers["is_team"] & turnovers["period"].isin([3, 4])]
    mine = int((t["player_name"] == focus_last).sum())
    counts = t["player_name"].dropna().value_counts()
    if counts.empty:
        who = "No player turnovers after halftime."
    else:
        top = sorted(counts[counts == counts.iloc[0]].index)
        each = " each" if len(top) > 1 else ""
        names = top[0] if len(top) == 1 else ", ".join(top[:-1]) + " and " + top[-1]
        who = f"Most: {names} ({counts.iloc[0]}{each})."
        if focus_last not in top:
            who += f" {focus_last} had {mine}."
    return [
        Verdict("At halftime", f"{ht:+d}", "Clippers ahead (+) or behind (−) going into the break.",
                "good" if ht > 0 else "bad" if ht < 0 else "info", "game"),
        Verdict("Second half", f"{h2:+.0f}", "How much the Clippers won (+) or lost (−) the 3rd and 4th quarters by.",
                "good" if h2 > 0 else "bad" if h2 < 0 else "info", "game"),
        Verdict("Final", f"{fin:+d}", f"{'Won' if fin > 0 else 'Lost'} by {abs(fin)}" + (" in overtime." if
                                                                                      row["overtime_periods"] else "."),
                "good" if fin > 0 else "bad", "game"),
        Verdict("Turnovers after halftime", str(len(t)), who + f" Filled dots: {focus_last}'s.", "warn", "game",
                ("dots", mine, len(t))),
    ]


def player_cards(ph: pd.DataFrame, focus: str) -> list[Verdict]:
    """Who loses the ball most after halftime, who falls off most, and who is safest."""
    h2 = ph[ph["split"] == "H2"]
    if h2.empty:
        return []
    out = []
    most = h2.sort_values("tov", ascending=False).iloc[0]
    out.append(Verdict(f"Most turnovers after halftime: {short(most['player_name'])}", f"{most['tov']:.0f}",
                       f"{most['tov_share']:.0%} of the team's second-half turnovers, while handling "
                       f"{most['play_share']:.0%} of its plays.", "bad", "who",
                       ("pair", most["play_share"], most["tov_share"], "plays", "turnovers")))
    w = ph.pivot(index="player_name", columns="split", values="tov_pct")
    if {"H1", "H2"} <= set(w.columns):
        jump = (w["H2"] - w["H1"]).sort_values()
        name = jump.index[-1]
        out.append(Verdict(f"Sloppiest after halftime: {short(name)}",
                           f"{w.loc[name, 'H1']:.0%} → {w.loc[name, 'H2']:.0%}",
                           f"Share of {short(name)}'s plays that end in a turnover, first half then second.",
                           "warn", "sloppier", ("pair", w.loc[name, "H1"], w.loc[name, "H2"], "1st half", "2nd half")))
    regular = h2[h2["play_share"] >= 0.08]
    if len(regular):
        safe = regular.sort_values("tov_pct").iloc[0]
        last = short(safe["player_name"])
        out.append(Verdict(f"Safest with the ball: {last}", f"1 in {round(1 / safe['tov_pct'])}" if safe["tov_pct"]
                           else "0", f"of {last}'s second-half plays ends in a turnover (among regular ball-handlers).",
                           "good", "who",
                           ("dots", 1, min(20, round(1 / safe["tov_pct"]))) if safe["tov_pct"] else None))
    return out


def league_cards(ranks: pd.DataFrame) -> list[Verdict]:
    """``ranks``: the team's ranks by stat (rows), with the first-to-second-half change rank in the last column."""
    change = ranks.iloc[:, -1].drop(labels=["Pace (possessions)"], errors="ignore").astype(int)
    out = []
    if "Net rating" in change.index:
        overall = int(change["Net rating"])
        out.append(Verdict("How much they fall off after halftime", f"{ordinal(overall)} of 30",
                           "Rank among the 30 teams for the change from first half to second (1st = holds up best).",
                           "bad" if overall > 20 else "good" if overall <= 10 else "info", "teams",
                           ("rank", overall, 30)))
    rest = change.drop(labels=["Net rating"], errors="ignore")
    if len(rest):
        worst, best = rest.idxmax(), rest.idxmin()
        out.append(Verdict(f"Worst after halftime: {_lower(worst)}", f"{ordinal(int(rest[worst]))} of 30",
                           f"Only {30 - int(rest[worst])} teams fall off more on this after halftime.", "bad", "ranks",
                           ("rank", int(rest[worst]), 30)))
        out.append(Verdict(f"Holds up best: {_lower(best)}", f"{ordinal(int(rest[best]))} of 30",
                           "Their best change after halftime, compared with the other 29 teams.", "good", "ranks",
                           ("rank", int(rest[best]), 30)))
    return out


# ------------------------------------------------------------- The Wall

INTUIT_DOME_FIRST_SEASON = "2024-25"


def wall_check(ft: pd.DataFrame) -> Verdict | None:
    """Do opponents miss more free throws after halftime at Intuit Dome (The Wall)?

    Pools every Intuit Dome season in ``ft`` (from ``dashboard_data.opp_ft_by_half``). The Wall "could be
    helping" only if opponents' home second-half FT% is lower than their first-half FT% by more than two
    standard errors of the difference. Teams switch baskets at halftime, and the box score doesn't say which
    end a team shot at, so this can't isolate The Wall's half; it can only say whether a half-to-half drop
    exists at home at all.
    """
    home = ft[(ft["where"] == "home") & (ft["season"] >= INTUIT_DOME_FIRST_SEASON)]
    if home.empty:
        return None
    by = home.groupby("split")[["ftm", "fta"]].sum()
    if not {"H1", "H2"} <= set(by.index) or (by["fta"] == 0).any():
        return None
    p = by["ftm"] / by["fta"]
    se = ((p * (1 - p) / by["fta"]).sum()) ** 0.5
    diff = p["H2"] - p["H1"]
    helping = diff < -2 * se
    return Verdict(
        "Is it The Wall?", "Maybe" if helping else "No",
        f"At Intuit Dome, opponents make {p['H1']:.0%} of their free throws before halftime and {p['H2']:.0%} "
        "after, so they don't miss more in the second half." if not helping else
        f"At Intuit Dome, opponents make {p['H1']:.0%} of their free throws before halftime and only "
        f"{p['H2']:.0%} after.",
        "warn" if helping else "info", "wall")
