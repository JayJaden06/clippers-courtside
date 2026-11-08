"""Plain-English layer for the dashboard: names, explanations and generated sentences.

Everything a non-basketball reader sees is written here, from the same numbers
the report uses. Sentences are templates filled from data (with the report's
thresholds), so they change with the data rather than being written to an
expected answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from leadcollapse.analysis.decompose import FACTORS
from leadcollapse.analysis.hypotheses import CONTRIBUTES, DRIVES

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
    "turnovers": ("Turnovers", "Possessions lost to turnovers instead of ending in a shot."),
    "rebounding": ("Second chances", "Offensive rebounds: extra shots for the Clippers (or given up to "
                   "opponents)."),
    "free_throws": ("Free throws", "Points from the free-throw line, for and against."),
    "pace": ("Number of possessions", "Fewer or more trips down the floor."),
    "residual": ("Can't tell", "Points the bookkeeping can't place in any one bucket."),
}

LEAD_TAG = {"blown_lead": "Blown lead", "held_lead": "Held the lead"}

QUESTIONS = {
    "H1": "Do they really fall apart after halftime?",
    "H2": "Is it turnovers?",
    "H3": "Is it Darius Garland?",
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
        "H3", QUESTIONS["H3"], ans, tone,
        f"He's their biggest turnover problem after halftime: {k3['tov_share_h2']:.0%} of the team's turnovers "
        f"while handling {k3['play_share_h2']:.0%} of its plays. But even counting generously, his extra "
        f"turnovers explain about {k3['share_of_swing']:.0%} of the drop"
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
    }


def game_story(leads_row: pd.Series, halves: pd.DataFrame, attribution: pd.DataFrame) -> str:
    """A short paragraph about one game, from its halves and its decomposition."""
    h = halves.assign(split=halves["split"].astype(str)).set_index("split")
    hm, final, mx = int(leads_row["halftime_margin"]), int(leads_row["final_margin"]), int(leads_row["max_lead"])
    opp = leads_row["opponent_abbreviation"]
    lead = f"They led by as many as {mx}" if mx > 0 else "They never led"
    at_half = "tied at halftime" if hm == 0 else f"{'up' if hm > 0 else 'down'} {abs(hm)} at halftime"
    pts2, opp2 = int(h.loc["H2", "pts"]), int(h.loc["H2", "opp_pts"])
    second = (f"outscored {opp2}-{pts2}" if pts2 < opp2 else f"won the second half {pts2}-{opp2}" if pts2 > opp2
              else f"even in the second half at {pts2}-{opp2}")
    ot = int(leads_row["overtime_periods"])
    result = "won" if leads_row["result"] == "W" else "lost"
    out = (f"{lead} against {opp} and were {at_half}. After the break they were {second}, and {result} by "
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
