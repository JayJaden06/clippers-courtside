"""Score-margin time series and blown-lead classification.

``margin_series`` is one row per scoring change (plus tip-off at 0-0):
seconds elapsed -> Clippers margin. ``game_leads`` is one row per game:

- ``max_lead`` / ``max_lead_elapsed``: largest Clippers lead and when it was first reached
- ``max_lead_window``: largest lead in the first ``lead_window_minutes`` (default 30)
- ``halftime_margin``, ``q4_start_margin``, ``final_margin``
- ``max_deficit``: largest Clippers deficit (positive number)
- ``minutes_led`` / ``minutes_trailed``: game time spent ahead / behind
- ``largest_2h_run_allowed``: most opponent points in the second half (Q3 onwards,
  including OT) scored without a Clippers point in between

Classification at threshold T (``lead_class_<T>``):

- ``blown_lead``: led by >= T at some point in the first 30 minutes and lost
- ``held_lead``: same lead condition and won
- ``never_led_big``: otherwise

Halftime falls inside the 30-minute window, so "or at halftime" in the spec
adds nothing beyond the window test. A lead exactly equal to T counts.

The classification also catches leads that were gone before halftime (e.g.
up 11 in Q1, down 6 at the half). ``led_at_half`` separates genuine
second-half collapses from those, so the hypothesis tests can use either.
"""

from __future__ import annotations

import pandas as pd



def margin_series(events: pd.DataFrame) -> pd.DataFrame:
    """Rows where the score changed, plus a 0-0 row at tip-off for each game."""
    cols = ["season", "game_id", "elapsed", "period", "team_score", "opp_score", "margin"]
    changed = events[cols].copy()
    key = changed["team_score"].astype(str) + "-" + changed["opp_score"].astype(str)
    prev = key.groupby(changed["game_id"]).shift()
    changed = changed[key != prev]
    starts = changed.groupby("game_id", as_index=False).first()
    starts[["elapsed", "team_score", "opp_score", "margin"]] = 0
    starts["period"] = 1
    out = pd.concat([starts[cols], changed[cols]])
    # Drop the duplicate 0-0 row (the first event of each game is 0-0 too).
    out = out.drop_duplicates(["game_id", "team_score", "opp_score"], keep="first")
    return out.sort_values(["game_id", "elapsed"], kind="stable").reset_index(drop=True)


def _after_period(series: pd.DataFrame, period: int, col: str = "margin") -> int:
    """Value at the end of ``period``: the last row from that period or earlier.

    Uses the period, not the clock: free throws for a technical at the start
    of Q3 carry the same elapsed time (24:00) as the end of Q2, so a time-based
    cut would count them in the first half.
    """
    return int(series.loc[series["period"] <= period, col].iloc[-1])


def _time_weighted(series: pd.DataFrame, end: float) -> tuple[float, float]:
    """Minutes ahead and minutes behind, holding each margin until the next change."""
    t = series["elapsed"].tolist() + [end]
    m = series["margin"].tolist()
    ahead = sum(t[i + 1] - t[i] for i in range(len(m)) if m[i] > 0)
    behind = sum(t[i + 1] - t[i] for i in range(len(m)) if m[i] < 0)
    return ahead / 60, behind / 60


def _largest_run_allowed(series: pd.DataFrame, after_period: int) -> int:
    """Most opponent points scored after ``after_period`` with no Clippers points in between."""
    s = series[series["period"] > after_period]
    prev_team = _after_period(series, after_period, "team_score")
    prev_opp = _after_period(series, after_period, "opp_score")
    best = run = 0
    for team, opp in zip(s["team_score"], s["opp_score"]):
        if team > prev_team:
            run = 0
        run += opp - prev_opp
        best = max(best, run)
        prev_team, prev_opp = team, opp
    return best


def classify(max_lead_window: int, result: str, threshold: int) -> str:
    if max_lead_window >= threshold:
        return "blown_lead" if result == "L" else "held_lead"
    return "never_led_big"


def game_leads(
    series: pd.DataFrame,
    games: pd.DataFrame,
    threshold: int = 8,
    window_minutes: int = 30,
    sensitivity: list[int] | None = None,
) -> pd.DataFrame:
    """One row per game with lead metrics and blown/held classification."""
    meta = games.set_index("game_id")
    rows = []
    for game_id, s in series.groupby("game_id", sort=False):
        g = meta.loc[game_id]
        end = 48 * 60 + 5 * 60 * int(g["overtime_periods"])
        window = s[s["elapsed"] <= window_minutes * 60]
        leading = s[s["margin"] == s["margin"].max()]
        minutes_led, minutes_trailed = _time_weighted(s, end)
        rows.append({
            "season": g["season"],
            "game_id": game_id,
            "game_date": g["game_date"],
            "opponent_abbreviation": g["opponent_abbreviation"],
            "is_home": g["is_home"],
            "result": g["result"],
            "overtime_periods": int(g["overtime_periods"]),
            "max_lead": max(int(s["margin"].max()), 0),
            "max_lead_elapsed": float(leading["elapsed"].iloc[0]) if s["margin"].max() > 0 else None,
            "max_lead_window": max(int(window["margin"].max()), 0),
            "max_deficit": max(-int(s["margin"].min()), 0),
            "halftime_margin": _after_period(s, 2),
            "led_at_half": _after_period(s, 2) > 0,
            "q4_start_margin": _after_period(s, 3),
            "final_margin": int(s["margin"].iloc[-1]),
            "minutes_led": round(minutes_led, 2),
            "minutes_trailed": round(minutes_trailed, 2),
            "largest_2h_run_allowed": _largest_run_allowed(s, after_period=2),
        })
    out = pd.DataFrame(rows)
    for t in sorted(set((sensitivity or []) + [threshold])):
        out[f"lead_class_{t}"] = [classify(m, r, t) for m, r in zip(out["max_lead_window"], out["result"])]
    out["lead_class"] = out[f"lead_class_{threshold}"]
    return out.sort_values(["season", "game_date"]).reset_index(drop=True)


def write_leads(series: pd.DataFrame, leads: pd.DataFrame, out_dir) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    series.to_parquet(out_dir / "margin_series.parquet", index=False)
    leads.to_parquet(out_dir / "game_leads.parquet", index=False)
    leads.to_csv(out_dir / "game_leads.csv", index=False)


def check_halftime_margins(leads: pd.DataFrame, game_halves: pd.DataFrame) -> None:
    """Play-by-play halftime margin must equal the H1 box-score margin for every game."""
    h1 = game_halves[game_halves["split"].astype(str) == "H1"].set_index("game_id")["margin"]
    pbp = leads.set_index("game_id")["halftime_margin"]
    bad = pbp[pbp != h1.reindex(pbp.index)]
    if not bad.empty:
        raise ValueError(f"halftime margin from play-by-play != H1 box score for {list(bad.index)}")
