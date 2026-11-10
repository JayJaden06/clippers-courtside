"""Exact decomposition of the first-half -> second-half margin swing, and a game bootstrap.

Why not "points off turnovers + FT points + eFG swing + rebound swing"? Those
overlap: a layup after a steal counts as points off a turnover *and* in the
opponent's eFG%, so the pieces would not add up to the actual swing. This
decomposition is additive by construction.

For each side, points = 2 * eFG * FGA + FTM. With the midpoint rule
``d(xy) = dx * mean(y) + mean(x) * dy`` (exact for a product of two terms):

    d(points) = 2 * d(eFG) * mean(FGA)  +  2 * mean(eFG) * d(FGA)  +  d(FTM)
                  shooting efficiency         shot volume             free throws

Shot volume is then split with the possession identity
``FGA = possessions + OREB - TOV - FT trips + e``:

    turnovers   -2 * mean(eFG) * d(TOV)            TOV includes team turnovers
    rebounding  +2 * mean(eFG) * d(OREB)
    FT trips    -2 * mean(eFG) * d(trips)          folded into the free-throw factor
    pace        +2 * mean(eFG) * d(possessions)    the API's possession count
    residual    +2 * mean(eFG) * d(e)

FT trips are counted from play-by-play (first shot of a 2- or 3-shot trip)
rather than the usual 0.44 * FTA shortcut, which drifts when the mix of
free throws changes (this season's opponents nearly double their FTs after
halftime). ``e`` is mostly team offensive rebounds and end-of-period
possessions that the identity does not see; it is reported, not hidden.

Margin factors are Clippers minus opponent. They sum exactly to the change in
margin. It is a description of where the points went, not a causal model.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

FACTORS = ["shooting", "opp_shooting", "turnovers", "rebounding", "free_throws", "pace", "residual"]
FACTOR_LABELS = {
    "shooting": "Clippers shooting (eFG%)",
    "opp_shooting": "Opponent shooting (eFG%)",
    "turnovers": "Turnovers (shots lost / gained)",
    "rebounding": "Offensive rebounding",
    "free_throws": "Free throws (made FTs, net of trips)",
    "pace": "Pace (possessions)",
    "residual": "Unattributed (possession-accounting residual)",
}
SIDE_COUNTS = ["pts", "fgm", "fga", "fg3m", "ftm", "fta", "oreb", "tov", "possessions"]


def _side(half: pd.Series, prefix: str) -> dict[str, float]:
    s = {c: float(half[f"{prefix}{c}"]) for c in SIDE_COUNTS}
    s["tov"] += float(half.get(f"{prefix}tov_team", 0.0))  # possessions also end on team turnovers
    s["efg"] = (s["fgm"] + 0.5 * s["fg3m"]) / s["fga"]
    trips_col = f"{prefix}ft_trips"
    s["trips"] = float(half[trips_col]) if trips_col in half.index else 0.44 * s["fta"]
    s["identity_gap"] = s["fga"] - (s["possessions"] + s["oreb"] - s["tov"] - s["trips"])
    return s


def _side_factors(a: dict, b: dict) -> dict[str, float]:
    """Contribution of each factor to one side's points change from half ``a`` to half ``b``."""
    mean_efg = (a["efg"] + b["efg"]) / 2
    d = {k: b[k] - a[k] for k in a}
    return {
        "shooting": 2 * d["efg"] * (a["fga"] + b["fga"]) / 2,
        "turnovers": -2 * mean_efg * d["tov"],
        "rebounding": 2 * mean_efg * d["oreb"],
        "free_throws": d["ftm"] - 2 * mean_efg * d["trips"],
        "pace": 2 * mean_efg * d["possessions"],
        "residual": 2 * mean_efg * d["identity_gap"],
    }


def decompose(h1: pd.Series, h2: pd.Series) -> dict[str, float]:
    """Factors of the change in margin from ``h1`` to ``h2`` (summed counts for one or many games)."""
    mine = _side_factors(_side(h1, ""), _side(h2, ""))
    theirs = _side_factors(_side(h1, "opp_"), _side(h2, "opp_"))
    out = {
        "shooting": mine["shooting"],
        "opp_shooting": -theirs["shooting"],
        **{k: mine[k] - theirs[k] for k in ("turnovers", "rebounding", "free_throws", "pace", "residual")},
    }
    out["total"] = sum(out[f] for f in FACTORS)
    out["margin_change"] = (h2["pts"] - h2["opp_pts"]) - (h1["pts"] - h1["opp_pts"])
    return out


def half_totals(game_halves: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """Sum every numeric count column over games for H1 and H2 (optionally within groups)."""
    gh = game_halves[game_halves["split"].astype(str).isin(["H1", "H2"])].assign(
        split=lambda d: d["split"].astype(str)
    )
    numeric = [c for c in gh.select_dtypes("number").columns if not c.endswith(("_pct", "_rate", "rtg", "pace"))
               and c not in ("ortg", "drtg", "net_rtg", "margin")]
    return gh.groupby((by or []) + ["split"])[numeric].sum()


def decompose_frame(game_halves: pd.DataFrame, by: list[str] | None = None) -> pd.DataFrame:
    """One decomposition per group (e.g. per season, or per game)."""
    totals = half_totals(game_halves, by)
    rows = []
    groups = totals.index.droplevel("split").unique() if by else [None]
    for g in groups:
        t = totals.loc[g] if g is not None else totals
        r = decompose(t.loc["H1"], t.loc["H2"])
        if by:
            r |= dict(zip(by, g if isinstance(g, tuple) else (g,)))
        rows.append(r)
    out = pd.DataFrame(rows)
    return out[(by or []) + FACTORS + ["total", "margin_change"]]


def bootstrap(
    game_halves: pd.DataFrame,
    stat: Callable[[pd.Series, pd.Series], float],
    n: int = 4000,
    seed: int = 0,
    level: float = 0.90,
) -> tuple[float, float, float]:
    """Point estimate and ``level`` percentile interval of ``stat(h1_totals, h2_totals)``, resampling games.

    Whole games are resampled with replacement and their H1 and H2 counts
    summed, so every draw is a plausible "season" of the same length. With 8
    games the interval is wide by design: it shows how much the answer could
    move if a different handful of games had been played.
    """
    per_game = half_totals(game_halves, ["game_id"])
    games = per_game.index.get_level_values("game_id").unique()
    h1 = per_game.xs("H1", level="split").reindex(games)
    h2 = per_game.xs("H2", level="split").reindex(games)
    point = stat(h1.sum(), h2.sum())
    rng = np.random.default_rng(seed)
    a1, a2, cols = h1.to_numpy(), h2.to_numpy(), h1.columns
    draws = np.empty(n)
    for i in range(n):
        idx = rng.integers(0, len(games), len(games))
        draws[i] = stat(pd.Series(a1[idx].sum(axis=0), index=cols), pd.Series(a2[idx].sum(axis=0), index=cols))
    lo, hi = np.nanpercentile(draws, [(1 - level) / 2 * 100, (1 + level) / 2 * 100])
    return float(point), float(lo), float(hi)
