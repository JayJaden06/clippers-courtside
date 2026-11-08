"""Static figures for ``reports/findings.md`` (matplotlib, PNG).

One function per figure. Each takes tidy frames and an output path, so the
report and the tests can draw them without the CLI. Styling is shared:

- 1st half = blue, 2nd half = orange (validated as a CVD-safe pair);
- "helped after halftime" = blue, "hurt" = red, unassigned = gray;
- the Clippers are the one colored mark among 29 gray teams;
- thin marks, hairline grid, direct labels only where they carry the story.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from leadcollapse.analysis.decompose import FACTOR_LABELS  # noqa: E402
from leadcollapse.transform.league_context import METRICS  # noqa: E402

H1, H2 = "#2a78d6", "#eb6834"
HELPED, HURT, NEUTRAL = "#2a78d6", "#e34948", "#898781"
INK, INK_2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, SURFACE, FAINT = "#e1e0d9", "#c3c2b7", "#fcfcfb", "#d8d7d0"
DPI = 150

SHORT_LABELS = {
    "shooting": "Clippers shooting",
    "opp_shooting": "Opponent shooting",
    "turnovers": "Turnovers",
    "rebounding": "Offensive rebounding",
    "free_throws": "Free throws",
    "pace": "Pace",
    "residual": "Unattributed",
}
METRIC_LABELS = {
    "net_rtg": "Net rating",
    "efg_pct": "eFG%",
    "opp_efg_pct": "Opponent eFG%",
    "tov_pct": "Turnover %",
    "opp_ft_rate": "Opponent FT rate",
    "opp_oreb_pct": "Opponent OREB%",
    "pf_per100": "Fouls per 100",
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_2,
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.facecolor": SURFACE,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK_2,
    "ytick.labelcolor": INK_2,
    "legend.frameon": False,
    "figure.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def _finish(fig, path: Path, title: str, subtitle: str, source: str) -> Path:
    fig.suptitle(title, x=0.01, y=0.995, ha="left", va="top", fontsize=13, fontweight="bold", color=INK)
    fig.text(0.01, 0.945, subtitle, ha="left", va="top", fontsize=9.5, color=INK_2)
    fig.text(0.01, 0.005, source, ha="left", va="bottom", fontsize=7.5, color=MUTED)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def _pct(v: float, digits: int = 1) -> str:
    return f"{100 * v:.{digits}f}%"


# ------------------------------------------------------------- 1. margins

def margin_small_multiples(margins: pd.DataFrame, leads: pd.DataFrame, season: str, threshold: int,
                           path: Path, source: str) -> Path:
    """Score margin through every game; games where an early lead was blown are drawn in orange."""
    lg = leads[leads["season"] == season].sort_values("game_date").reset_index(drop=True)
    ms = margins[margins["season"] == season]
    n = len(lg)
    cols = min(4, max(n, 1))
    rows = int(np.ceil(n / cols)) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(11, 2.35 * rows + 1.0), sharey=True, squeeze=False)
    lim = max(10, ms["margin"].abs().max() + 3) if len(ms) else 10
    for ax in axes.flat[n:]:
        ax.set_visible(False)
    for ax, (_, g) in zip(axes.flat, lg.iterrows()):
        m = ms[ms["game_id"] == g["game_id"]].sort_values("elapsed")
        blown = g["lead_class"] == "blown_lead"
        color = H2 if blown else H1
        x, y = m["elapsed"].to_numpy() / 60, m["margin"].to_numpy()
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.axvline(24, color=MUTED, lw=0.6)
        ax.step(x, y, where="post", color=color, lw=1.6)
        ax.fill_between(x, y, 0, step="post", color=color, alpha=0.10, lw=0)
        end = max(48, x.max() if len(x) else 48)
        ax.set_xlim(0, end)
        ax.set_ylim(-lim, lim)
        ax.set_xticks([0, 12, 24, 36, 48])
        ax.grid(axis="x", visible=False)
        when = pd.Timestamp(g["game_date"]).strftime("%b %d").replace(" 0", " ")
        where = "vs" if g["is_home"] else "at"
        ot = f" ({g['overtime_periods']}OT)" if g["overtime_periods"] else ""
        ax.set_title(f"{when} {where} {g['opponent_abbreviation']}  {g['result']} {g['final_margin']:+d}{ot}",
                     fontsize=9)
        if blown:
            first30 = m[m["elapsed"] <= 30 * 60]
            peak = first30.loc[first30["margin"].idxmax()]
            ax.plot(peak["elapsed"] / 60, peak["margin"], "o", ms=5, color=color, mec=SURFACE, mew=1.5)
            ax.annotate(f"led by {int(peak['margin'])}", (peak["elapsed"] / 60, peak["margin"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8, color=INK_2)
    for ax in axes[:, 0]:
        ax.set_ylabel("Clippers margin")
    for ax in axes[-1, :]:
        ax.set_xlabel("Game minute")
    first = axes.flat[0]
    first.annotate("half", (24, -lim), xytext=(3, 3), textcoords="offset points", fontsize=7.5, color=MUTED)
    handles = [plt.Line2D([], [], color=H2, lw=1.6), plt.Line2D([], [], color=H1, lw=1.6)]
    fig.legend(handles, [f"Blown lead (led by {threshold}+ in the first 30 min, lost)", "Other games"],
               loc="upper right", bbox_to_anchor=(0.995, 0.955), ncol=2, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    blown_n = int((lg["lead_class"] == "blown_lead").sum())
    leads_n = int(lg["lead_class"].isin(["blown_lead", "held_lead"]).sum())
    return _finish(fig, path, f"{season}: the score margin through every game",
                   f"{blown_n} of {leads_n} games with an early {threshold}-point lead ended in a loss. "
                   "Vertical line = halftime.", source)


# ----------------------------------------------- 2. team stats by half

def half_team_stats(team_halves: pd.DataFrame, seasons: list[str], path: Path, source: str,
                    metrics: tuple[str, ...] = ("net_rtg", "efg_pct", "opp_efg_pct",
                                                "tov_pct", "opp_ft_rate", "opp_oreb_pct")) -> Path:
    """First vs second half for the team-level stats behind H2, H4 and H5, both seasons.

    ``team_halves`` has one row per season and split (H1/H2) and a column per metric.
    """
    t = team_halves[team_halves["split"].isin(["H1", "H2"])]
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.2))
    width = 0.36
    for ax, metric in zip(axes.flat, metrics):
        d = t.pivot(index="season", columns="split", values=metric).reindex(seasons)
        pct = metric not in ("net_rtg", "tov_pct", "pf_per100")
        xs = np.arange(len(seasons))
        for off, split, color in ((-width / 2 - 0.01, "H1", H1), (width / 2 + 0.01, "H2", H2)):
            vals = d[split].to_numpy()
            ax.bar(xs + off, vals, width, color=color, label="1st half" if split == "H1" else "2nd half")
            for x, v in zip(xs + off, vals):
                label = f"{v:.3f}" if pct else f"{v:+.1f}" if metric == "net_rtg" else f"{v:.1f}"
                ax.annotate(label, (x, v), xytext=(0, 3 if v >= 0 else -11), textcoords="offset points",
                            ha="center", fontsize=7.5, color=INK_2)
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.set_xticks(xs, seasons)
        ax.grid(axis="x", visible=False)
        better = METRICS.get(metric)
        hint = "" if better is None else "  (higher is better)" if better else "  (lower is better)"
        ax.set_title(METRIC_LABELS.get(metric, metric) + hint, fontsize=9.5)
        lo, hi = np.nanmin(d.to_numpy()), np.nanmax(d.to_numpy())
        if pct:
            ax.set_ylim(0, hi * 1.18)
        else:
            pad = (max(hi, 0) - min(lo, 0)) * 0.18 or 1
            ax.set_ylim(min(lo, 0) - pad, max(hi, 0) + pad)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (H1, H2)]
    fig.legend(handles, ["1st half", "2nd half"], loc="upper right", bbox_to_anchor=(0.995, 0.955), ncol=2,
               fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.9))
    return _finish(fig, path, "First half vs second half, this season and last",
                   "Season totals by half. Rates are computed from summed counts, not averaged across games.",
                   source)


# ------------------------------------------------ 3. player TOV% by half

def player_tov_by_half(rotation: pd.DataFrame, focus_player: str, season: str, path: Path, source: str) -> Path:
    """Dumbbell of each rotation player's TOV%, first half to second half."""
    w = rotation.pivot(index="player_name", columns="split", values="tov_pct").dropna()
    w = w.sort_values("H2")
    counts = rotation.pivot(index="player_name", columns="split", values="tov").reindex(w.index)
    fig, ax = plt.subplots(figsize=(9, 0.42 * len(w) + 2.2))
    ys = np.arange(len(w))
    for y, (name, r) in zip(ys, w.iterrows()):
        focus = name == focus_player
        ax.plot([r["H1"], r["H2"]], [y, y], color=INK_2 if focus else FAINT, lw=2 if focus else 1.4, zorder=1)
    ax.scatter(w["H1"], ys, s=46, color=H1, edgecolor=SURFACE, linewidth=1.5, zorder=2, label="1st half")
    ax.scatter(w["H2"], ys, s=46, color=H2, edgecolor=SURFACE, linewidth=1.5, zorder=3, label="2nd half")
    ax.set_yticks(ys, [f"{n}  ({counts.loc[n, 'H1']:.0f} / {counts.loc[n, 'H2']:.0f})" for n in w.index])
    for lbl in ax.get_yticklabels():
        if lbl.get_text().startswith(focus_player):
            lbl.set_fontweight("bold")
            lbl.set_color(INK)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("Turnovers per 100 plays (TOV%)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(w.max().max() * 1.3, 0.1))
    if focus_player in w.index:
        r = rotation[(rotation["player_name"] == focus_player) & (rotation["split"] == "H2")].iloc[0]
        y = list(w.index).index(focus_player)
        ax.annotate(f"{_pct(r['tov_share'], 0)} of team 2nd-half turnovers\non {_pct(r['play_share'], 0)} "
                    f"of team plays", (w.loc[focus_player].max(), y), xytext=(10, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK_2)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.86))
    return _finish(fig, path, f"{season}: turnover rate by half, rotation players",
                   "Plays = shots + 0.44 x free-throw attempts + turnovers. (1st / 2nd-half turnover counts.) "
                   "Sorted by second-half rate.", source)


# --------------------------------------------- 4. second-half lineups

def _short_lineup(name: str, focus_player: str) -> tuple[str, bool]:
    parts = [p.strip() for p in name.split(" - ")]
    last = focus_player.split(" ", 1)[-1]
    has = any(p.endswith(last) for p in parts)
    return " · ".join(p.split(" ", 1)[-1] for p in parts), has


def lineup_net_2h(lineups: pd.DataFrame, season: str, focus_player: str, path: Path, source: str,
                  top: int = 8) -> Path:
    """Net rating of the most-used second-half lineups (diverging: blue = outscored opponents)."""
    d = lineups[(lineups["season"] == season) & (lineups["split"] == "H2")]
    d = d.sort_values("min", ascending=False).head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(d) + 2.2))
    ys = np.arange(len(d))
    colors = [HELPED if v >= 0 else HURT for v in d["net_rtg"]]
    ax.barh(ys, d["net_rtg"], height=0.6, color=colors)
    labels, bold = zip(*(_short_lineup(n, focus_player) for n in d["group_name"])) if len(d) else ((), ())
    ax.set_yticks(ys, labels)
    for lbl, b in zip(ax.get_yticklabels(), bold):
        if b:
            lbl.set_fontweight("bold")
            lbl.set_color(INK)
    ax.axvline(0, color=AXIS, lw=0.8)
    lim = max(10, d["net_rtg"].abs().max() * 1.35) if len(d) else 10
    ax.set_xlim(-lim, lim)
    for y, (_, r) in zip(ys, d.iterrows()):
        ax.annotate(f"{r['net_rtg']:+.0f}  ({r['min']:.0f} min)", (r["net_rtg"], y),
                    xytext=(4 if r["net_rtg"] >= 0 else -4, 0), textcoords="offset points",
                    ha="left" if r["net_rtg"] >= 0 else "right", va="center", fontsize=7.5, color=INK_2)
    ax.set_xlabel("Second-half net rating (points per 100 possessions)")
    ax.grid(axis="y", visible=False)
    fig.tight_layout(rect=(0, 0.03, 1, 0.86))
    return _finish(fig, path, f"{season}: the most-used second-half lineups",
                   f"Bold = lineups with {focus_player}. Small samples: a few minutes can swing these by 30+.",
                   source)


# ------------------------------------------------- 5. league rank dots

def league_rank_dots(ranks: pd.DataFrame, season: str, path: Path, source: str,
                     metrics: tuple[str, ...] = ("net_rtg", "efg_pct", "opp_efg_pct", "tov_pct",
                                                 "opp_ft_rate", "opp_oreb_pct")) -> Path:
    """Change from first to second half, all 30 teams; the Clippers highlighted. Better is always right."""
    d = ranks[(ranks["season"] == season) & (ranks["split"] == "change")]
    fig, axes = plt.subplots(len(metrics), 1, figsize=(10, 0.95 * len(metrics) + 1.6))
    for ax, metric in zip(np.atleast_1d(axes), metrics):
        m = d[d["metric"] == metric]
        flip = METRICS.get(metric) is False
        x = -m["value"] if flip else m["value"]
        team = m["is_team"].to_numpy()
        ax.scatter(x[~team], np.zeros((~team).sum()), s=36, facecolor="none", edgecolor=MUTED, lw=1, zorder=2)
        ax.scatter(x[team], [0], s=80, color=H2, edgecolor=SURFACE, lw=1.5, zorder=3)
        ax.axvline(0, color=AXIS, lw=0.8)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.grid(axis="y", visible=False)
        span = x.abs().max() * 1.15 or 1
        ax.set_xlim(-span, span)
        ax.set_ylim(-1, 1)
        row = m[m["is_team"]].iloc[0]
        pct = metric in ("efg_pct", "opp_efg_pct", "opp_ft_rate", "opp_oreb_pct")
        fmt = _signed(3 if pct else 1)
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda v, _, fmt=fmt, flip=flip: fmt(-v if flip else v)))
        ax.set_title(f"{METRIC_LABELS[metric]} change: {fmt(row['value'])}, {_ordinal(int(row['rank']))} of 30",
                     fontsize=9, pad=2)
    axes[-1].set_xlabel("1st half to 2nd half change   (each axis oriented so better is to the right)")
    fig.tight_layout(rect=(0, 0.03, 1, 0.88), h_pad=0.6)
    return _finish(fig, path, f"{season}: how the Clippers change after halftime vs the league",
                   "Orange = Clippers, gray rings = the other 29 teams. Rank 1 = best change, 30 = worst.", source)


def _signed(digits: int):
    def fmt(v: float) -> str:
        return "0" if round(v, digits) == 0 else f"{v:+.{digits}f}"
    return fmt


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


# ------------------------------------------------- 6. attribution

def attribution_waterfall(att: pd.DataFrame, season: str, n_games: int, path: Path, source: str) -> Path:
    """Waterfall from 0 to the total 1H -> 2H margin change, one step per factor (largest first)."""
    a = att.set_index("factor")
    order = [f for f in att["factor"] if f != "residual"] + ["residual"]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    running = 0.0
    xs = np.arange(len(order) + 1)
    for x, f in zip(xs, order):
        v = float(a.loc[f, "points"])
        color = NEUTRAL if f == "residual" else HELPED if v >= 0 else HURT
        ax.bar(x, v, bottom=running, width=0.62, color=color)
        ax.annotate(f"{v:+.1f}", (x, running + v if v >= 0 else running + v), xytext=(0, 3 if v >= 0 else -11),
                    textcoords="offset points", ha="center", fontsize=8, color=INK_2)
        if x < len(order):
            ax.plot([x + 0.31, x + 0.69], [running + v, running + v], color=MUTED, lw=0.6)
        running += v
    total = att.attrs.get("swing", running)
    ax.bar(xs[-1], total, width=0.62, color=INK_2)
    ax.annotate(f"{total:+.0f}", (xs[-1], total), xytext=(0, 3 if total >= 0 else -11),
                textcoords="offset points", ha="center", fontsize=8.5, fontweight="bold", color=INK)
    ax.axhline(0, color=AXIS, lw=0.8)
    ax.set_xticks(xs, [SHORT_LABELS[f].replace(" ", "\n", 1) for f in order] + ["Total\nchange"], fontsize=8.5)
    ax.set_ylabel("Points (sum over all games)")
    ax.grid(axis="x", visible=False)
    lo = min(0, *np.cumsum([a.loc[f, "points"] for f in order]), total)
    hi = max(0, *np.cumsum([a.loc[f, "points"] for f in order]), total)
    ax.set_ylim(lo - (hi - lo) * 0.12, hi + (hi - lo) * 0.12)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (HURT, HELPED, NEUTRAL)]
    ax.legend(handles, ["Worse after halftime", "Better after halftime", "Unattributed"], loc="lower left",
              fontsize=8.5, ncol=3)
    fig.tight_layout(rect=(0, 0.03, 1, 0.88))
    return _finish(fig, path, f"{season}: where the second-half points went",
                   f"Change in Clippers margin, 2nd half minus 1st half, over {n_games} games. "
                   "The steps add up exactly to the total.", source)


def season_factors(comparison: pd.DataFrame, seasons: list[str], path: Path, source: str) -> Path:
    """Per-game value of each factor, this season vs last. Below zero = costs points after halftime."""
    factors = [f for f in FACTOR_LABELS if f != "residual"]
    c = comparison.set_index("season").reindex(seasons)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ys = np.arange(len(factors))[::-1]
    for y, f in zip(ys, factors):
        vals = c[f"{f}_per_game"].to_numpy()
        ax.plot(vals, [y, y], color=FAINT, lw=1.6, zorder=1)
    styles = ((seasons[1], MUTED, "o", "none"), (seasons[0], INK, "o", INK))
    for season, color, marker, face in styles:
        vals = [c.loc[season, f"{f}_per_game"] for f in factors]
        ax.scatter(vals, ys, s=52, marker=marker, facecolor=face, edgecolor=color, lw=1.5, zorder=3,
                   label=f"{season} ({int(c.loc[season, 'games'])} games)")
    cur = [c.loc[seasons[0], f"{f}_per_game"] for f in factors]
    for y, v in zip(ys, cur):
        ax.annotate(f"{v:+.1f}", (v, y), xytext=(0, 8), textcoords="offset points", ha="center", fontsize=7.5,
                    color=INK)
    ax.axvline(0, color=AXIS, lw=0.8)
    ax.set_yticks(ys, [SHORT_LABELS[f] for f in factors])
    ax.set_xlabel("Points per game gained (+) or lost (-) after halftime")
    ax.grid(axis="y", visible=False)
    span = np.nanmax(np.abs(c[[f"{f}_per_game" for f in factors]].to_numpy())) * 1.2
    ax.set_xlim(-span, span)
    ax.set_ylim(-0.7, len(factors) - 0.3)
    ax.legend(loc="lower right", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.03, 1, 0.86))
    return _finish(fig, path, "Chronic or new? Each factor per game, this season vs last",
                   "Same exact decomposition, run on each season separately.", source)
