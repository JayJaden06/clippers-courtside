"""'What now?' verdicts follow their written rules, on synthetic and on the committed data."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd
import pytest

from leadcollapse.config import previous_season
from leadcollapse.viz import whatnow as W

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
needs_data = pytest.mark.skipif(not (PROCESSED / "player_history_halves.parquet").exists(),
                                reason="run leadcollapse build")


def test_previous_season():
    assert previous_season("2025-26") == "2024-25"
    assert previous_season("2000-01") == "1999-00"


def _hist_row(name, season, split, **kw):
    base = dict(minutes=600.0, fgm=100, fga=200, fg3m=40, fg3a=100, fta=40, oreb=10, dreb=40, tov=20, pts=280,
                plus_minus=0, ast=40)
    return {"player_name": name, "player_id": 1, "season": season, "split": split, **(base | kw)}


def _empty_box():
    return pd.DataFrame(columns=["season", "team_abbreviation", "split", "player_name", *W.COUNTS])


def test_candidate_tags_follow_the_rules():
    hist = pd.DataFrame([
        _hist_row("Shooter", "2025-26", "H2", fgm=110, fg3m=45, fg3a=110, tov=10),  # eFG .6625, 3P .41, TOV ~4%
        _hist_row("Big", "2025-26", "H2", fgm=120, fg3m=0, fg3a=2, oreb=60, dreb=120, tov=40),  # 10.8 reb/36
    ])
    c = {x.name: x for x in W.candidates(hist, _empty_box(), "2026-27", "LAC", ["Shooter", "Big", "Nobody"], 0.53)}
    assert set(c) == {"Shooter", "Big"}
    assert set(c["Shooter"].helps) == {"shooting", "threes", "ball security"}
    assert "rebounding" in c["Big"].helps and "threes" not in c["Big"].helps
    assert c["Shooter"].sample == "2025-26, all teams" and not c["Shooter"].small_sample


def test_candidate_prefers_the_latest_big_sample():
    hist = pd.DataFrame([_hist_row("P", "2025-26", "H2", minutes=100.0), _hist_row("P", "2024-25", "H2")])
    c = W.candidates(hist, _empty_box(), "2026-27", "LAC", ["P"], 0.53)[0]
    assert c.sample.startswith("2024-25")
    small = W.candidates(hist[hist["season"] == "2025-26"], _empty_box(), "2026-27", "LAC", ["P"], 0.53)[0]
    assert small.small_sample


@needs_data
def test_on_the_committed_data():
    pb = pd.read_parquet(PROCESSED / "player_box.parquet")
    lu = pd.read_parquet(PROCESSED / "lineup_halves.parquet")
    hist = pd.read_parquet(PROCESSED / "player_history_halves.parquet")
    analysis = json.loads((ROOT / "reports" / "analysis.json").read_text(encoding="utf-8"))
    season = analysis["meta"]["season"]
    k3 = next(r["key"] for r in analysis["results"] if r["id"] == "H3")

    sd = W.shooting_drop(pb, season, "LAC", 0.526)
    assert sd.team_h2 == pytest.approx(next(r["key"] for r in analysis["results"] if r["id"] == "H1")["efg_h2"],
                                       abs=1e-3)
    assert 0 <= sd.n_worse <= sd.n_players

    sf = W.starting_five(pb, lu, season, "LAC")
    assert len(sf.names) == 5 and sf.games_together >= 1
    assert sf.not_the_problem == (sf.h2_min > 0 and sf.h2_net >= sf.rest_h2_net + W.STARTERS_MARGIN)
    # Starters + everyone else = all 192 second-half minutes (8 games x 24).
    assert sf.h2_min + sf.rest_h2_min == pytest.approx(24 * analysis["meta"]["games"], abs=0.5)

    bl = W.ball_load(pb, lu, k3, season, "LAC", "Darius Garland", analysis["meta"]["games"])
    assert bl.should_handle_less == (k3["tov_pct_h2"] >= W.LOAD_RATIO * k3["prior_h2_tov_pct"])
    assert bl.safest["tov_pct"].is_monotonic_increasing

    cands = W.candidates(hist, pb, season, "LAC", sorted(set(hist["player_name"])), 0.526)
    assert cands and all(c.h2_minutes > 0 for c in cands)

    lines = W.summary_lines(sf, bl, cands, sd, "Darius Garland")
    assert lines[0].startswith("<b>Keep the starting five" if sf.not_the_problem else "<b>The starting five is")
    assert any("Garland" in line for line in lines)
    assert not any(re.search(r"\bnan\b", line, re.IGNORECASE) for line in lines)  # not "Konan"
