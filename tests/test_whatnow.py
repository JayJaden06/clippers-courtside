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


# ------------------------------------------------------------- tracking-based rules

def _matchup(gid, off_id, def_id, poss, tov=0, pts=0, fgm=0, fga=0, season="2026-27", off_team="LAC",
             def_team="OPP"):
    return {"season": season, "game_id": gid, "off_team": off_team, "off_player_id": off_id,
            "off_player": f"P{off_id}", "def_team": def_team, "def_player_id": def_id, "def_player": f"D{def_id}",
            "partial_poss": poss, "tov": tov, "pts": pts, "fgm": fgm, "fga": fga, "shooting_fouls": 0}


def _tov(gid, pid, stealer=None, season="2026-27", period=3):
    return {"season": season, "game_id": gid, "team_id": 1, "person_id": pid, "period": period,
            "live_ball": stealer is not None, "stealer_id": stealer, "category": "bad_pass"}


def test_help_steals_counts_steals_by_someone_other_than_his_defender():
    m = pd.DataFrame([_matchup("g1", 10, 100, 50, tov=2), _matchup("g1", 10, 101, 20, tov=1)])
    t = pd.DataFrame([_tov("g1", 10, 100), _tov("g1", 10, 100), _tov("g1", 10, 102), _tov("g1", 10)])
    hs = W.help_steals(t, m, "2026-27", "LAC", 1)
    assert hs.loc[10].to_dict() == {"tov": 4, "steals": 3, "own": 2, "help": 1}
    # a defender's steals beyond the turnovers he's credited with count as help
    t2 = pd.DataFrame([_tov("g1", 10, 101), _tov("g1", 10, 101)])
    assert W.help_steals(t2, m, "2026-27", "LAC", 1).loc[10, "help"] == 1


def test_length_and_size_rules():
    bio = pd.DataFrame([
        {"season": "2026-27", "player_id": 100, "height_in": 80, "weight_lb": 230, "rookie": False},
        {"season": "2026-27", "player_id": 101, "height_in": 74, "weight_lb": 190, "rookie": False},
        {"season": "2026-27", "player_id": 10, "height_in": 73, "weight_lb": 175, "rookie": True},
        {"season": "2026-27", "player_id": 11, "height_in": 80, "weight_lb": 230, "rookie": False},
    ])
    # P10 turns it over 10 per 100 vs the long defender and 2 vs the short one; teammate P11 has no gap
    m = pd.DataFrame([_matchup("g1", 10, 100, 100, tov=10), _matchup("g1", 10, 101, 100, tov=2),
                      _matchup("g1", 11, 100, 100, tov=3), _matchup("g1", 11, 101, 100, tov=3),
                      _matchup("g0", 10, 100, 100, tov=3, season="2025-26"),
                      _matchup("g0", 10, 101, 100, tov=3, season="2025-26")])
    t = pd.DataFrame([_tov("g1", 10) for _ in range(12)])
    p = W.pressure(m, t, bio, "2026-27", "2025-26", "LAC", 1, 10)
    assert p.tov100_long == pytest.approx(10) and p.tov100_short == pytest.approx(2)
    assert p.length_problem and not p.trapped and p.prior_tov100_long == pytest.approx(3)

    # On defense: 10 (175 lb) guards 100 (230 lb) and gives up 55 lb; 11 guards 101 and doesn't
    # opponents shoot 60% on 10 and 40% on 11 (team 50%), on 40 shots each
    d = pd.DataFrame([_matchup("g1", 100, 10, 50, pts=60, fgm=24, fga=40, off_team="OPP", def_team="LAC"),
                      _matchup("g1", 101, 11, 50, pts=40, fgm=16, fga=40, off_team="OPP", def_team="LAC")])
    hustle = pd.DataFrame(columns=["season", "team_abbreviation", "player_id", "minutes", "contested_shots",
                                   "deflections", "def_box_outs"])
    dfn = W.defenders(d, hustle, bio, "2026-27", "LAC", min_poss=0).table.set_index("player_id")
    assert dfn.loc[10, "gives_up_size"] and not dfn.loc[11, "gives_up_size"]
    assert dfn.loc[10, "attacked"] and not dfn.loc[11, "attacked"] and dfn.loc[10, "rookie"]


def test_verdict_cards_on_synthetic_rules():
    sf = W.StartingFive(["A", "B", "C", "D", "E"], 5, 50, 0, 40, 2.0, 200, -12.0, True, h2_pm=1.0, rest_h2_pm=-45.0)
    sd = W.ShootingDrop(pd.DataFrame(), 0, 0, 0.55, 0.50, 0.53, 0.33)
    cards = W.verdicts(sf, None, [], sd, "Darius Garland", n_games=9)
    assert [c.title for c in cards] == ["Keep the starting five"]
    assert cards[0].big == "+0.1 vs -5.0" and cards[0].anchor == "starters"


@pytest.mark.skipif(not (PROCESSED / "matchups.parquet").exists(), reason="run leadcollapse build")
def test_tracking_rules_on_the_committed_data():
    r = lambda n: pd.read_parquet(PROCESSED / f"{n}.parquet")  # noqa: E731
    pb, lu, oo, bio, m, hu, t, gh = (r(n) for n in ("player_box", "lineup_halves", "onoff_halves", "player_bio",
                                                    "matchups", "hustle", "turnover_events", "game_halves"))
    analysis = json.loads((ROOT / "reports" / "analysis.json").read_text(encoding="utf-8"))
    season, prior = analysis["meta"]["season"], analysis["meta"]["comparison_season"]

    rot = W.rotation(pb, oo, bio, season, "LAC", min_minutes=0)
    full = pb[(pb["season"] == season) & (pb["team_abbreviation"] == "LAC") & (pb["split"].astype(str) == "FULL")]
    assert (rot["mpg"] * rot["gp"]).sum() == pytest.approx(full["minutes"].sum(), rel=1e-6)
    assert (rot["h2_mpg"] <= rot["mpg"] + 1e-9).all() and (rot["q4_mpg"] <= rot["h2_mpg"] + 1e-9).all()

    dfn = W.defenders(m, hu, bio, season, "LAC", min_poss=0)
    mine = m[(m["season"] == season) & (m["def_team"] == "LAC")]
    assert dfn.table["poss"].sum() == pytest.approx(mine["partial_poss"].sum())

    gl = W.glass(gh, season, prior)
    assert 0 < gl.opp_oreb_h1 < 1 and 0 < gl.opp_oreb_h2 < 1
    assert gl.worse == (gl.opp_oreb_h2 >= gl.prior_opp_oreb_h2 + W.GLASS_WORSE)

    pid = int(pb.loc[pb["player_name"] == analysis["meta"]["focus_player"], "player_id"].iloc[0])
    p = W.pressure(m, t, bio, season, prior, "LAC", 1610612746, pid)
    assert p.help <= p.steals <= p.tov
    assert p.length_problem == (p.tov100_long >= W.LENGTH_RATIO * p.tov100_short and p.tov_long >= W.LENGTH_MIN_TOV)

    c = W.chemistry(pb, lu, season, prior, "LAC")
    assert 0 < c.continuity < 1 and not set(c.new) & set(c.returning)
    assert (c.pairs["min"] >= 25).all() and (c.pairs["a_id"] < c.pairs["b_id"]).all()
