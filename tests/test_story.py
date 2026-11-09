"""The dashboard's plain-English sentences follow the numbers, not a script."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from leadcollapse.viz import story as S

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS = ROOT / "reports" / "analysis.json"
pytestmark = pytest.mark.skipif(not ANALYSIS.exists(), reason="run leadcollapse analyze")


@pytest.fixture(scope="module")
def analysis() -> dict:
    return json.loads(ANALYSIS.read_text(encoding="utf-8"))


def _with(analysis: dict, hid: str, **key) -> dict:
    a = copy.deepcopy(analysis)
    for r in a["results"]:
        if r["id"] == hid:
            r["key"].update(key)
    return a


def test_one_card_per_question_with_numbers(analysis):
    cards = S.answer_cards(analysis)
    assert [c.id for c in cards] == list(S.QUESTIONS)
    for c in cards:
        assert c.tone in ("yes", "partly", "no") and c.answer
        assert any(ch.isdigit() for ch in c.text)
        assert "nan" not in c.text.lower()


@pytest.mark.parametrize("share, answer, tone", [(0.55, "Yes, a big part", "yes"), (0.25, "Partly", "partly"),
                                                 (0.05, "Only a little", "no")])
def test_garland_card_follows_his_share(analysis, share, answer, tone):
    card = next(c for c in S.answer_cards(_with(analysis, "H3", share_of_swing=share)) if c.id == "H3")
    assert (card.answer, card.tone) == (answer, tone)
    assert f"{share:.0%}" in card.text
    assert ("short of the" in card.text) == (tone != "yes")


def test_collapse_card_is_hedged_when_the_interval_includes_zero(analysis):
    unsure = next(c for c in S.answer_cards(_with(analysis, "H1", net_change_rank=28, boot_hi=3.0)) if c.id == "H1")
    sure = next(c for c in S.answer_cards(_with(analysis, "H1", net_change_rank=28, boot_hi=-1.0)) if c.id == "H1")
    assert unsure.answer == "Probably" and "bad luck" in unsure.text
    assert sure.answer == "Yes" and "bad luck" not in sure.text
    assert "Only 2 other teams" in sure.text


def test_game_story_reads_the_game():
    row = pd.Series({"halftime_margin": 15, "final_margin": -1, "max_lead": 20, "opponent_abbreviation": "MIA",
                     "overtime_periods": 1, "result": "L"})
    halves = pd.DataFrame({"split": ["H1", "H2"], "pts": [67, 40], "opp_pts": [52, 55]})
    att = pd.DataFrame({"factor": S.factor_order(), "points": [-26, 11, -4, -4, -7, 1, -1]}).assign(margin_change=-30)
    text = S.game_story(row, halves, att)
    assert text.startswith("They led by as many as 20 against MIA and were up 15 at halftime.")
    assert "outscored 55-40" in text and "lost by 1 in overtime" in text
    assert "Clippers' shooting (-26" in text


def test_game_story_when_they_never_led_and_improved():
    row = pd.Series({"halftime_margin": -18, "final_margin": -16, "max_lead": 0, "opponent_abbreviation": "LAL",
                     "overtime_periods": 0, "result": "L"})
    halves = pd.DataFrame({"split": ["H1", "H2"], "pts": [40, 60], "opp_pts": [58, 58]})
    att = pd.DataFrame({"factor": S.factor_order(), "points": [7, 4, 6, -3, 1, -1, 6]}).assign(margin_change=20)
    text = S.game_story(row, halves, att)
    assert text.startswith("They never led") and "won the second half 60-58" in text
    assert "improvement: Clippers' shooting" in text


# ------------------------------------------------------------ the full answer

@pytest.fixture(scope="module")
def answer_inputs():
    from leadcollapse.viz import dashboard_data as D

    d = D.load(ROOT / "data" / "processed", ANALYSIS)
    meta = d.analysis["meta"]
    sh = D.season_halves(d.game_halves)
    return d, meta, sh, S.ranked_reasons(d.attribution[meta["season"]], d.attribution[meta["comparison_season"]],
                                         sh.loc[meta["season"]])


def test_reasons_are_ranked_and_add_up_to_the_swing(answer_inputs):
    d, meta, _, reasons = answer_inputs
    att = d.attribution[meta["season"]].set_index("factor")
    assert [r.per_game for r in reasons] == sorted(r.per_game for r in reasons)
    total = sum(r.per_game for r in reasons) + att.loc["residual", "per_game"]
    assert total == pytest.approx(att["per_game"].sum())
    assert sum(r.share for r in reasons) + att.loc["residual", "share_of_swing"] == pytest.approx(1.0, abs=1e-3)
    assert all(r.detail and "nan" not in r.detail for r in reasons)


@pytest.mark.parametrize("cur, prev, status", [(-6.1, 1.6, S.NEW), (-1.5, -0.8, S.SAME), (2.5, 0.1, S.BETTER),
                                               (-0.4, -0.2, S.SIMILAR)])
def test_status_uses_the_report_rules(cur, prev, status):
    assert S._status(cur, prev) == status


def test_scorecard_covers_both_seasons(answer_inputs):
    _, meta, sh, _ = answer_inputs
    rows = S.scorecard(sh, meta["season"], meta["comparison_season"])
    assert len(rows) == len(S.SCORECARD)
    margin = rows[0]
    comp = pd.read_csv(ROOT / "data" / "processed" / "analysis" / "season_comparison.csv").set_index("season")
    assert float(margin.cur_h2) == pytest.approx(comp.loc[meta["season"], "margin_h2_per_game"], abs=0.05)
    assert {r.verdict for r in rows} <= {"Worse than last season", "Better than last season", "About the same"}


def test_what_if_is_additive_and_only_improves(answer_inputs):
    _, _, _, reasons = answer_inputs
    base = -6.25
    assert S.what_if(base, reasons, set()) == base
    by = {r.factor: r for r in reasons}
    one = S.what_if(base, reasons, {"shooting"})
    assert one == pytest.approx(base + by["shooting"].prev_per_game - by["shooting"].per_game)
    both = S.what_if(base, reasons, {"shooting", "rebounding"})
    assert both - base == pytest.approx((one - base) + (S.what_if(base, reasons, {"rebounding"}) - base))
    # Fixing something that was worse last season never makes it worse.
    assert S.what_if(base, reasons, {r.factor for r in reasons}) >= base
    # Garland's piece is skipped when all turnovers are already fixed.
    assert S.what_if(base, reasons, {"turnovers"}, 2.2, True) == S.what_if(base, reasons, {"turnovers"})


def test_sixty_seconds_leads_with_the_biggest_reason(answer_inputs):
    d, meta, _, reasons = answer_inputs
    hl = S.headline(d.analysis, d.comparison, d.games, meta["season"])
    res = {r["id"]: r for r in d.analysis["results"]}
    lines = S.sixty_seconds(reasons, res, hl, meta["comparison_season"])
    assert 3 <= len(lines) <= 4
    assert S._lower(reasons[0].name) in lines[0]
    garland = next(line for line in lines if "Garland" in line)
    assert ("a main cause" in garland) == (res["H3"]["key"]["share_of_swing"] >= S.DRIVES)


def test_worst_collapse_is_a_blown_lead_with_the_biggest_fall(answer_inputs):
    d, meta, _, _ = answer_inputs
    w = S.worst_collapse(d.game_leads, meta["season"])
    g = d.game_leads[(d.game_leads["season"] == meta["season"]) & (d.game_leads["lead_class"] == "blown_lead")]
    fall = g["max_lead"] - g["final_margin"]
    row = g.set_index("game_id").loc[w["game_id"]]
    assert row["max_lead"] - row["final_margin"] == fall.max()


# ------------------------------------------------------------- page cards

def test_answer_page_cards_follow_the_headline(analysis):
    hl = {"wins": 2, "losses": 7, "h1_per_game": 0.7, "h2_per_game": -4.6, "blown": 4, "leads": 6,
          "efg_h1": 0.577, "efg_h2": 0.517}
    reasons = [S.Reason("shooting", "Clippers' shooting", -5.0, 0.0, 0.95, S.NEW, ""),
               S.Reason("turnovers", "Turnovers", -1.9, -0.5, 0.35, S.SAME, "")]
    res = {r["id"]: r for r in analysis["results"]}
    cards = {c.title: c for c in S.answer_page_cards(hl, reasons, res, "2025-26")}
    assert cards["Early leads blown"].big == "4 of 6"
    assert cards["Fine before halftime, behind after it"].big == "+0.7 → -4.6"
    assert cards["Biggest reason: Clippers' shooting"].big == "-5.0"
    assert cards["New problems this season"].big == "1"


def test_player_and_league_cards():
    ph = pd.DataFrame([
        {"player_name": "A Guard", "split": "H1", "tov": 5, "tov_pct": 0.10, "tov_share": 0.2, "play_share": 0.2},
        {"player_name": "A Guard", "split": "H2", "tov": 9, "tov_pct": 0.20, "tov_share": 0.4, "play_share": 0.2},
        {"player_name": "B Wing", "split": "H1", "tov": 2, "tov_pct": 0.08, "tov_share": 0.1, "play_share": 0.1},
        {"player_name": "B Wing", "split": "H2", "tov": 2, "tov_pct": 0.05, "tov_share": 0.1, "play_share": 0.1},
    ])
    cards = S.player_cards(ph, "A Guard")
    assert [c.title for c in cards] == ["Most turnovers after halftime: Guard", "Sloppiest after halftime: Guard",
                                        "Safest with the ball: Wing"]
    assert cards[2].big == "1 in 20"

    ranks = pd.DataFrame({"1st half": [15, 3, 23], "2nd half": [25, 19, 20], "change": [24, 28, 12]},
                         index=["Net rating", "Shooting accuracy (eFG%)", "Offensive rebounding"])
    lc = S.league_cards(ranks)
    assert [c.big for c in lc] == ["24th of 30", "28th of 30", "12th of 30"]
    assert lc[1].title == "Worst after halftime: shooting accuracy (eFG%)"


def test_game_cards():
    row = pd.Series({"halftime_margin": 7, "final_margin": -3, "overtime_periods": 0}, name="g1")
    halves = pd.DataFrame({"split": ["H1", "H2"], "margin": [7, -10]})
    tov = pd.DataFrame({"game_id": ["g1"] * 3, "is_team": [True, True, False], "period": [3, 4, 3],
                        "player_name": ["Garland", "Dunn", "Other"]})
    cards = S.game_cards(row, halves, tov, "Garland")
    assert [c.big for c in cards] == ["+7", "-10", "-3", "2"]
    assert cards[3].small == "Most: Dunn and Garland (1 each)." and cards[2].tone == "bad"
    tov2 = pd.concat([tov, tov.iloc[[1]]])  # Dunn now has 2
    assert S.game_cards(row, halves, tov2, "Garland")[3].small == "Most: Dunn (2). Garland had 1."
