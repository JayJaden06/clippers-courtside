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
