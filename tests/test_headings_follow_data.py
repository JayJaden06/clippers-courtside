"""Headings that state a direction are written from the numbers, so they can't contradict them."""

from __future__ import annotations

from courtside.viz import story as S


def test_the_shooting_card_says_what_the_numbers_say():
    assert S.shooting_title(0.56, 0.52) == "Shots stop falling after halftime"
    assert S.shooting_title(0.539, 0.538) == "Their own shooting holds steady"  # 54% -> 54% isn't a drop
    assert S.shooting_title(0.52, 0.56) == "Their shots fall more after halftime"


def test_no_answer_card_claims_a_drop_the_numbers_dont_show():
    hl = {"h1_per_game": -1.0, "h2_per_game": -6.4, "blown": 4, "leads": 9, "efg_h1": 0.539, "efg_h2": 0.538}
    cards = S.answer_page_cards(hl, [], {}, "2025-26")
    shooting = next(c for c in cards if "shoot" in c.title.lower())
    assert "stop falling" not in shooting.title and shooting.tone == "info"
