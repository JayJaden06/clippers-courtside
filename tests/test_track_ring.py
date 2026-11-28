"""Before/after tracks: the 'before' mark is a ring, so direction reads even when both marks are gray."""

from __future__ import annotations

from courtside.web.theme import MUTED_BAR
from courtside.web.ui import track


def test_before_is_a_ring_and_after_a_filled_dot_even_in_gray():
    html = track(0.2, 0.1, 0, 0.3, MUTED_BAR)
    assert html.count('class="dot b"') == 1 and html.count('class="dot"') == 1
    before = html.split('class="dot b"')[1].split(">")[0]
    assert "border-color" in before and "background" not in before
