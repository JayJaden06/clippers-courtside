"""The SVG graphics carry the numbers they're given and are well-formed."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from leadcollapse.viz import graphics as G


def _svg(html: str) -> ET.Element:
    inner = re.search(r"<svg.*</svg>", html, re.S).group(0)
    return ET.fromstring(inner)  # raises if the SVG isn't well-formed


def _texts(html: str) -> list[str]:
    return [t.text for t in _svg(html).iter("{http://www.w3.org/2000/svg}text")]


def test_court_halves_shows_both_margins_and_tints_by_sign():
    html = G.court_halves(0.7, -4.6)
    assert "+0.7" in _texts(html) and "−4.6" in _texts(html)
    assert G.DARK.good in html and G.DARK.bad in html
    assert 'aria-label="Basketball court split in two' in html


def test_wall_court_percentages():
    texts = _texts(G.wall_court(0.764, 0.812))
    assert "76%" in texts and "81%" in texts and "THE WALL" in texts


def test_people_are_drawn_to_scale():
    html = G.size_matchup("Wagler", 78, 175, 77, 208, 0.57, 0.49)
    assert "+33 lb" in _texts(html) and "57%" in _texts(html)
    heights = [float(r.get("height")) for r in _svg(html).iter("{http://www.w3.org/2000/svg}rect")]
    assert heights  # torso and legs
    tall = _svg(G.height_matchup("Garland", 73, 192, 11, 24, 25, 22))
    heads = [float(c.get("cy")) for c in tall.iter("{http://www.w3.org/2000/svg}circle")]
    assert len(heads) == 4 and min(heads) < max(heads)  # the bigger defender's head sits higher


def test_missed_shots_waffle_counts():
    root = _svg(G.missed_shots(0.26, 0.31, 0.28))
    fills = [r.get("fill") for r in root.iter("{http://www.w3.org/2000/svg}rect")]
    assert len(fills) == 200
    assert fills[:100].count(G.DARK.gray) == 26 and fills[100:].count(G.DARK.bad) == 31
