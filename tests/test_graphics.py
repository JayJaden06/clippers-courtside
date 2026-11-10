"""The SVG graphics carry the numbers they're given and are well-formed."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET

from courtside.viz import graphics as G


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


def test_turnover_strip_one_square_per_turnover():
    root = _svg(G.turnover_strip(12, 8, 18))
    squares = [r for r in root.iter("{http://www.w3.org/2000/svg}rect") if r.get("width") == "26"]
    assert len(squares) == 38
    assert sum(r.get("fill") == G.DARK.focus for r in squares) == 8


def test_mini_pictures():
    dots = _svg(G.mini(("dots", 4, 6), "bad"))
    circles = list(dots.iter("{http://www.w3.org/2000/svg}circle"))
    assert len(circles) == 6 and sum(c.get("fill") == G.DARK.bad for c in circles) == 4
    rank = _svg(G.mini(("rank", 24, 30), "bad"))
    big = [c for c in rank.iter("{http://www.w3.org/2000/svg}circle") if c.get("r") == "7"]
    assert len(big) == 1
    for viz in (("pair", 0.58, 0.52, "1st", "2nd"), ("share", 0.35, 0.40), ("diverge", (0.7, -4.6), ("a", "b"))):
        _svg(G.mini(viz, "warn"))  # well-formed
    assert G.mini(("nope",)) == ""


def test_scoreboard_and_ball_ladder():
    texts = _texts(G.scoreboard("LAC", "MIA", [24, 43, 20, 20, 9], [33, 19, 26, 29, 10], False))
    assert {"Q1", "Q4", "OT", "2ND HALF", "FINAL"} <= set(texts)
    assert "116" in texts and "117" in texts and "40" in texts and "55" in texts  # finals; 2nd halves
    ladder = _svg(G.ball_ladder([("Garland", 5.0, 4.0), ("Lopez", 10.0, 12.0)]))
    reds = [c for c in ladder.iter("{http://www.w3.org/2000/svg}circle") if c.get("fill") == G.DARK.bad]
    balls = list(ladder.iter("{http://www.w3.org/2000/svg}circle"))
    assert len(reds) == 2 and len(balls) == 4 + 12


def test_figures_can_wear_the_players_face():
    url = "https://cdn.nba.com/headshots/nba/latest/260x190/1629636.png"
    plain = G.size_matchup("Wagler", 78, 175, 77, 208, 0.57, 0.49)
    faced = G.size_matchup("Wagler", 78, 175, 77, 208, 0.57, 0.49, photo=url)
    assert "<image" not in plain and faced.count(f'href="{url}"') == 1 and "clipPath" in faced
    tall = G.height_matchup("Garland", 73, 192, 11, 24, 25, 22, photo=url)
    assert tall.count(f'href="{url}"') == 2  # one face per panel, each with its own clip id
    assert len(set(re.findall(r'clipPath id="(\w+)"', tall))) == 2
