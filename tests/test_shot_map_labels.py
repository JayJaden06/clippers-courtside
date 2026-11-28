"""The shot map's zone labels never overlap, and each sits in its own zone."""

from __future__ import annotations

from math import hypot

from courtside.viz.graphics import ZONE_SPOTS

BOX_W, BOX_H = 196, 62  # the label box drawn behind each zone's numbers


def test_zone_label_boxes_do_not_overlap():
    spots = list(ZONE_SPOTS.items())
    for i, (a, (ax, ay)) in enumerate(spots):
        for b, (bx, by) in spots[i + 1:]:
            apart = abs(ax - bx) >= BOX_W or abs(ay - by) >= BOX_H
            assert apart, (a, b)


def test_mid_range_label_stays_inside_the_arc():
    x, y = ZONE_SPOTS["Mid-range"]
    assert hypot(x, y) < 237.5  # the three-point arc is 23.75 ft
