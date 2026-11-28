"""Verdict chips: white text on each chip's own background color reaches 4.5:1."""

from __future__ import annotations

from courtside.web.views.players import CHIP_INK


def _lum(h: str) -> float:
    c = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    c = [x / 12.92 if x <= .03928 else ((x + .055) / 1.055) ** 2.4 for x in c]
    return .2126 * c[0] + .7152 * c[1] + .0722 * c[2]


def test_white_on_every_chip_is_readable():
    for base, ink in CHIP_INK.items():
        assert 1.05 / (_lum(ink) + .05) >= 4.5, (base, ink)
