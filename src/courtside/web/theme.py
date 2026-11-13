"""Colors and small text helpers shared by every page.

One meaning per color, validated on the light surface (dataviz validator: blue/orange pass every
colorblind check; every mark is 3:1+ against the page):
  blue = better / good · orange = worse / bad, the second half, and the focus · gray = context.
"""

from __future__ import annotations

import re


# One meaning per color, validated on the dark surface (dataviz validator: blue/orange pass every
# colorblind check; every mark is 3:1+ against the page):
#   blue = better / good · orange = worse / bad, the second half, and the focus · gray = context.
H1C, H2C = "#86847e", "#d95926"  # 1st half (context, 3.4:1 on the page) / 2nd half (the story)


GOOD, BAD, GRAY = "#3987e5", "#d95926", "#898781"


FOCUS = "#d95926"


INK_MARK = "#3b3f4c"


LIGHT = "#86847e"  # 3.4:1 on the light page


MUTED_BAR = "#7f7d77"  # 3.8:1 on the light page


BG = "#ffffff"  # the white panel every chart sits on (marker outlines match it)


PLOTLY_CONFIG = {"displayModeBar": False}


TONE_BADGE = {"yes": ("orange", "The answer"), "partly": ("violet", "Part of it"), "no": ("gray", "Not really")}


def signed(v: float, digits: int = 1) -> str:
    return f"{v:+.{digits}f}".replace("-", "−")


TONE_COLOR = {"good": GOOD, "bad": BAD, "warn": FOCUS, "info": GRAY}


def _minus(text: str) -> str:
    """A hyphen before a digit becomes a true minus sign, so every number on the page looks the same."""
    return re.sub(r"(?<![\w])-(?=\d)", "−", text)
