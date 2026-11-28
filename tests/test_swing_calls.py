"""Players: sloppier or safer only when the change beats the normal game-to-game swing."""

from __future__ import annotations

import pandas as pd

from courtside.viz import story as S


def _ph(rows):
    out = []
    for name, plays, h1, h2, share in rows:
        for split, pct in (("H1", h1), ("H2", h2)):
            out.append({"player_name": name, "split": split, "plays": plays, "tov": plays * pct,
                        "tov_pct": pct, "tov_share": share, "play_share": share})
    return pd.DataFrame(out)


def test_a_small_change_on_few_plays_is_within_normal_swing():
    c = S.change_calls(_ph([("A Guard", 190, .17, .22, .3)])).loc["A Guard"]
    assert c["call"] == "Within normal swing" and c["swing"] > abs(c["d"])


def test_the_same_change_on_many_plays_is_a_call_and_the_sign_sets_the_word():
    up = S.change_calls(_ph([("A Guard", 2000, .17, .22, .3)])).loc["A Guard"]
    down = S.change_calls(_ph([("A Guard", 2000, .22, .17, .3)])).loc["A Guard"]
    assert (up["call"], down["call"]) == ("Sloppier", "Safer")


def test_tile_says_nobody_clearly_when_no_change_clears_the_swing():
    cards = S.player_cards(_ph([("A Guard", 190, .17, .22, .3), ("B Wing", 150, .11, .09, .2)]), "A Guard")
    tile = next(c for c in cards if "loppi" in c.title)
    assert tile.title == "Sloppier after halftime: nobody clearly" and "Guard" in tile.small


def test_safest_needs_enough_plays():
    cards = S.player_cards(_ph([("A Guard", 600, .15, .15, .3), ("B Bench", 20, .0, .0, .1)]), "A Guard")
    assert "Safest with the ball: Guard" in [c.title for c in cards]


def test_short_half_names_the_half_and_the_real_count():
    c = S.change_calls(_ph([("A Guard", 49.76, .1, .12, .1)])).loc["A Guard"]
    assert S.short_half(c) == "only 49 plays before halftime"
