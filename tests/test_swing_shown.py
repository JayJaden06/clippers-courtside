"""Players: the normal swing is drawn and said in words, and the rotation gets one sentence from the calls."""

from __future__ import annotations

import pandas as pd

from courtside.viz import story as S
from courtside.viz.graphics import mini
from courtside.web.ui import track


def _ph(rows):
    out = []
    for name, plays, h1, h2 in rows:
        for split, pct in (("H1", h1), ("H2", h2)):
            out.append({"player_name": name, "split": split, "plays": plays, "tov": plays * pct,
                        "tov_pct": pct, "tov_share": .3, "play_share": .3})
    return pd.DataFrame(out)


def test_words_match_the_rounded_percents_and_flip_with_the_sign():
    up = S.change_calls(_ph([("A Guard", 190, .17, .224)])).loc["A Guard"]
    down = S.change_calls(_ph([("A Guard", 190, .224, .17)])).loc["A Guard"]
    assert S.change_words(up).startswith("up 5 in every 100 plays · his games usually swing ±")
    assert S.change_words(down).startswith("down 5 in every 100 plays")
    assert "pts" not in S.change_words(up)


def test_summary_says_it_isnt_one_player_when_nobody_moves():
    calls = S.change_calls(_ph([("A Guard", 190, .17, .20), ("B Wing", 150, .11, .10), ("C Bench", 20, .1, .4)]))
    assert S.swing_summary(calls) == ("All 2 players with enough plays stayed within their normal swing after "
                                      "halftime: it isn't one player.")


def test_summary_names_who_moved_when_someone_does():
    calls = S.change_calls(_ph([("A Guard", 3000, .12, .20), ("B Wing", 150, .11, .10)]))
    assert S.swing_summary(calls) == "Beyond their normal swing: Guard sloppier. The other 1 stayed within it."


def test_band_is_drawn_around_before():
    assert 'class="band"' in track(.2, .22, 0, .4, "#999", band=.05)
    assert 'class="band"' not in track(.2, .22, 0, .4, "#999")
    assert "normal swing" in mini(("swing", .17, .22, .08))


def test_tile_figure_is_a_word_not_an_empty_value():
    cards = S.player_cards(_ph([("A Guard", 190, .17, .22), ("B Wing", 150, .11, .09)]), "A Guard")
    tile = next(c for c in cards if "loppi" in c.title)
    assert tile.big == "Nobody" and tile.viz[0] == "swing"
