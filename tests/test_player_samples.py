"""Players: a small sample gets no confident verdict, and nothing prints as "nan"."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from courtside.viz import story as S

ROOT = Path(__file__).resolve().parents[1]


def _ph(rows):
    out = []
    for name, plays, h1, h2, share in rows:
        for split, pct in (("H1", h1), ("H2", h2)):
            out.append({"player_name": name, "split": split, "plays": plays, "tov": round(plays * pct),
                        "tov_pct": pct, "tov_share": share, "play_share": share})
    return pd.DataFrame(out)


def test_enough_plays_needs_the_minimum_in_both_halves():
    ph = _ph([("A Guard", 120, .1, .2, .3), ("B Bench", S.MIN_PLAYS - 1, .2, .0, .1)])
    assert S.enough_plays(ph) == {"A Guard"}


def test_sloppiest_skips_a_tiny_sample_with_a_bigger_jump():
    ph = _ph([("A Guard", 120, .10, .14, .3), ("B Bench", 20, .05, .40, .1), ("C Role", 150, .02, .03, .02)])
    titles = [c.title for c in S.player_cards(ph, "A Guard")]
    assert "Sloppiest after halftime: Guard" in titles
    assert not any("Bench" in t or "Role" in t for t in titles if t.startswith("Sloppiest"))


def test_no_sloppiest_card_when_nobody_got_sloppier():
    ph = _ph([("A Guard", 120, .15, .14, .3), ("B Wing", 120, .10, .09, .2)])
    assert not any(c.title.startswith("Sloppiest") for c in S.player_cards(ph, "A Guard"))


def test_every_player_renders_without_nan():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120)
    at.session_state["_dashboard_page"] = "players"
    at.run()
    for name in at.selectbox(key="player_pick").options:
        at.selectbox(key="player_pick").set_value(name).run()
        assert not at.exception, (name, at.exception[0].value)
        html = " ".join(m.value for m in at.markdown)
        assert "nan" not in html.lower().replace("fernando", ""), name
