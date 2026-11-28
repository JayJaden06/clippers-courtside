"""Players: every card is a link, and following one opens that player (and season) once."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _app():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120)
    at.session_state["_dashboard_page"] = "players"
    return at


def test_every_card_links_to_its_player():
    at = _app()
    at.run()
    html = " ".join(m.value for m in at.markdown)
    assert html.count('class="pcard"') == html.count('class="pc-link"') > 0
    assert "?pick=" in html


def test_a_card_link_opens_that_player_and_season():
    at = _app()
    at.run()
    seasons = at.selectbox(key="player_season").options
    at = _app()
    at.query_params["pick"] = "Darius Garland"
    at.query_params["season"] = seasons[-1]
    at.run()
    assert not at.exception
    assert at.selectbox(key="player_season").value == seasons[-1]
    assert at.selectbox(key="player_pick").value == "Darius Garland"
    assert "pick" not in at.query_params  # a later rerun doesn't snap back to the card's player
