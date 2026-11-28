"""Players: the season line follows the picker, and the game table reads in words."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_kicker_follows_the_picked_season_and_table_is_readable():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120)
    at.session_state["_dashboard_page"] = "players"
    at.run()
    old = at.selectbox(key="player_season").options[-1]
    at.selectbox(key="player_season").set_value(old).run()
    assert not at.exception
    assert f"· {old} season ·" in " ".join(m.value for m in at.markdown)
    cols = list(at.dataframe[0].value.columns)
    assert "Date" in cols and not any("_" in c for c in cols)
