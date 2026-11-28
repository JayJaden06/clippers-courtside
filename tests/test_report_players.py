"""The game report's player cards: the toughest two need real minutes, and each view ends on the way on."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "data" / "processed" / "game_halves.parquet").exists(),
                                reason="run courtside build")


def _replay():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120)
    at.session_state["_dashboard_page"] = "replay"
    at.run()
    assert not at.exception
    return at


def test_toughest_nights_need_twelve_minutes_and_the_report_ends_on_the_way_on():
    from courtside.web.views.games import TOUGH_MIN
    at = _replay()
    cards = next(m.value for m in at.markdown if "Ranked by Game Score" in m.value)
    assert cards  # the section rendered
    html = " ".join(m.value for m in at.markdown)
    i = html.index("Ranked by Game Score")
    played = re.findall(r"in (\d+) min · ", html[i:i + 20000])
    assert played[:5], "player cards"
    assert all(int(m) >= TOUGH_MIN for m in played[3:5]), played[:5]  # the toughest two
    assert any("Why the second half keeps deciding" in m.value for m in at.markdown)
