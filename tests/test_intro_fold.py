"""Games page: a headless run of the app (split from test_dashboard.py so parallel changes don't collide)."""

from __future__ import annotations

from pathlib import Path

import pytest

from courtside.viz import dashboard_data as D

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
pytestmark = pytest.mark.skipif(not (PROCESSED / "game_halves.parquet").exists(), reason="run courtside build")


@pytest.fixture(scope="module")
def data() -> D.DashboardData:
    return D.load(PROCESSED, ROOT / "reports" / "analysis.json")


def _app(page: str):
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120)
    at.session_state["_dashboard_page"] = page
    at.run()
    assert not at.exception, (page, at.exception[0].value)
    return at



def test_the_intro_folds_after_the_first_view_and_the_banner_hides_on_its_own_game(data):
    at = _app("replay")
    html = lambda: " ".join(m.value for m in at.markdown)  # noqa: E731
    assert "Clippers Courtside</div>" in html()  # the full welcome on arrival
    at.run()
    assert "intro-line" in html() and "Clippers Courtside</div>" not in html()  # one line once you're exploring
    next(b for b in at.button if b.label == "About this site").click().run()
    assert "Clippers Courtside</div>" in html()
    key = f"pick_{data.seasons[0]}"
    nxt = [o for o in at.selectbox(key=key).options if "next game" in o]
    if nxt:
        assert 'class="nextup"' in html()
        at.selectbox(key=key).set_value(nxt[0]).run()
        assert not at.exception and 'class="nextup"' not in html()  # no banner above the plan it opens
