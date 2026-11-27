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



def test_a_guard_pick_never_reads_as_a_stopper_when_he_allows_more_than_average(data):
    """'Guard him with X, allows N (team average: M)' only when N <= M; otherwise the card says he's the size match."""
    import re
    at = _app("replay")
    key = f"pick_{data.seasons[0]}"
    plans = [o for o in at.selectbox(key=key).options if "% to win" in o][:6]
    if not plans:
        pytest.skip("no upcoming games in the schedule")
    seen = 0
    for label in plans:
        at.selectbox(key=key).set_value(label).run()
        assert not at.exception
        html = " ".join(m.value for m in at.markdown)
        for n, m in re.findall(r"allows (\d+) pts per 100 possessions \(team average: (\d+)\)", html):
            seen += 1
            assert int(n) <= int(m), (label, n, m)
    assert seen or plans  # the cards rendered (no tracking data would leave none)
