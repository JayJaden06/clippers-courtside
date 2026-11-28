"""The game report's story stays on one baseline: no first-half clause beside the normal-night ledger."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "data" / "processed" / "game_halves.parquet").exists(),
                                reason="run courtside build")


def test_the_game_report_story_has_no_first_half_clause():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120)
    at.session_state["_dashboard_page"] = "replay"
    at.run()
    assert not at.exception
    story = next(m.value for m in at.markdown if 'class="story"' in m.value)
    assert "their own first half" not in story and "vs the first half" not in story
