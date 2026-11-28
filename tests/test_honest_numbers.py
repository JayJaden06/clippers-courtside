"""Blown leads states how sure it is: Garland's share as an upper bound, a small-sample caveat, no ranked noise."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
pytestmark = pytest.mark.skipif(not (PROCESSED / "game_halves.parquet").exists(), reason="run courtside build")


def test_the_answer_page_says_how_sure_it_is():
    from courtside.web.views.blown_leads import SETTLED_GAMES
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120)
    at.session_state["_dashboard_page"] = "answer"
    at.run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "at most about" in text  # Garland's share is the generous upper bound, and says so
    assert "−0.0" not in text and "-0.0" not in text
    games = pd.read_parquet(PROCESSED / "games.parquet")
    played = int((games["season"] == games["season"].max()).sum())
    assert ("lead, not a verdict" in text) == (played < SETTLED_GAMES)  # the caveat shows while the sample is small
