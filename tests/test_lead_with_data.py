"""Blown leads leads with whatever explains the most of the drop, and its shot titles follow the zones."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
pytestmark = pytest.mark.skipif(not (PROCESSED / "game_halves.parquet").exists(), reason="run courtside build")


def test_headline_follows_the_top_reason_and_the_first_half():
    from courtside.web.views.blown_leads import _headline
    assert _headline(-1.0, "opp_shooting") == "They're about even for a half. Then opponents start scoring."
    assert _headline(4.0, "shooting") == "They're ahead for a half. Then the shots stop falling."
    assert _headline(-6.0, None) == "They're behind early. Then it falls apart."


def test_shot_titles_only_claim_a_change_the_zones_show():
    from courtside.web.views.blown_leads import _shots_title
    steady = pd.DataFrame({"season": "x", "game_id": "g", "half": ["H1", "H2"] * 4, "loc_x": 0, "loc_y": 0,
                           "zone": ["At the rim"] * 8, "made": [True, True, False, False] * 2, "is_three": False})
    assert _shots_title(steady, "own") == "Their own shots, half by half"
    assert _shots_title(steady, "opp") == "Opponents' shots, half by half"


def test_the_answer_page_renders_the_opponent_map_when_opponents_lead():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120)
    at.session_state["_dashboard_page"] = "answer"
    at.run()
    assert not at.exception
    subs = [s.value for s in at.subheader]
    if (PROCESSED / "opp_shots.parquet").exists() and "opponents start scoring" in " ".join(m.value for m in at.markdown):
        assert any(s.startswith(("Where opponents", "Opponents'")) for s in subs), subs


def test_rates_and_shares_are_worded_by_their_numbers():
    from courtside.viz import story as S
    assert S._moves(0.533, 0.535) == "holds about steady"
    assert S._moves(0.56, 0.52) == "drops" and S._moves(0.52, 0.56) == "rises"
    assert S.share_text(1.03).startswith("more than the whole drop, since") and S.share_text(0.42) == "about 42% of the drop"
