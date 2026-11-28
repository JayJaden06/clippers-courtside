"""Players: a small-sample player's card dims every number, not just the first two."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _hero(name: str) -> str:
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120)
    at.session_state["_dashboard_page"] = "players"
    at.run()
    at.selectbox(key="player_pick").set_value(name).run()
    return next(m.value for m in at.markdown if 'class="hero-card"' in m.value)


def test_small_sample_dims_all_four_numbers_and_a_regular_dims_none():
    from courtside.viz import dashboard_data as D
    from courtside.viz import story as S

    data = D.load(ROOT / "data" / "processed", ROOT / "reports" / "analysis.json")
    ph = D.player_halves(data.player_stats, data.seasons[0])
    few = sorted(set(ph["player_name"]) - S.enough_plays(ph))
    if not few:
        pytest.skip("no small-sample player this season")
    bold = re.findall(r'<b class="([^"]*)">', _hero(few[0]).split('hc-stats">', 1)[1])
    assert len(bold) == 4 and all("hc-few" in c for c in bold)
    regular = sorted(S.enough_plays(ph))[0]
    assert "hc-few" not in _hero(regular)
