"""Players: one unit for how often a player's plays end in a turnover (percent), not three."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from courtside.viz import story as S

ROOT = Path(__file__).resolve().parents[1]


def test_safest_card_leads_with_percent():
    ph = pd.DataFrame([
        {"player_name": "A Guard", "split": s, "tov": 5, "tov_pct": .12, "tov_share": .3, "play_share": .3}
        for s in ("H1", "H2")])
    safe = [c for c in S.player_cards(ph, "A Guard") if c.title.startswith("Safest")][0]
    assert safe.big == "12%" and "plays end in a turnover" in safe.small


def test_players_page_has_no_one_in_n_numbers():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120)
    at.session_state["_dashboard_page"] = "players"
    at.run()
    html = " ".join(m.value for m in at.markdown)
    # "1 in N" survives only as the small "about 1 in N" gloss
    assert not re.search(r"(?<!about )\b1 in \d+", re.sub(r"<[^>]+>", " ", html))
    assert "<span>plays ends" not in html and "of his plays end in a turnover</span>" in html
