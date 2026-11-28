"""Players: every card is on show in balanced rows, and each carries the player a click picks in place."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_cards_wrap_into_balanced_rows_and_name_their_player():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120)
    at.session_state["_dashboard_page"] = "players"
    at.run()
    html = next(m.value for m in at.markdown if 'class="pcards"' in m.value)
    n = html.count('class="pcard"')
    cols = int(re.search(r"--cols:(\d+)", html).group(1))
    rows = -(-n // cols)
    assert cols <= 8 and n > cols * (rows - 1) >= 0 and cols - (n - cols * (rows - 1)) <= rows  # no lone orphan
    assert html.count("data-pick=") == n
    css = (ROOT / "src" / "courtside" / "web" / "style.css").read_text(encoding="utf-8")
    assert "overflow-x: auto" not in css.split(".pcards {", 1)[1].split("}", 1)[0]
