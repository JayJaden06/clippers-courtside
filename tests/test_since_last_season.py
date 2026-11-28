"""'Since last season' and the what-if agree with the ranked chart: no helper called a new problem, and a part that
costs less than last season isn't said to cost 'about the same'."""

from __future__ import annotations

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
pytestmark = pytest.mark.skipif(not (PROCESSED / "game_halves.parquet").exists(), reason="run courtside build")


def _reasons():
    from courtside.viz import dashboard_data as D
    from courtside.viz import story as S
    data = D.load(PROCESSED, ROOT / "reports" / "analysis.json")
    meta = data.analysis.get("meta", {})
    now, prev = meta.get("season", data.seasons[0]), meta.get("comparison_season", data.seasons[-1])
    if now not in data.attribution or prev not in data.attribution:
        pytest.skip("no comparison season")
    sh = D.season_halves(data.game_halves)
    return S.ranked_reasons(data.attribution[now], data.attribution[prev], sh.loc[now])


def test_new_problems_are_only_parts_that_cost_points():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120)
    at.session_state["_dashboard_page"] = "answer"
    at.run()
    assert not at.exception
    html = " ".join(m.value for m in at.markdown)
    i = html.find("New this season</div>")
    card = html[i:i + 300] if i >= 0 else ""
    for r in _reasons():
        if r.per_game >= 0:  # a part that still helps can't be a new problem
            assert r.name not in card, r.name
    captions = " ".join(c.value for c in at.caption)
    for r in _reasons():
        if r.per_game < 0 and r.per_game - r.prev_per_game >= 0.1:  # cheaper than last season
            name = r.name.split(",")[0].lower()
            assert f"No switch for {name}: it costs less" in captions or name not in captions.lower(), name
