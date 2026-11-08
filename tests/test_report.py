"""Figures and findings.md, rendered from the committed processed tables."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
import pytest

from leadcollapse.analysis import hypotheses as H
from leadcollapse.analysis import report as R

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
needs_data = pytest.mark.skipif(not (PROCESSED / "game_halves.parquet").exists(), reason="run leadcollapse build")

META = {"analyzed_on": "2026-11-08", "lead_threshold": 8}


def test_md_table_formats_rows_and_columns():
    by_col = pd.DataFrame({"name": ["a", "b"], "n": [3.0, 12.0], "rate": [0.12345, 0.5], "pts": [-49.14, 2.0]})
    assert R.md_table(by_col).splitlines()[2:] == ["| a | 3 | 0.123 | -49.1 |", "| b | 12 | 0.500 | 2.0 |"]
    by_row = pd.DataFrame({"metric": ["Fouls", "FT rate"], "H1": [55.0, 0.196], "H2": [100.0, 0.383]})
    assert R.md_table(by_row).splitlines()[2:] == ["| Fouls | 55 | 100 |", "| FT rate | 0.196 | 0.383 |"]


def test_md_table_drops_overtime_and_escapes_pipes():
    df = pd.DataFrame({"split": ["H1", "OT"], "label": ["a|b", "c"]})
    lines = R.md_table(df).splitlines()
    assert len(lines) == 3 and "a\\|b" in lines[2]


@pytest.fixture(scope="module")
def inp():
    return H.load_inputs(PROCESSED, "2026-27", "2025-26", "Darius Garland")


@pytest.fixture(scope="module")
def results(inp):
    return H.run_all(inp)


@pytest.fixture(scope="module")
def written(tmp_path_factory, inp, results):
    out = tmp_path_factory.mktemp("reports")
    margins = pd.read_parquet(PROCESSED / "margin_series.parquet")
    path = R.write_report(results, inp, META, margins, out)
    return out, path.read_text(encoding="utf-8")


@needs_data
def test_report_has_every_section_and_image(written, results):
    out, text = written
    for heading in ["## Executive summary", "## Methodology", "## Limitations"]:
        assert heading in text
    for r in results:
        assert f"## {r.id}. {r.title}" in text
    images = re.findall(r"\]\(figures/([\w.]+\.png)\)", text)
    assert {r.chart for r in results} <= set(images)
    for name in images:
        assert (out / "figures" / name).stat().st_size > 10_000, name
    assert not re.search(r"\bnan\b", text, re.IGNORECASE)


@needs_data
def test_executive_summary_length_and_garland_answer(inp, results):
    by_id = {r.id: r for r in results}
    sentences = R.executive_summary(by_id, inp, 8)
    assert 5 <= len(sentences) <= 8
    garland = next(s for s in sentences if "Garland" in s)
    main = by_id["H3"].key["share_of_swing"] >= H.DRIVES
    assert garland.startswith("The starting hypothesis holds up" if main else "It is not mainly Darius Garland")


@needs_data
def test_summary_follows_the_flip(inp):
    # Same flip as test_hypotheses: if Garland's turnovers triple, the summary must say so.
    ps = inp.player_stats.copy()
    mask = (ps["season"] == inp.season) & (ps["split"] == "H2") & (ps["player_name"] == inp.focus_player)
    ps.loc[mask, "tov"] *= 3
    flipped = H.Inputs(**{**inp.__dict__, "player_stats": ps})
    by_id = {r.id: r for r in H.run_all(flipped)}
    assert any(s.startswith("The starting hypothesis holds up") for s in R.executive_summary(by_id, flipped, 8))


@needs_data
def test_render_is_deterministic(inp, results):
    meta = {**META, "last_game": "2026-11-04"}
    assert R.render(results, inp, meta, tov_agreement=0.9) == R.render(results, inp, meta, tov_agreement=0.9)
