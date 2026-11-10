"""Hypothesis tests on the committed processed tables.

These do not pin today's numbers (they change after every game). They check
that the analysis is internally consistent, carries its caveats, and that the
verdicts follow the data: in particular, that H3 *can* conclude "it's Garland"
when the data say so.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from courtside.analysis import hypotheses as H

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"
pytestmark = pytest.mark.skipif(not (PROCESSED / "game_halves.parquet").exists(), reason="run courtside build")


@pytest.fixture(scope="module")
def inp() -> H.Inputs:
    return H.load_inputs(PROCESSED, "2026-27", "2025-26", "Darius Garland")


@pytest.fixture(scope="module")
def results(inp):
    return {r.id: r for r in H.run_all(inp)}


def test_every_result_has_verdict_caveat_and_tables(results):
    assert set(results) == {"H1", "H2", "H3", "H4", "H5", "ATTR", "SEASONS"}
    for r in results.values():
        assert r.verdict and r.tables and r.chart
        assert "Sample:" in r.verdict or r.id == "ATTR"
        assert "nan" not in r.verdict.lower()


def test_attribution_adds_up_to_the_swing(inp):
    for season in (inp.season, inp.comparison_season):
        att = H.attribution(inp, season)
        assert att["points"].sum() == pytest.approx(att.attrs["swing"])
        assert att["share_of_swing"].sum() == pytest.approx(1.0)


def test_attribution_is_ranked_by_size(inp):
    pts = H.attribution(inp)["points"].abs().tolist()
    assert pts == sorted(pts, reverse=True)


@pytest.mark.parametrize("n, s", [(1, "1st"), (2, "2nd"), (3, "3rd"), (4, "4th"), (11, "11th"), (12, "12th"),
                                  (13, "13th"), (21, "21st"), (22, "22nd"), (23, "23rd"), (30, "30th")])
def test_ordinal(n, s):
    assert H.ordinal(n) == s


def test_share_words():
    assert H._share_word(0.5) == "drives"
    assert H._share_word(0.2) == "contributes to"
    assert H._share_word(0.05) == "is a minor part of"


def test_garland_verdict_flips_when_his_turnovers_explode(inp):
    # Triple his second-half turnovers this season. If the verdict were written to
    # match an expected answer rather than the data, it would not change.
    ps = inp.player_stats.copy()
    mask = (ps["season"] == inp.season) & (ps["split"] == "H2") & (ps["player_name"] == inp.focus_player)
    ps.loc[mask, "tov"] *= 3
    flipped = H.h3_garland(H.Inputs(**{**inp.__dict__, "player_stats": ps}))
    assert flipped.key["share_of_swing"] >= H.DRIVES
    assert flipped.verdict.startswith("Yes")

    actual = H.h3_garland(inp)
    assert actual.verdict.split(":")[0] == ("Yes" if actual.key["share_of_swing"] >= H.DRIVES else "Partly")


def test_h1_answer_follows_rank_and_interval(results):
    k = results["H1"].key
    expected = ("Yes" if k["net_change_rank"] >= H.WORST_RANKS and k["boot_hi"] < 0
                else "Probably" if k["net_change_rank"] >= H.WORST_RANKS else "Not clearly")
    assert results["H1"].verdict.startswith(expected)


def test_write_results(tmp_path, results):
    import json

    H.write_results(list(results.values()), tmp_path / "tables", tmp_path / "analysis.json", {"season": "x"})
    data = json.loads((tmp_path / "analysis.json").read_text())
    assert [r["id"] for r in data["results"]] == list(results)
    assert (tmp_path / "tables" / "attribution.csv").exists()
