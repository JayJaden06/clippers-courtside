"""The shot map's data: every shot accounted for, zones in order."""

from __future__ import annotations

from pathlib import Path

import pytest

from courtside.viz import dashboard_data as D

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
pytestmark = pytest.mark.skipif(not (PROCESSED / "shots.parquet").exists(), reason="run courtside build")


@pytest.fixture(scope="module")
def data() -> D.DashboardData:
    return D.load(PROCESSED, ROOT / "reports" / "analysis.json")


def test_shot_map_counts_match_the_box_scores(data):
    if data.shots.empty:
        pytest.skip("no shot data")
    s = data.shots[data.shots["half"].isin(["H1", "H2"])].groupby(["game_id", "half"]).size()
    gh = data.game_halves[data.game_halves["split"].astype(str).isin(["H1", "H2"])]
    fga = gh.assign(half=gh["split"].astype(str)).set_index(["game_id", "half"])["fga"]
    both = s.to_frame("shots").join(fga, how="inner")
    assert len(both) > 0 and (both["shots"] == both["fga"]).all()
    z = D.shot_zones(data.shots, data.seasons[0])
    assert list(z.index) == [x for x in D.ZONE_ORDER if x in z.index]
    assert int(z[["size_h1", "size_h2"]].to_numpy().sum()) == int(
        fga.loc[fga.index.get_level_values(0).isin(
            data.shots.loc[data.shots["season"] == data.seasons[0], "game_id"])].sum())
