"""The season follows the calendar, stays put until the new one has games, and old missing tracking stops being
asked for."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from courtside.config import Settings, season_for, settle_season
from courtside.ingest import tracking


def test_season_follows_the_calendar():
    assert season_for(date(2026, 11, 10)) == "2026-27"
    assert season_for(date(2027, 3, 1)) == "2026-27"
    assert season_for(date(2027, 6, 20)) == "2026-27"  # the Finals still belong to it
    assert season_for(date(2027, 10, 20)) == "2026-27"  # the day before the switch
    assert season_for(date(2027, 10, 21)) == "2027-28"


def test_comparison_is_the_season_before_unless_set():
    s = Settings(current_season="2027-28")
    assert s.comparison_season == "2026-27"
    assert Settings(current_season="2027-28", comparison_season="2024-25").comparison_season == "2024-25"


def test_stays_on_the_last_season_until_the_new_one_has_a_game():
    s = Settings(current_season="2027-28")
    assert settle_season(s, {"2026-27", "2025-26"})  # opening night hasn't happened
    assert (s.current_season, s.comparison_season) == ("2026-27", "2025-26")
    s = Settings(current_season="2027-28")
    assert not settle_season(s, {"2027-28", "2026-27"})  # one game in: switch over
    assert s.current_season == "2027-28"


class _Client:
    """Pretends to be the API client: one game cached, everything else fails to fetch."""

    def __init__(self, cached: set[str], root: Path):
        self.cached, self.root, self.asked = cached, root, []

    def cache_path(self, name: str, params: dict) -> Path:
        p = self.root / name / f"{params['game_id']}.json"
        if params["game_id"] in self.cached:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text("{}")
        return p

    def get(self, endpoint, **params):
        self.asked.append(params["game_id"])
        raise KeyError("not published")


def test_gives_up_on_tracking_a_week_after_the_game(tmp_path):
    games = pd.DataFrame({"game_id": ["old_missing", "old_cached", "recent"], "season": "2026-27",
                          "game_date": pd.to_datetime(["2026-10-25", "2026-10-26", "2026-11-08"])})
    client = _Client({"old_cached"}, tmp_path)
    _, _, skipped = tracking.load_tracking(client, games, today=pd.Timestamp("2026-11-10"))
    assert "gave-up:old_missing" in skipped  # 16 days old and never published: not asked again
    assert "old_missing" not in client.asked
    assert {"old_cached", "recent"} <= set(client.asked)  # cached ones read; recent ones still retried
    assert "recent" in skipped


def test_a_check_without_enough_data_steps_aside_instead_of_stopping_the_run():
    from courtside.analysis import hypotheses as H

    processed = Path(__file__).resolve().parents[1] / "data" / "processed"
    if not (processed / "game_halves.parquet").exists():
        return
    games = pd.read_parquet(processed / "games.parquet")
    season = games["season"].max()
    prior = sorted(games["season"].unique())[-2]
    # a focus player who hasn't played: the Garland check can't run, everything else still does
    results = H.run_all(H.load_inputs(processed, season, prior, "Nobody Here"))
    ids = {r.id for r in results}
    assert "H3" not in ids and {"H1", "H2", "ATTR"} <= ids
