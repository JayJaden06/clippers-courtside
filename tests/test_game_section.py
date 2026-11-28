"""Players, one player game by game: missed games say so, the table sorts as numbers, a season switch explains itself."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from courtside.web.views import players as P

ROOT = Path(__file__).resolve().parents[1]


def test_bars_mark_missed_and_clean_games():
    team = pd.DataFrame({"game_id": ["a", "b", "c"], "game_date": pd.to_datetime(["2026-11-01", "2026-11-03",
                                                                                    "2026-11-05"]),
                         "opponent_abbreviation": ["PHX", "NOP", "UTA"]})
    g = pd.DataFrame({"game_id": ["a", "c"], "tov_h1": [2, 0], "tov_h2": [1, 0]})
    fig = P.turnover_bars(g, team)
    words = {t.text: list(t.x) for t in fig.data if t.type == "scatter"}
    assert words == {"out": ["Nov 3 NOP"], "0": ["Nov 5 UTA"]}
    assert fig.layout.yaxis.dtick == 1


def test_table_rates_are_numbers():
    g = pd.DataFrame({"game_date": ["2026-11-01"], "opponent_abbreviation": ["PHX"], "result": ["W"],
                      "final_margin": [5], "lead_class": ["Held lead"], "minutes_h1": [18.0], "minutes_h2": [17.0],
                      "tov_h1": [1], "tov_h2": [2], "tov_live_h2": [1], "tov_pct_h1": [0.07], "tov_pct_h2": [0.36]})
    t = P.game_table(g)
    assert pd.api.types.is_numeric_dtype(t["Plays ending in a turnover, 2nd half"])
    assert t["Plays ending in a turnover, 2nd half"].iloc[0] == 36


def test_season_switch_explains_the_new_player():
    testing = pytest.importorskip("streamlit.testing.v1")
    at = testing.AppTest.from_file(str(ROOT / "dashboard" / "app.py"), default_timeout=120)
    at.session_state["_dashboard_page"] = "players"
    at.run()
    now, old = at.selectbox(key="player_season").options[0], at.selectbox(key="player_season").options[-1]
    gone = [n for n in at.selectbox(key="player_pick").options]
    at.selectbox(key="player_season").set_value(old).run()
    then = set(at.selectbox(key="player_pick").options)
    missing = [n for n in gone if n not in then]
    if not missing:
        pytest.skip(f"every {now} player also played in {old}")
    at.selectbox(key="player_season").set_value(now).run()
    at.selectbox(key="player_pick").set_value(missing[0]).run()
    at.selectbox(key="player_season").set_value(old).run()
    assert not at.exception
    assert any(f"{missing[0]} didn't play enough in {old}" in c.value for c in at.caption)
