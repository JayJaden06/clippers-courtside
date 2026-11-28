"""Players polish: face names avoid each other, and the card says when a player missed games."""

from __future__ import annotations

import pandas as pd

from courtside.web.views import players as P


def test_close_faces_get_different_spots_and_low_faces_go_above():
    spots = P.label_spots([0.06, 0.062, 0.03], [0.08, 0.081, 0.01], 0.2, 0.3)
    assert spots[0] != spots[1]
    assert spots[2] != "bottom center"


def test_played_note_says_since_or_missed():
    team = pd.DataFrame({"game_date": pd.date_range("2026-10-21", periods=18, freq="2D")})
    late = pd.DataFrame({"game_date": team["game_date"].iloc[-3:]})
    gaps = pd.DataFrame({"game_date": team["game_date"].iloc[::2]})
    assert P.played_note(late, team) == f" · played since {late['game_date'].min():%b} {late['game_date'].min().day}"
    assert P.played_note(gaps, team) == " · missed 9 of 18 games"
    assert P.played_note(team, team) == ""
