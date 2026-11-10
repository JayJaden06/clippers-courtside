"""Every Clippers shot with where it was taken, for the shot map by half.

``ShotChartDetail`` with ``player_id=0`` returns all of a team's field-goal attempts for a season in
one call: the period, the spot on the floor (``LOC_X``/``LOC_Y`` in tenths of a foot, basket at 0,0),
the zone and whether it went in. Free throws aren't included.
"""

from __future__ import annotations

import pandas as pd
from nba_api.stats.endpoints import ShotChartDetail

from courtside.nba_client import NBAClient

# The league's basic zones, grouped into the four the map labels.
ZONES = {
    "Restricted Area": "At the rim",
    "In The Paint (Non-RA)": "Short range",
    "Mid-Range": "Mid-range",
    "Left Corner 3": "Three",
    "Right Corner 3": "Three",
    "Above the Break 3": "Three",
    "Backcourt": "Three",
}


def load_shots(client: NBAClient, team_id: int, season: str, season_type: str) -> pd.DataFrame:
    """One row per field-goal attempt: game, period, half, spot, zone, made."""
    raw = client.get(ShotChartDetail, team_id=team_id, player_id=0, season_nullable=season,
                     season_type_all_star=season_type, context_measure_simple="FGA")
    rs = next(r for r in raw["resultSets"] if r["name"] == "Shot_Chart_Detail")
    df = pd.DataFrame(rs["rowSet"], columns=rs["headers"])
    cols = ["season", "game_id", "period", "half", "player_id", "player_name", "loc_x", "loc_y", "distance",
            "zone", "is_three", "made"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    out = pd.DataFrame({
        "season": season,
        "game_id": df["GAME_ID"].astype(str),
        "period": df["PERIOD"].astype(int),
        "player_id": df["PLAYER_ID"].astype(int),
        "player_name": df["PLAYER_NAME"].astype(str),
        "loc_x": df["LOC_X"].astype(int),
        "loc_y": df["LOC_Y"].astype(int),
        "distance": df["SHOT_DISTANCE"].astype(int),
        "zone": df["SHOT_ZONE_BASIC"].map(ZONES).fillna("Mid-range"),
        "is_three": df["SHOT_TYPE"].astype(str).str.startswith("3"),
        "made": df["SHOT_MADE_FLAG"].astype(int).eq(1),
    })
    out["half"] = out["period"].map(lambda p: "H1" if p <= 2 else "H2" if p <= 4 else "OT")
    return out[cols]
