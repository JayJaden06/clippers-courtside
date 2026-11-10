"""Project settings: team, seasons, thresholds and paths.

Every value can be overridden with an environment variable prefixed with
``COURTSIDE_`` (for example ``COURTSIDE_LEAD_THRESHOLD=12``), which keeps
the sensitivity runs in the analysis from needing code changes.
"""

from __future__ import annotations

from datetime import date
from functools import lru_cache
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# NBA regular seasons open in the third or fourth week of October. From this date on, the calendar points at
# the new season; until its first game is actually played, ``settle_season`` keeps the site on the last one.
SEASON_SWITCH = (10, 21)


def season_for(day: date) -> str:
    """The NBA season a date falls in: 2026-11-10 -> '2026-27'; 2027-03-01 -> '2026-27'."""
    start = day.year if (day.month, day.day) >= SEASON_SWITCH else day.year - 1
    return f"{start}-{str(start + 1)[-2:]}"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="COURTSIDE_", extra="ignore")

    # Team and seasons
    team_id: int = 1610612746  # LA Clippers
    team_abbreviation: str = "LAC"
    # Follow the calendar unless set (COURTSIDE_CURRENT_SEASON); the comparison is the season before.
    current_season: str = Field(default_factory=lambda: season_for(date.today()))
    comparison_season: str = ""
    season_type: str = "Regular Season"
    # Seasons that also get per-quarter box scores; None means the current
    # season only. The comparison season gets full game + halves (~500 calls
    # instead of ~1,150).
    quarter_seasons_override: list[str] | None = Field(default=None, alias="quarter_seasons")

    # Blown-lead classification: a lead of at least this many points in the
    # first `lead_window_minutes` minutes (or at halftime) counts as "built a lead".
    lead_threshold: int = 8
    lead_window_minutes: int = 30
    sensitivity_thresholds: list[int] = Field(default_factory=lambda: [5, 8, 12])

    # The player the turnover hypothesis is about; looked up by name, never by id.
    focus_player: str = "Darius Garland"
    # The "What now?" page finds returning players on its own, from today's roster (see
    # viz.whatnow.returning). Names here are always added on top, e.g. a player worth watching who
    # doesn't meet that rule. Usually empty.
    watch_players: list[str] = Field(default_factory=list)

    # Companion project: did Intuit Dome (and The Wall) change the Clippers' home court?
    wall_study_url: str = "https://clippers-home-court-h46at8hpdebg9gdcaunm87.streamlit.app"

    # HTTP client
    request_timeout: int = 60
    request_sleep_seconds: float = 0.6
    max_retries: int = 5
    retry_backoff_seconds: float = 2.0  # first retry wait; doubles each attempt, capped at 60s

    # Paths
    data_dir: Path = PROJECT_ROOT / "data"
    reports_dir: Path = PROJECT_ROOT / "reports"

    @model_validator(mode="after")
    def _comparison_follows_current(self) -> "Settings":
        if not self.comparison_season:
            self.comparison_season = previous_season(self.current_season)
        return self

    @property
    def seasons(self) -> list[str]:
        return [self.current_season, self.comparison_season]

    @property
    def quarter_seasons(self) -> list[str]:
        return self.quarter_seasons_override or [self.current_season]

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def tableau_dir(self) -> Path:
        return self.data_dir / "tableau"

    @property
    def figures_dir(self) -> Path:
        return self.reports_dir / "figures"


@lru_cache
def get_settings() -> Settings:
    return Settings()


def previous_season(season: str) -> str:
    """'2025-26' -> '2024-25'."""
    start = int(season[:4]) - 1
    return f"{start}-{str(start + 1)[-2:]}"


def settle_season(settings: Settings, seasons_with_games: set[str]) -> bool:
    """Stay on the last season until the new one has a game: early in October, or between the date switch and
    opening night, the calendar's season has no games yet, and every page needs some. Returns True if it moved.
    """
    if settings.current_season in seasons_with_games:
        return False
    settings.current_season = previous_season(settings.current_season)
    settings.comparison_season = previous_season(settings.current_season)
    return True
