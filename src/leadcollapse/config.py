"""Project settings: team, seasons, thresholds and paths.

Every value can be overridden with an environment variable prefixed with
``LEADCOLLAPSE_`` (for example ``LEADCOLLAPSE_LEAD_THRESHOLD=12``), which keeps
the sensitivity runs in the analysis from needing code changes.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LEADCOLLAPSE_", extra="ignore")

    # Team and seasons
    team_id: int = 1610612746  # LA Clippers
    team_abbreviation: str = "LAC"
    current_season: str = "2026-27"
    comparison_season: str = "2025-26"
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

    # HTTP client
    request_timeout: int = 60
    request_sleep_seconds: float = 0.6
    max_retries: int = 5
    retry_backoff_seconds: float = 2.0  # first retry wait; doubles each attempt, capped at 60s

    # Paths
    data_dir: Path = PROJECT_ROOT / "data"
    reports_dir: Path = PROJECT_ROOT / "reports"

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
