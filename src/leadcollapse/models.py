"""Pydantic models for records parsed at the ingest boundary.

Raw stats.nba.com payloads are validated into these models as soon as they are
read, so a schema change on the NBA side fails loudly in ``pull``/``build``
instead of silently producing wrong numbers downstream. After this boundary
the pipeline works in pandas.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

REGULATION_TEAM_MINUTES = 240  # 48 minutes x 5 players
OVERTIME_TEAM_MINUTES = 25  # 5 minutes x 5 players


class TeamGameLogRow(BaseModel):
    """One team's line for one game, as returned by ``LeagueGameLog`` (PlayerOrTeam=T)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    season_id: str = Field(alias="SEASON_ID")
    team_id: int = Field(alias="TEAM_ID")
    team_abbreviation: str = Field(alias="TEAM_ABBREVIATION")
    game_id: str = Field(alias="GAME_ID", pattern=r"^\d{10}$")
    game_date: date = Field(alias="GAME_DATE")
    matchup: str = Field(alias="MATCHUP")
    wl: Literal["W", "L"] | None = Field(alias="WL")
    minutes: int = Field(alias="MIN")
    pts: int = Field(alias="PTS")
    plus_minus: int = Field(alias="PLUS_MINUS")

    @field_validator("matchup")
    @classmethod
    def _matchup_has_separator(cls, v: str) -> str:
        if " vs. " not in v and " @ " not in v:
            raise ValueError(f"unrecognised MATCHUP format: {v!r}")
        return v

    @property
    def is_home(self) -> bool:
        # "LAC vs. SAC" is a home game, "LAC @ LAL" is a road game.
        return " vs. " in self.matchup

    @property
    def is_final(self) -> bool:
        return self.wl is not None


class Game(BaseModel):
    """A completed game from the point of view of the team being analysed."""

    model_config = ConfigDict(frozen=True)

    game_id: str = Field(pattern=r"^\d{10}$")
    season: str = Field(pattern=r"^\d{4}-\d{2}$")
    game_date: date
    team_id: int
    team_abbreviation: str
    opponent_id: int
    opponent_abbreviation: str
    is_home: bool
    result: Literal["W", "L"]
    team_pts: int = Field(ge=0)
    opp_pts: int = Field(ge=0)
    overtime_periods: int = Field(ge=0)

    @property
    def margin(self) -> int:
        return self.team_pts - self.opp_pts

    @model_validator(mode="after")
    def _result_matches_score(self) -> Game:
        if self.team_pts == self.opp_pts:
            raise ValueError(f"{self.game_id}: tied final score {self.team_pts}-{self.opp_pts}")
        expected = "W" if self.team_pts > self.opp_pts else "L"
        if self.result != expected:
            raise ValueError(
                f"{self.game_id}: result {self.result} contradicts score {self.team_pts}-{self.opp_pts}"
            )
        return self

    @classmethod
    def from_rows(cls, team: TeamGameLogRow, opponent: TeamGameLogRow, season: str) -> Game:
        """Combine a team's game-log row with its opponent's row for the same game."""
        if team.game_id != opponent.game_id:
            raise ValueError(f"rows are from different games: {team.game_id} vs {opponent.game_id}")
        if team.plus_minus != team.pts - opponent.pts:
            raise ValueError(
                f"{team.game_id}: PLUS_MINUS {team.plus_minus} != {team.pts} - {opponent.pts}"
            )
        extra = team.minutes - REGULATION_TEAM_MINUTES
        if extra < 0 or extra % OVERTIME_TEAM_MINUTES:
            raise ValueError(f"{team.game_id}: unexpected team minutes {team.minutes}")
        assert team.wl is not None
        return cls(
            game_id=team.game_id,
            season=season,
            game_date=team.game_date,
            team_id=team.team_id,
            team_abbreviation=team.team_abbreviation,
            opponent_id=opponent.team_id,
            opponent_abbreviation=opponent.team_abbreviation,
            is_home=team.is_home,
            result=team.wl,
            team_pts=team.pts,
            opp_pts=opponent.pts,
            overtime_periods=extra // OVERTIME_TEAM_MINUTES,
        )
