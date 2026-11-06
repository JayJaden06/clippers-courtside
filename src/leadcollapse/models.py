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


# --------------------------------------------------------------------------- box scores


def parse_minutes(value: str | float | int | None) -> float:
    """Box-score minutes: ``"36:42"`` -> 36.7; blank (DNP) -> 0.0."""
    if value is None or value == "":
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    mins, _, secs = str(value).partition(":")
    return int(mins) + (int(secs) if secs else 0) / 60


class TraditionalStats(BaseModel):
    """The ``statistics`` block of a ``BoxScoreTraditionalV3`` player or team."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    minutes: float
    fgm: int = Field(alias="fieldGoalsMade", ge=0)
    fga: int = Field(alias="fieldGoalsAttempted", ge=0)
    fg3m: int = Field(alias="threePointersMade", ge=0)
    fg3a: int = Field(alias="threePointersAttempted", ge=0)
    ftm: int = Field(alias="freeThrowsMade", ge=0)
    fta: int = Field(alias="freeThrowsAttempted", ge=0)
    oreb: int = Field(alias="reboundsOffensive", ge=0)
    dreb: int = Field(alias="reboundsDefensive", ge=0)
    ast: int = Field(alias="assists", ge=0)
    stl: int = Field(alias="steals", ge=0)
    blk: int = Field(alias="blocks", ge=0)
    tov: int = Field(alias="turnovers", ge=0)
    pf: int = Field(alias="foulsPersonal", ge=0)
    pts: int = Field(alias="points", ge=0)
    plus_minus: float = Field(alias="plusMinusPoints", default=0.0)

    @field_validator("minutes", mode="before")
    @classmethod
    def _minutes(cls, v):
        return parse_minutes(v)

    @model_validator(mode="after")
    def _consistent(self) -> TraditionalStats:
        if self.fgm > self.fga or self.fg3m > self.fg3a or self.ftm > self.fta or self.fg3a > self.fga:
            raise ValueError(f"makes exceed attempts: {self}")
        if self.pts != 2 * self.fgm + self.fg3m + self.ftm:
            raise ValueError(f"points {self.pts} != 2*FGM + 3PM + FTM")
        return self


class AdvancedStats(BaseModel):
    """The ``statistics`` block of a ``BoxScoreAdvancedV3`` player or team (subset used here)."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    off_rating: float = Field(alias="offensiveRating")
    def_rating: float = Field(alias="defensiveRating")
    net_rating: float = Field(alias="netRating")
    efg_pct: float = Field(alias="effectiveFieldGoalPercentage")
    ts_pct: float = Field(alias="trueShootingPercentage")
    usg_pct: float = Field(alias="usagePercentage")
    tov_ratio: float = Field(alias="turnoverRatio")
    oreb_pct: float = Field(alias="offensiveReboundPercentage")
    dreb_pct: float = Field(alias="defensiveReboundPercentage")
    pace: float
    possessions: float
