"""The per-run context every page and building block reads: the data, the seasons, the names, the page list.

Streamlit runs the app script once per visitor interaction, many visitors at a time, each in its own thread. Values
that are the same for everyone (the data) could be module globals, but some aren't: the On this page list grows as
a page draws its headings. So each run builds one ``Ctx`` and stores it in a ContextVar, which is private to the
thread doing the run, and the pages read it with ``current()``.
"""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from courtside.config import Settings
from courtside.viz.dashboard_data import DashboardData


@dataclass
class Ctx:
    settings: Settings
    data: DashboardData
    meta: dict
    season_now: str
    season_prev: str
    focus: str  # the player the turnover question is about, full name
    focus_last: str  # his short name
    player_ids: dict[str, int]  # every player name we can show -> NBA player id (for photos)
    team_last: dict[str, int]  # the team's short names -> player id (to find faces in card titles)
    toc: list[tuple[str, str]] = field(default_factory=list)  # (label, anchor) of this page's sections, in order
    pages: dict[str, Any] = field(default_factory=dict)  # "replay", "answer", ... -> st.Page, for links


_CTX: ContextVar[Ctx] = ContextVar("courtside_ctx")


def use(ctx: Ctx) -> None:
    """Make ``ctx`` the context for the rest of this run."""
    _CTX.set(ctx)


def current() -> Ctx:
    """This run's context (set by the app script before any page draws)."""
    return _CTX.get()
