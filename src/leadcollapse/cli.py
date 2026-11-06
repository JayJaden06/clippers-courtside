"""Command-line entry point: ``leadcollapse pull`` then ``leadcollapse build``.

``pull`` is the only command that talks to stats.nba.com. Every later step
uses an offline client, so it can only read what ``pull`` already cached.
"""

from __future__ import annotations

import logging
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.logging import RichHandler

from leadcollapse.config import get_settings
from leadcollapse.ingest import games
from leadcollapse.nba_client import NBAClient

app = typer.Typer(help="Why do the Clippers lose second-half leads?", no_args_is_help=True)
console = Console()

SeasonOpt = Annotated[
    Optional[list[str]],
    typer.Option("--season", "-s", help="Season like 2026-27; repeatable. Default: both configured seasons."),
]


@app.callback()
def main(verbose: Annotated[bool, typer.Option("--verbose", "-v", help="Log every cache hit.")] = False) -> None:
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(message)s",
        handlers=[RichHandler(console=console, show_path=False)],
    )


@app.command()
def pull(season: SeasonOpt = None) -> None:
    """Download and cache raw stats.nba.com data (current season's log is refreshed)."""
    settings = get_settings()
    client = NBAClient(settings)
    df = games.load_games(client, settings, season, refresh_current=True)
    for s, n in df.groupby("season").size().items():
        console.print(f"[bold]{s}[/]: {n} completed games")
    console.print(f"pull done: {client.stats}")


@app.command()
def build(season: SeasonOpt = None) -> None:
    """Build processed tables from the cache (no network)."""
    settings = get_settings()
    client = NBAClient(settings, offline=True)
    df = games.load_games(client, settings, season)
    games.write_games(df, settings)
    console.print(f"wrote {len(df)} games to {settings.processed_dir / 'games.parquet'}")
