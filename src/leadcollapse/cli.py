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
from leadcollapse.ingest import boxscores, games, playbyplay
from leadcollapse.nba_client import NBAClient
from leadcollapse.transform import halves, leads

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
    console.print(f"game logs: {client.stats}")
    boxscores.load_boxscores(client, df, settings.quarter_seasons)
    playbyplay.load_pbp(client, df)
    console.print(f"pull done (game logs, box scores, play-by-play): {client.stats}")


@app.command()
def build(season: SeasonOpt = None) -> None:
    """Build processed tables from the cache (no network)."""
    settings = get_settings()
    client = NBAClient(settings, offline=True)
    df = games.load_games(client, settings, season)
    games.write_games(df, settings)
    console.print(f"games: {len(df)} rows")

    team_box, player_box = boxscores.load_boxscores(client, df, settings.quarter_seasons)
    gaps = boxscores.check_split_totals(team_box)
    boxscores.write_boxscores(team_box, player_box, settings.processed_dir)
    gaps.to_csv(settings.processed_dir / "split_discrepancies.csv", index=False)
    console.print(
        f"box scores: {len(team_box)} team rows, {len(player_box)} player rows; "
        f"split totals OK ({len(gaps)} small stat corrections logged to split_discrepancies.csv)"
    )

    game_halves = halves.build_game_halves(team_box, df, settings.team_id)
    halves.write_game_halves(game_halves, settings.processed_dir)
    settings.tableau_dir.mkdir(parents=True, exist_ok=True)
    floats = game_halves.select_dtypes("float").columns
    game_halves.assign(split=game_halves["split"].astype(str)).round({c: 4 for c in floats}).to_csv(
        settings.tableau_dir / "game_halves_wide.csv", index=False
    )
    console.print(f"game halves: {len(game_halves)} rows (+ tableau/game_halves_wide.csv)")

    events = playbyplay.load_pbp(client, df)
    playbyplay.check_final_scores(events, df)
    series = leads.margin_series(events)
    game_leads = leads.game_leads(
        series, df, settings.lead_threshold, settings.lead_window_minutes, settings.sensitivity_thresholds
    )
    leads.check_halftime_margins(game_leads, game_halves)
    leads.write_leads(series, game_leads, settings.processed_dir)
    console.print(
        f"play-by-play: {len(events)} events; final scores and halftime margins match box scores; "
        f"{len(game_leads)} games classified"
    )
    console.print(f"wrote to {settings.processed_dir}")
