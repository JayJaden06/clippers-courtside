# Clippers Courtside

Follows the Clippers' season game by game: a report on every game (who played well, what decided it, who guarded whom, what needed to happen), the odds and a game plan for every game coming up, and the analysis that started it: why they keep building first-half leads and giving them back after halftime. Built on play-by-play, box score, lineup, tracking and league-wide data from stats.nba.com.

Work in progress.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest

courtside pull    # download + cache stats.nba.com data (the only networked step)
courtside build   # build data/processed/ tables from the cache, offline
```

The first `pull` makes about 610 API calls (about 10 minutes, throttled to stay polite to stats.nba.com); later pulls only fetch new games.

All stats.nba.com responses are cached under `data/raw/` (gitignored), so each request is made once.
