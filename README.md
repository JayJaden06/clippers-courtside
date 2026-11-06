# Why do the Clippers lose leads?

The Clippers keep building first-half leads and giving them back after halftime. This repo checks why with play-by-play, box score, lineup and league-wide data from stats.nba.com, comparing 2026-27 with 2025-26.

Work in progress.

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate        # macOS/Linux: source .venv/bin/activate
pip install -e ".[dev]"
pytest

leadcollapse pull    # download + cache stats.nba.com data (the only networked step)
leadcollapse build   # build data/processed/ tables from the cache, offline
```

All stats.nba.com responses are cached under `data/raw/` (gitignored), so each request is made once.
