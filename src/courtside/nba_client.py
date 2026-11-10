"""Thin wrapper over ``nba_api`` with browser headers, throttling, retries and a disk cache.

Every successful response is written to ``data/raw/<Endpoint>/<params-hash>.json``
before it is returned, so re-running any step of the pipeline costs zero API
calls once the data has been pulled. The cache key is the endpoint class name
plus the sorted keyword arguments, so the same request always maps to the same
file regardless of argument order.

Usage::

    from nba_api.stats.endpoints import LeagueGameLog
    client = NBAClient()
    raw = client.get(LeagueGameLog, season="2026-27", player_or_team_abbreviation="T")
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

import requests
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from courtside.config import Settings, get_settings

log = logging.getLogger(__name__)

# Browser-like headers. stats.nba.com drops or stalls requests that do not look
# like they come from nba.com in a browser.
NBA_HEADERS: dict[str, str] = {
    "Host": "stats.nba.com",
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Origin": "https://www.nba.com",
    "Referer": "https://www.nba.com/",
    "Pragma": "no-cache",
    "Cache-Control": "no-cache",
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-site",
}

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class TransientAPIError(Exception):
    """Rate limit, server error or a truncated body: worth retrying."""


class PermanentAPIError(Exception):
    """A 4xx (other than 429): retrying will not help, the request is wrong."""


class CacheMissError(Exception):
    """Raised in offline mode when a request is not in the cache."""


@dataclass
class CacheStats:
    hits: int = 0
    fetches: int = 0

    def __str__(self) -> str:
        return f"{self.hits} cache hits, {self.fetches} API calls"


def cache_key(endpoint_name: str, params: dict[str, Any]) -> str:
    """Stable 16-hex-character hash of an endpoint name and its parameters."""
    payload = json.dumps({"endpoint": endpoint_name, "params": params}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


class NBAClient:
    """Fetches stats.nba.com endpoints through ``nba_api`` with a read-through cache.

    Args:
        settings: project settings; defaults to the global settings.
        cache_dir: overrides ``settings.raw_dir`` (tests point this at a tmp dir).
        offline: if True, never touch the network and raise ``CacheMissError``
            for anything not cached. Used by ``build``/``analyze`` so that
            transforms can only ever read what ``pull`` already downloaded.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        cache_dir: Path | None = None,
        offline: bool = False,
    ) -> None:
        self.settings = settings or get_settings()
        self.cache_dir = Path(cache_dir) if cache_dir else self.settings.raw_dir
        self.offline = offline
        self.stats = CacheStats()
        self._last_request_at: float | None = None
        self._refresh_default = False

    @contextmanager
    def refreshing(self, enabled: bool = True) -> Iterator[None]:
        """Within the block, every ``get`` behaves as if ``refresh=enabled`` was passed.

        For season-to-date endpoints (dashboards, game logs) of the current season,
        whose responses change after every game.
        """
        previous, self._refresh_default = self._refresh_default, enabled
        try:
            yield
        finally:
            self._refresh_default = previous

    # ------------------------------------------------------------------ cache
    def cache_path(self, endpoint_name: str, params: dict[str, Any]) -> Path:
        return self.cache_dir / endpoint_name / f"{cache_key(endpoint_name, params)}.json"

    def _read_cache(self, path: Path) -> dict[str, Any] | None:
        if not path.exists():
            return None
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)["response"]

    def _write_cache(self, path: Path, endpoint_name: str, params: dict[str, Any], response: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        envelope = {
            "endpoint": endpoint_name,
            "params": params,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "response": response,
        }
        # Write to a temp file then rename so an interrupted run never leaves
        # a half-written JSON file that later reads as a cache hit.
        tmp = path.with_suffix(".json.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(envelope, fh)
        tmp.replace(path)

    # ---------------------------------------------------------------- network
    def _throttle(self) -> None:
        if self._last_request_at is None:
            return
        wait = self.settings.request_sleep_seconds - (time.monotonic() - self._last_request_at)
        if wait > 0:
            time.sleep(wait)

    def _fetch(self, endpoint_cls: Callable[..., Any], params: dict[str, Any]) -> dict[str, Any]:
        @retry(
            retry=retry_if_exception_type(
                (TransientAPIError, requests.exceptions.Timeout, requests.exceptions.ConnectionError)
            ),
            wait=wait_exponential(
                multiplier=self.settings.retry_backoff_seconds,
                min=self.settings.retry_backoff_seconds,
                max=60,
            ),
            stop=stop_after_attempt(self.settings.max_retries),
            before_sleep=before_sleep_log(log, logging.WARNING),
            reraise=True,
        )
        def attempt() -> dict[str, Any]:
            self._throttle()
            try:
                endpoint = endpoint_cls(
                    **params, headers=NBA_HEADERS, timeout=self.settings.request_timeout
                )
            finally:
                self._last_request_at = time.monotonic()
            response = endpoint.nba_response
            status = response._status_code
            if status in RETRYABLE_STATUS:
                raise TransientAPIError(f"HTTP {status} from {response.get_url()}")
            if status is not None and status >= 400:
                raise PermanentAPIError(f"HTTP {status} from {response.get_url()}")
            if not response.valid_json():
                raise TransientAPIError(f"Non-JSON body from {response.get_url()}")
            return response.get_dict()

        return attempt()

    # ------------------------------------------------------------------ public
    def get(
        self, endpoint_cls: Callable[..., Any], *, refresh: bool = False, **params: Any
    ) -> dict[str, Any]:
        """Return the raw JSON dict for an endpoint call, from cache when possible.

        ``refresh=True`` re-downloads and overwrites the cached copy. Use it for
        responses that change over time (the current season's game log); a
        finished game's box score or play-by-play never needs it. Offline
        clients ignore it and read the cache.
        """
        name = endpoint_cls.__name__
        path = self.cache_path(name, params)

        refresh = refresh or self._refresh_default
        cached = None if (refresh and not self.offline) else self._read_cache(path)
        if cached is not None:
            self.stats.hits += 1
            log.info("cache hit  %s %s", name, params)
            return cached

        if self.offline:
            raise CacheMissError(f"{name} {params} is not cached; run `courtside pull` first")

        log.info("fetching   %s %s", name, params)
        response = self._fetch(endpoint_cls, params)
        self._write_cache(path, name, params, response)
        self.stats.fetches += 1
        return response
