"""Client tests: caching, offline mode and retry behaviour, with no network access.

A fake endpoint class stands in for an ``nba_api`` endpoint. It mimics the
one piece of the interface the client uses: construct with params, then read
``.nba_response`` (status code, JSON validity, dict body).
"""

from __future__ import annotations

import json

import pytest
import requests

from leadcollapse.config import Settings
from leadcollapse.nba_client import (
    CacheMissError,
    NBAClient,
    PermanentAPIError,
    TransientAPIError,
    cache_key,
)


class FakeResponse:
    def __init__(self, status: int, body: dict | None):
        self._status_code = status
        self._body = body

    def valid_json(self) -> bool:
        return self._body is not None

    def get_dict(self) -> dict:
        assert self._body is not None
        return self._body

    def get_url(self) -> str:
        return "https://stats.nba.com/stats/fake"


def make_endpoint(script: list):
    """Build a fake endpoint class that plays back ``script`` one call at a time.

    Each script item is either an exception instance (raised on construction)
    or a ``(status, body)`` tuple.
    """
    calls: list[dict] = []

    class FakeEndpoint:
        def __init__(self, headers=None, timeout=None, **params):
            calls.append({"params": params, "headers": headers, "timeout": timeout})
            step = script[min(len(calls), len(script)) - 1]
            if isinstance(step, Exception):
                raise step
            self.nba_response = FakeResponse(*step)

    FakeEndpoint.calls = calls
    return FakeEndpoint


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        data_dir=tmp_path / "data",
        request_sleep_seconds=0,
        retry_backoff_seconds=0,
        max_retries=3,
    )


@pytest.fixture
def client(settings) -> NBAClient:
    return NBAClient(settings=settings)


BODY = {"resultSets": [{"headers": ["GAME_ID"], "rowSet": [["0022600001"]]}]}


def test_cache_key_ignores_argument_order():
    assert cache_key("X", {"a": 1, "b": 2}) == cache_key("X", {"b": 2, "a": 1})
    assert cache_key("X", {"a": 1}) != cache_key("Y", {"a": 1})
    assert cache_key("X", {"a": 1}) != cache_key("X", {"a": 2})


def test_second_call_is_served_from_cache(client):
    endpoint = make_endpoint([(200, BODY)])
    first = client.get(endpoint, season="2026-27")
    second = client.get(endpoint, season="2026-27")

    assert first == second == BODY
    assert len(endpoint.calls) == 1
    assert (client.stats.hits, client.stats.fetches) == (1, 1)


def test_cache_file_layout_and_envelope(client):
    endpoint = make_endpoint([(200, BODY)])
    client.get(endpoint, season="2026-27")

    path = client.cache_path("FakeEndpoint", {"season": "2026-27"})
    assert path.parent.name == "FakeEndpoint"
    envelope = json.loads(path.read_text(encoding="utf-8"))
    assert envelope["params"] == {"season": "2026-27"}
    assert envelope["response"] == BODY
    assert "fetched_at" in envelope
    assert not list(path.parent.glob("*.tmp"))


def test_headers_and_timeout_are_passed_through(client, settings):
    endpoint = make_endpoint([(200, BODY)])
    client.get(endpoint, season="2026-27")

    call = endpoint.calls[0]
    assert call["timeout"] == settings.request_timeout
    assert call["headers"]["Referer"] == "https://www.nba.com/"
    assert call["params"] == {"season": "2026-27"}


def test_offline_mode_raises_on_miss_and_reads_hits(settings):
    endpoint = make_endpoint([(200, BODY)])
    NBAClient(settings=settings).get(endpoint, season="2026-27")

    offline = NBAClient(settings=settings, offline=True)
    assert offline.get(endpoint, season="2026-27") == BODY
    with pytest.raises(CacheMissError):
        offline.get(endpoint, season="2025-26")
    assert len(endpoint.calls) == 1


def test_refresh_refetches_and_overwrites_cache(client, settings):
    newer = {"resultSets": [{"headers": ["GAME_ID"], "rowSet": [["0022600001"], ["0022600002"]]}]}
    endpoint = make_endpoint([(200, BODY), (200, newer)])
    client.get(endpoint, season="2026-27")

    assert client.get(endpoint, season="2026-27", refresh=True) == newer
    assert client.get(endpoint, season="2026-27") == newer  # cache now holds the newer copy
    assert len(endpoint.calls) == 2
    # The refresh flag is not an endpoint parameter and must not leak into it.
    assert endpoint.calls[1]["params"] == {"season": "2026-27"}

    offline = NBAClient(settings=settings, offline=True)
    assert offline.get(endpoint, season="2026-27", refresh=True) == newer
    assert len(endpoint.calls) == 2


@pytest.mark.parametrize(
    "failure",
    [
        (429, None),
        (503, None),
        (200, None),  # truncated / non-JSON body
        requests.exceptions.ReadTimeout("read timed out"),
        requests.exceptions.ConnectionError("reset"),
    ],
    ids=["429", "503", "bad-json", "timeout", "conn-reset"],
)
def test_transient_failures_are_retried(client, failure):
    endpoint = make_endpoint([failure, (200, BODY)])
    assert client.get(endpoint, season="2026-27") == BODY
    assert len(endpoint.calls) == 2


def test_gives_up_after_max_retries_and_caches_nothing(client, settings):
    endpoint = make_endpoint([(429, None)])
    with pytest.raises(TransientAPIError):
        client.get(endpoint, season="2026-27")
    assert len(endpoint.calls) == settings.max_retries
    assert not client.cache_path("FakeEndpoint", {"season": "2026-27"}).exists()


def test_client_errors_are_not_retried(client):
    endpoint = make_endpoint([(400, None), (200, BODY)])
    with pytest.raises(PermanentAPIError):
        client.get(endpoint, season="bad")
    assert len(endpoint.calls) == 1


def test_refreshing_block_refetches_and_restores(client):
    endpoint = make_endpoint([(200, BODY)])
    client.get(endpoint, season="2026-27")
    with client.refreshing():
        client.get(endpoint, season="2026-27")
        with client.refreshing(False):
            client.get(endpoint, season="2026-27")  # nested opt-out reads the cache
    client.get(endpoint, season="2026-27")  # back to normal after the block
    assert len(endpoint.calls) == 2
