import asyncio
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from resilient_httpx import (
    AsyncProxyHttpClient,
    ProxyHttpClient,
    RetryPolicy,
)
from resilient_httpx.pool import ProxyPool
from resilient_httpx.sync_pool import SyncProxyPool

URL = "https://api.example.com/data"
NO_WAIT = RetryPolicy(min_wait=0, max_wait=0)
PROXIES = ["http://proxy1:8080", "http://proxy2:8080", "http://proxy3:8080"]


def _response(status: int = 200) -> httpx.Response:
    return httpx.Response(status, text="ok", request=httpx.Request("GET", URL))


@contextmanager
def mock_async_httpx(side_effect):
    mock = AsyncMock(side_effect=side_effect)
    with (
        patch.object(httpx.AsyncClient, "request", mock),
        patch.object(httpx.AsyncClient, "aclose", new_callable=AsyncMock),
    ):
        yield mock


@contextmanager
def mock_sync_httpx(side_effect):
    mock = MagicMock(side_effect=side_effect)
    with (
        patch.object(httpx.Client, "request", mock),
        patch.object(httpx.Client, "close"),
    ):
        yield mock


async def test_pool_callbacks_fire_on_blacklist_and_unblacklist():
    blacklisted: list[str] = []
    unblacklisted: list[str] = []

    pool = ProxyPool(
        proxies=PROXIES,
        blacklist_threshold=2,
        blacklist_ttl=0.1,
        on_blacklist=blacklisted.append,
        on_unblacklist=unblacklisted.append,
    )

    await pool.report_failure("http://proxy1:8080")
    assert blacklisted == []
    await pool.report_failure("http://proxy1:8080")
    assert blacklisted == ["http://proxy1:8080"]

    await asyncio.sleep(0.3)
    await pool.get_proxy()
    assert unblacklisted == ["http://proxy1:8080"]


def test_sync_pool_callbacks_fire_on_blacklist_and_unblacklist():
    blacklisted: list[str] = []
    unblacklisted: list[str] = []
    import time

    pool = SyncProxyPool(
        proxies=PROXIES,
        blacklist_threshold=2,
        blacklist_ttl=0.1,
        on_blacklist=blacklisted.append,
        on_unblacklist=unblacklisted.append,
    )

    pool.report_failure("http://proxy1:8080")
    pool.report_failure("http://proxy1:8080")
    assert blacklisted == ["http://proxy1:8080"]

    time.sleep(0.3)
    pool.get_proxy()
    assert unblacklisted == ["http://proxy1:8080"]


async def test_pool_snapshot_fields():
    pool = ProxyPool(
        proxies=PROXIES,
        blacklist_threshold=2,
        blacklist_ttl=1.0,
    )
    await pool.report_success("http://proxy1:8080")
    await pool.report_success("http://proxy1:8080")
    await pool.report_failure("http://proxy2:8080")
    await pool.report_failure("http://proxy2:8080")

    snap = pool.snapshot()
    by_proxy = {row["proxy"]: row for row in snap}

    p1 = by_proxy["http://proxy1:8080"]
    assert p1["total_ok"] == 2
    assert p1["total_fail"] == 0
    assert p1["fail_count"] == 0
    assert p1["in_rotation"] is True
    assert p1["ttl_remaining_seconds"] is None

    p2 = by_proxy["http://proxy2:8080"]
    assert p2["total_fail"] == 2
    assert p2["fail_count"] == 2
    assert p2["in_rotation"] is False
    assert 0 < p2["ttl_remaining_seconds"] <= 1.0


async def test_async_client_snapshot_excludes_internal_all_pool():
    client = AsyncProxyHttpClient(
        proxies={"foreign": ["http://a:1"], "domestic": ["http://b:1"]},
        retry=NO_WAIT,
    )
    snap = client.blacklist_snapshot()
    assert set(snap.keys()) == {"foreign", "domestic"}


async def test_async_client_outcome_callback_receives_pool_and_status():
    outcomes: list[tuple] = []

    def hook(pool, proxy, status, duration):
        outcomes.append((pool, proxy, status, round(duration, 2) >= 0))

    with mock_async_httpx([_response(200), _response(500)] + [_response(500)] * 5):
        async with AsyncProxyHttpClient(
            proxies={"foreign": PROXIES},
            retry=NO_WAIT,
            on_request_outcome=hook,
        ) as client:
            await client.get(URL, pool="foreign")
            try:
                await client.get(URL, pool="foreign")
            except Exception:
                pass

    assert outcomes[0][0] == "foreign"
    assert outcomes[0][2] == "success"
    assert any(o[2] == "failed" for o in outcomes[1:])


def test_sync_client_outcome_callback_receives_pool_and_status():
    outcomes: list[tuple] = []

    def hook(pool, proxy, status, duration):
        outcomes.append((pool, proxy, status))

    with mock_sync_httpx([_response(200)]):
        with ProxyHttpClient(
            proxies={"foreign": PROXIES},
            retry=NO_WAIT,
            on_request_outcome=hook,
        ) as client:
            client.get(URL, pool="foreign")

    assert outcomes[0][0] == "foreign"
    assert outcomes[0][2] == "success"


async def test_one_dead_proxy_does_not_cascade_under_load():
    # Regression: with threshold=3, one dead proxy out of 30 used to cascade-block
    # the entire pool because TTL recovery would reset fail_count to threshold-1
    # → a single transient failure on the recovered proxy re-blacklisted it
    # immediately. Healthy proxies were not directly affected here (the simulator
    # only fails the dead one), but the BLACKLIST EVENT COUNT for the dead proxy
    # is the leading indicator: under the bug it grows roughly once per recovery,
    # while under the fix it grows roughly once per `threshold` failures.
    proxies = [f"http://proxy{i}:8080" for i in range(30)]
    dead = "http://proxy0:8080"

    blacklists: list[str] = []

    def on_bl(proxy: str) -> None:
        blacklists.append(proxy)

    threshold = 3
    pool = ProxyPool(
        proxies=proxies,
        blacklist_threshold=threshold,
        blacklist_ttl=0.05,
        on_blacklist=on_bl,
    )

    # 600 sequential operations. The dead proxy is selected ~600/30 = 20 times.
    # Each visit fails. With sleeps that let TTL expire, we cycle through
    # blacklist → recovery → blacklist. Under the fix: ~ceil(20/threshold) = ~7
    # blacklist events. Under the bug: every recovery → 1 failure → re-blacklist
    # → ~16+ blacklist events.
    n_iters = 600
    for i in range(n_iters):
        chosen = await pool.get_proxy()
        assert chosen is not None
        if chosen == dead:
            await pool.report_failure(dead)
        else:
            await pool.report_success(chosen)
        if i % 50 == 0:
            await asyncio.sleep(0.06)

    # Tight bound: dead proxy visits / threshold + (number of TTL recoveries).
    # Recoveries occur at most n_iters/50 = 12 times. So upper bound ~= 20/3 + 12 = ~18.
    # Pre-fix: would be much higher because each recovery triggered a fresh blacklist.
    assert len(blacklists) <= 12, f"too many blacklist events: {len(blacklists)}"
    # All 29 healthy proxies stayed in rotation.
    snap = pool.snapshot()
    healthy = [row for row in snap if row["proxy"] != dead]
    assert all(row["in_rotation"] for row in healthy)


async def test_probe_on_start_blacklists_dead_proxies():
    proxies = ["http://alive:1", "http://dead:1"]
    captured: list[str] = []

    client = AsyncProxyHttpClient(
        proxies={"pool": proxies},
        retry=NO_WAIT,
        on_blacklist=lambda p, proxy: captured.append(proxy),
    )

    real_get_client = client._get_client
    proxy_for_httpx_client: dict[int, str | None] = {}

    def stub_get_client(proxy):
        c = real_get_client(proxy)
        proxy_for_httpx_client[id(c)] = proxy

        async def fake_get(url, timeout=None):
            if proxy_for_httpx_client[id(c)] and "dead" in proxy_for_httpx_client[id(c)]:
                raise httpx.ConnectError("boom")
            return _response(200)

        c.get = fake_get
        return c

    client._get_client = stub_get_client

    await client.probe_all()

    snap = client.blacklist_snapshot()["pool"]
    by_proxy = {row["proxy"]: row for row in snap}
    assert by_proxy["http://dead:1"]["in_rotation"] is False
    assert by_proxy["http://alive:1"]["in_rotation"] is True
    assert captured == ["http://dead:1"]
