import asyncio

import pytest

from resilient_httpx.pool import ProxyPool


@pytest.fixture
def pool(proxy_list):
    return ProxyPool(proxies=proxy_list, blacklist_threshold=3, blacklist_ttl=10.0)


async def test_round_robin_order(pool, proxy_list):
    results = [await pool.get_proxy() for _ in range(6)]
    assert results == proxy_list * 2


async def test_random_strategy(proxy_list):
    pool = ProxyPool(proxies=proxy_list, strategy="random")
    result = await pool.get_proxy()
    assert result in proxy_list


async def test_blacklist_after_threshold(pool):
    for _ in range(3):
        await pool.report_failure("http://proxy1:8080")
    result = await pool.get_proxy()
    assert result == "http://proxy2:8080"


async def test_report_failure_returns_blacklisted_flag(pool):
    assert await pool.report_failure("http://proxy1:8080") is False
    assert await pool.report_failure("http://proxy1:8080") is False
    assert await pool.report_failure("http://proxy1:8080") is True


async def test_all_blacklisted_returns_none(pool):
    for proxy in ["http://proxy1:8080", "http://proxy2:8080", "http://proxy3:8080"]:
        for _ in range(3):
            await pool.report_failure(proxy)
    assert await pool.get_proxy() is None


async def test_ttl_recovery(proxy_list):
    pool = ProxyPool(
        proxies=proxy_list, blacklist_threshold=2, blacklist_ttl=0.1,
    )
    for _ in range(2):
        await pool.report_failure("http://proxy1:8080")

    assert await pool.get_proxy() == "http://proxy2:8080"

    await _sleep(0.15)

    results = [await pool.get_proxy() for _ in range(3)]
    assert "http://proxy1:8080" in results


async def test_fail_count_reset_after_recovery(proxy_list):
    # After TTL, a recovered proxy must start with fail_count=0, not threshold-1.
    # Otherwise a single transient failure re-blacklists it immediately under load.
    pool = ProxyPool(
        proxies=proxy_list, blacklist_threshold=3, blacklist_ttl=0.1,
    )
    for _ in range(3):
        await pool.report_failure("http://proxy1:8080")

    await _sleep(0.15)

    await pool.get_proxy()
    state = pool._state["http://proxy1:8080"]
    assert state.fail_count == 0
    assert state.blacklisted_until is None

    # One more transient failure must NOT immediately re-blacklist.
    assert await pool.report_failure("http://proxy1:8080") is False
    assert state.blacklisted_until is None


async def test_report_success_resets(pool):
    await pool.report_failure("http://proxy1:8080")
    await pool.report_failure("http://proxy1:8080")
    await pool.report_success("http://proxy1:8080")
    state = pool._state["http://proxy1:8080"]
    assert state.fail_count == 0
    assert state.blacklisted_until is None


async def test_round_robin_skips_blacklisted(pool):
    for _ in range(3):
        await pool.report_failure("http://proxy2:8080")
    results = [await pool.get_proxy() for _ in range(4)]
    assert results == [
        "http://proxy1:8080",
        "http://proxy3:8080",
        "http://proxy1:8080",
        "http://proxy3:8080",
    ]


async def _sleep(seconds: float) -> None:
    await asyncio.sleep(seconds)
