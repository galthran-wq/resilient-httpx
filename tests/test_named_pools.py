from unittest.mock import AsyncMock, patch

import httpx
import pytest

from resilient_httpx import ProxyHttpClient, RetryPolicy

from .test_client import URL, NO_WAIT, _response, mock_httpx

EXTERNAL = ["http://external1:8080", "http://external2:8080"]
INTERNAL = ["http://internal1:8080", "http://internal2:8080"]


async def test_named_pools_per_request():
    captured_proxies = []
    original_get_client = ProxyHttpClient._get_client

    def spy_get_client(self, proxy):
        captured_proxies.append(proxy)
        return original_get_client(self, proxy)

    with mock_httpx([_response(200)] * 2):
        with patch.object(ProxyHttpClient, "_get_client", spy_get_client):
            async with ProxyHttpClient(
                proxies={"external": EXTERNAL, "internal": INTERNAL},
                retry=NO_WAIT,
            ) as client:
                await client.get(URL, pool="external")
                await client.get(URL, pool="internal")
    assert captured_proxies[0] in EXTERNAL
    assert captured_proxies[1] in INTERNAL


async def test_named_pools_default_combined():
    captured_proxies = []
    original_get_client = ProxyHttpClient._get_client

    def spy_get_client(self, proxy):
        captured_proxies.append(proxy)
        return original_get_client(self, proxy)

    with mock_httpx([_response(200)] * 4):
        with patch.object(ProxyHttpClient, "_get_client", spy_get_client):
            async with ProxyHttpClient(
                proxies={"external": EXTERNAL, "internal": INTERNAL},
                retry=NO_WAIT,
            ) as client:
                for _ in range(4):
                    await client.get(URL)
    assert set(captured_proxies) == set(EXTERNAL + INTERNAL)


async def test_named_pools_explicit_default():
    captured_proxies = []
    original_get_client = ProxyHttpClient._get_client

    def spy_get_client(self, proxy):
        captured_proxies.append(proxy)
        return original_get_client(self, proxy)

    with mock_httpx([_response(200)] * 2):
        with patch.object(ProxyHttpClient, "_get_client", spy_get_client):
            async with ProxyHttpClient(
                proxies={"external": EXTERNAL, "internal": INTERNAL},
                default_pool="internal",
                retry=NO_WAIT,
            ) as client:
                await client.get(URL)
                await client.get(URL)
    assert all(p in INTERNAL for p in captured_proxies)


async def test_add_pool():
    extra = ["http://extra1:8080"]
    captured_proxies = []
    original_get_client = ProxyHttpClient._get_client

    def spy_get_client(self, proxy):
        captured_proxies.append(proxy)
        return original_get_client(self, proxy)

    with mock_httpx([_response(200)]):
        with patch.object(ProxyHttpClient, "_get_client", spy_get_client):
            async with ProxyHttpClient(
                proxies={"external": EXTERNAL},
                retry=NO_WAIT,
            ) as client:
                client.add_pool("extra", extra)
                await client.get(URL, pool="extra")
    assert captured_proxies[0] in extra


async def test_add_pool_updates_combined_default():
    extra = ["http://extra1:8080"]
    client = ProxyHttpClient(
        proxies={"external": EXTERNAL},
        retry=NO_WAIT,
    )
    client.add_pool("extra", extra)
    combined = client._pools["_all"]
    assert "http://extra1:8080" in combined._proxies
    assert all(p in combined._proxies for p in EXTERNAL)


async def test_unknown_pool_raises():
    async with ProxyHttpClient(
        proxies={"external": EXTERNAL},
        retry=NO_WAIT,
    ) as client:
        with pytest.raises(ValueError, match="Unknown pool"):
            await client.get(URL, pool="nonexistent")


async def test_unknown_default_pool_raises():
    with pytest.raises(ValueError, match="Unknown default pool"):
        ProxyHttpClient(
            proxies={"external": EXTERNAL},
            default_pool="nonexistent",
        )
