from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from resilient_httpx import (
    AllProxiesExhausted,
    MaxRetriesExceeded,
    ProxyHttpClient,
    RetryPolicy,
)

URL = "https://api.example.com/data"
NO_WAIT = RetryPolicy(min_wait=0, max_wait=0)


def _response(status: int = 200, text: str = "ok") -> httpx.Response:
    return httpx.Response(status, text=text, request=httpx.Request("GET", URL))


@contextmanager
def mock_httpx(side_effect):
    mock = AsyncMock(side_effect=side_effect)
    with (
        patch.object(httpx.AsyncClient, "request", mock),
        patch.object(httpx.AsyncClient, "aclose", new_callable=AsyncMock),
    ):
        yield mock


async def test_successful_request_no_proxies():
    with mock_httpx([_response(200)]) as mock:
        async with ProxyHttpClient(retry=NO_WAIT) as client:
            resp = await client.get(URL)
    assert resp.status_code == 200
    assert mock.call_count == 1


async def test_successful_request_with_proxies(proxy_list):
    with mock_httpx([_response(200)]):
        async with ProxyHttpClient(proxies=proxy_list, retry=NO_WAIT) as client:
            resp = await client.get(URL)
    assert resp.status_code == 200


async def test_retry_on_502():
    with mock_httpx([_response(502), _response(200)]) as mock:
        async with ProxyHttpClient(retry=NO_WAIT) as client:
            resp = await client.get(URL)
    assert resp.status_code == 200
    assert mock.call_count == 2


async def test_retry_on_429():
    with mock_httpx([_response(429), _response(429), _response(200)]) as mock:
        policy = RetryPolicy(max_attempts=5, min_wait=0, max_wait=0)
        async with ProxyHttpClient(retry=policy) as client:
            resp = await client.get(URL)
    assert resp.status_code == 200
    assert mock.call_count == 3


async def test_retry_on_network_exception():
    side_effect = [httpx.ConnectError("connection refused"), _response(200)]
    with mock_httpx(side_effect) as mock:
        async with ProxyHttpClient(retry=NO_WAIT) as client:
            resp = await client.get(URL)
    assert resp.status_code == 200
    assert mock.call_count == 2


async def test_max_retries_exceeded():
    with mock_httpx([_response(502)] * 3):
        async with ProxyHttpClient(retry=NO_WAIT) as client:
            with pytest.raises(MaxRetriesExceeded):
                await client.get(URL)


async def test_max_retries_exceeded_has_cause():
    with mock_httpx([httpx.ConnectError("fail")] * 3):
        async with ProxyHttpClient(retry=NO_WAIT) as client:
            with pytest.raises(MaxRetriesExceeded) as exc_info:
                await client.get(URL)
    assert isinstance(exc_info.value.__cause__, httpx.ConnectError)


async def test_all_proxies_exhausted(proxy_list):
    responses = [_response(502)] * 4
    with mock_httpx(responses):
        policy = RetryPolicy(max_attempts=5, min_wait=0, max_wait=0)
        async with ProxyHttpClient(
            proxies=proxy_list,
            retry=policy,
            blacklist_threshold=1,
        ) as client:
            with pytest.raises(AllProxiesExhausted):
                await client.get(URL)


async def test_all_proxies_exhausted_fallback(proxy_list):
    responses = [_response(502)] * 3 + [_response(200)]
    with mock_httpx(responses) as mock:
        policy = RetryPolicy(max_attempts=5, min_wait=0, max_wait=0)
        async with ProxyHttpClient(
            proxies=proxy_list,
            retry=policy,
            blacklist_threshold=1,
            fallback_to_direct=True,
        ) as client:
            resp = await client.get(URL)
    assert resp.status_code == 200
    assert mock.call_count == 4


async def test_proxy_rotation_on_failure(proxy_list):
    captured_proxies = []
    original_get_client = ProxyHttpClient._get_client

    def spy_get_client(self, proxy):
        captured_proxies.append(proxy)
        return original_get_client(self, proxy)

    with mock_httpx([_response(502), _response(502), _response(200)]):
        with patch.object(ProxyHttpClient, "_get_client", spy_get_client):
            async with ProxyHttpClient(proxies=proxy_list, retry=NO_WAIT) as client:
                await client.get(URL)
    assert len(captured_proxies) == 3
    assert captured_proxies[0] != captured_proxies[1]


async def test_context_manager_closes_clients():
    with mock_httpx([_response(200)]):
        client = ProxyHttpClient(retry=NO_WAIT)
        async with client:
            await client.get(URL)
        assert client._clients == {}


async def test_http_methods():
    for method in ["get", "post", "put", "patch", "delete"]:
        with mock_httpx([_response(200)]) as mock:
            async with ProxyHttpClient(retry=NO_WAIT) as client:
                resp = await getattr(client, method)(URL)
        assert resp.status_code == 200
        assert mock.call_args[0][0] == method.upper()


async def test_no_retry_on_non_retriable_status():
    with mock_httpx([_response(400)]) as mock:
        async with ProxyHttpClient(retry=NO_WAIT) as client:
            resp = await client.get(URL)
    assert resp.status_code == 400
    assert mock.call_count == 1
