from contextlib import contextmanager
from unittest.mock import MagicMock, patch

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
    mock = MagicMock(side_effect=side_effect)
    with (
        patch.object(httpx.Client, "request", mock),
        patch.object(httpx.Client, "close"),
    ):
        yield mock


@contextmanager
def _as_stream(effect):
    if isinstance(effect, BaseException):
        raise effect
    yield effect


@contextmanager
def mock_httpx_stream(side_effects):
    effects = iter(side_effects)

    def side_effect_fn(*args, **kwargs):
        return _as_stream(next(effects))

    mock = MagicMock(side_effect=side_effect_fn)
    with (
        patch.object(httpx.Client, "stream", mock),
        patch.object(httpx.Client, "close"),
    ):
        yield mock


def test_successful_request_no_proxies():
    with mock_httpx([_response(200)]) as mock:
        with ProxyHttpClient(retry=NO_WAIT) as client:
            resp = client.get(URL)
    assert resp.status_code == 200
    assert mock.call_count == 1


def test_successful_request_with_proxies(proxy_list):
    with mock_httpx([_response(200)]):
        with ProxyHttpClient(proxies=proxy_list, retry=NO_WAIT) as client:
            resp = client.get(URL)
    assert resp.status_code == 200


def test_retry_on_502():
    with mock_httpx([_response(502), _response(200)]) as mock:
        with ProxyHttpClient(retry=NO_WAIT) as client:
            resp = client.get(URL)
    assert resp.status_code == 200
    assert mock.call_count == 2


def test_retry_on_429():
    with mock_httpx([_response(429), _response(429), _response(200)]) as mock:
        policy = RetryPolicy(max_attempts=5, min_wait=0, max_wait=0)
        with ProxyHttpClient(retry=policy) as client:
            resp = client.get(URL)
    assert resp.status_code == 200
    assert mock.call_count == 3


def test_retry_on_network_exception():
    side_effect = [httpx.ConnectError("connection refused"), _response(200)]
    with mock_httpx(side_effect) as mock:
        with ProxyHttpClient(retry=NO_WAIT) as client:
            resp = client.get(URL)
    assert resp.status_code == 200
    assert mock.call_count == 2


def test_max_retries_exceeded():
    with mock_httpx([_response(502)] * 3):
        with ProxyHttpClient(retry=NO_WAIT) as client:
            with pytest.raises(MaxRetriesExceeded):
                client.get(URL)


def test_max_retries_exceeded_has_cause():
    with mock_httpx([httpx.ConnectError("fail")] * 3):
        with ProxyHttpClient(retry=NO_WAIT) as client:
            with pytest.raises(MaxRetriesExceeded) as exc_info:
                client.get(URL)
    assert isinstance(exc_info.value.__cause__, httpx.ConnectError)


def test_all_proxies_exhausted(proxy_list):
    responses = [_response(502)] * 4
    with mock_httpx(responses):
        policy = RetryPolicy(max_attempts=5, min_wait=0, max_wait=0)
        with ProxyHttpClient(
            proxies=proxy_list,
            retry=policy,
            blacklist_threshold=1,
        ) as client:
            with pytest.raises(AllProxiesExhausted):
                client.get(URL)


def test_all_proxies_exhausted_fallback(proxy_list):
    responses = [_response(502)] * 3 + [_response(200)]
    with mock_httpx(responses) as mock:
        policy = RetryPolicy(max_attempts=5, min_wait=0, max_wait=0)
        with ProxyHttpClient(
            proxies=proxy_list,
            retry=policy,
            blacklist_threshold=1,
            fallback_to_direct=True,
        ) as client:
            resp = client.get(URL)
    assert resp.status_code == 200
    assert mock.call_count == 4


def test_proxy_rotation_on_failure(proxy_list):
    captured_proxies = []
    original_get_client = ProxyHttpClient._get_client

    def spy_get_client(self, proxy):
        captured_proxies.append(proxy)
        return original_get_client(self, proxy)

    with mock_httpx([_response(502), _response(502), _response(200)]):
        with patch.object(ProxyHttpClient, "_get_client", spy_get_client):
            with ProxyHttpClient(proxies=proxy_list, retry=NO_WAIT) as client:
                client.get(URL)
    assert len(captured_proxies) == 3
    assert captured_proxies[0] != captured_proxies[1]


def test_context_manager_closes_clients():
    with mock_httpx([_response(200)]):
        client = ProxyHttpClient(retry=NO_WAIT)
        with client:
            client.get(URL)
        assert client._clients == {}


def test_http_methods():
    for method in ["get", "post", "put", "patch", "delete"]:
        with mock_httpx([_response(200)]) as mock:
            with ProxyHttpClient(retry=NO_WAIT) as client:
                resp = getattr(client, method)(URL)
        assert resp.status_code == 200
        assert mock.call_args[0][0] == method.upper()


def test_no_retry_on_non_retriable_status():
    with mock_httpx([_response(400)]) as mock:
        with ProxyHttpClient(retry=NO_WAIT) as client:
            resp = client.get(URL)
    assert resp.status_code == 400
    assert mock.call_count == 1


def test_stream_success_no_proxies():
    with mock_httpx_stream([_response(200)]) as mock:
        with ProxyHttpClient(retry=NO_WAIT) as client:
            with client.stream("GET", URL) as resp:
                assert resp.status_code == 200
    assert mock.call_count == 1


def test_stream_success_with_proxies(proxy_list):
    with mock_httpx_stream([_response(200)]):
        with ProxyHttpClient(proxies=proxy_list, retry=NO_WAIT) as client:
            with client.stream("GET", URL) as resp:
                assert resp.status_code == 200


def test_stream_retry_on_502():
    with mock_httpx_stream([_response(502), _response(200)]) as mock:
        with ProxyHttpClient(retry=NO_WAIT) as client:
            with client.stream("GET", URL) as resp:
                assert resp.status_code == 200
    assert mock.call_count == 2


def test_stream_retry_on_network_exception():
    side_effects = [httpx.ConnectError("connection refused"), _response(200)]
    with mock_httpx_stream(side_effects) as mock:
        with ProxyHttpClient(retry=NO_WAIT) as client:
            with client.stream("GET", URL) as resp:
                assert resp.status_code == 200
    assert mock.call_count == 2


def test_stream_max_retries_exceeded():
    with mock_httpx_stream([_response(502)] * 3):
        with ProxyHttpClient(retry=NO_WAIT) as client:
            with pytest.raises(MaxRetriesExceeded):
                with client.stream("GET", URL):
                    pass


def test_stream_all_proxies_exhausted(proxy_list):
    responses = [_response(502)] * 4
    with mock_httpx_stream(responses):
        policy = RetryPolicy(max_attempts=5, min_wait=0, max_wait=0)
        with ProxyHttpClient(
            proxies=proxy_list,
            retry=policy,
            blacklist_threshold=1,
        ) as client:
            with pytest.raises(AllProxiesExhausted):
                with client.stream("GET", URL):
                    pass


def test_stream_no_retry_on_non_retriable_status():
    with mock_httpx_stream([_response(400)]) as mock:
        with ProxyHttpClient(retry=NO_WAIT) as client:
            with client.stream("GET", URL) as resp:
                assert resp.status_code == 400
    assert mock.call_count == 1
