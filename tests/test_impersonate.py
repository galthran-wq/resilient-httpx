"""Tests for the opt-in curl_cffi browser-impersonation request path.

curl_cffi is faked via sys.modules so these run without the native library and
without any network — they cover the wiring (client selection/caching, header
handling, per-request fingerprint, curl->httpx error mapping, streaming) rather
than a real TLS handshake.
"""

from __future__ import annotations

import sys
import types

import httpx
import pytest

from resilient_httpx.client import (
    IMPERSONATE_TARGETS,
    AsyncProxyHttpClient,
    _ImpersonateClient,
)


def _install_fake_curl_cffi(monkeypatch, *, status=200, content=b"ok", headers=None, raise_kind=None, calls=None):
    """Install a fake curl_cffi into sys.modules. ``raise_kind`` (one of
    'request' | 'timeout' | 'value' | None) makes session.request raise an
    exception built from THIS module's own classes, so the adapter's
    isinstance(exc, RequestException) check sees the same class it imports.
    """
    root = types.ModuleType("curl_cffi")
    req = types.ModuleType("curl_cffi.requests")
    exc_mod = types.ModuleType("curl_cffi.requests.exceptions")

    class RequestException(Exception):
        pass

    class Timeout(RequestException):
        pass

    exc_mod.RequestException = RequestException
    exc_mod.Timeout = Timeout

    if raise_kind == "request":
        to_raise: Exception | None = RequestException("boom")
    elif raise_kind == "timeout":
        to_raise = Timeout("slow")
    elif raise_kind == "value":
        to_raise = ValueError("programming bug")
    else:
        to_raise = None

    class _FakeResp:
        def __init__(self):
            self.status_code = status
            self.content = content
            self.headers = headers if headers is not None else {"Content-Type": "application/json"}

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def request(self, method, url, **kwargs):
            if calls is not None:
                calls.append({"method": method, "url": url, **kwargs})
            if to_raise is not None:
                raise to_raise
            return _FakeResp()

    req.AsyncSession = _FakeSession
    req.exceptions = exc_mod
    root.requests = req
    monkeypatch.setitem(sys.modules, "curl_cffi", root)
    monkeypatch.setitem(sys.modules, "curl_cffi.requests", req)
    monkeypatch.setitem(sys.modules, "curl_cffi.requests.exceptions", exc_mod)
    return exc_mod


async def test_get_client_selects_and_caches_impersonate_client():
    client = AsyncProxyHttpClient(proxies=["http://p:1"])
    a = client._get_client("http://p:1", impersonate=True)
    b = client._get_client("http://p:1", impersonate=True)
    assert isinstance(a, _ImpersonateClient)
    assert a is b  # cached by (proxy, impersonate)

    plain = client._get_client("http://p:1", impersonate=False)
    assert isinstance(plain, httpx.AsyncClient)
    assert plain is not a  # impersonate and plain are distinct cache entries


async def test_impersonate_request_builds_httpx_response(monkeypatch):
    calls: list[dict] = []
    _install_fake_curl_cffi(monkeypatch, status=200, content=b'{"ok":1}', calls=calls)
    client = _ImpersonateClient(proxy="http://p:1", timeout=5.0, headers={"User-Agent": "x", "X-K": "v"})

    resp = await client.request("GET", "https://example.com")

    assert isinstance(resp, httpx.Response)
    assert resp.status_code == 200
    assert resp.content == b'{"ok":1}'
    kw = calls[0]
    assert kw["impersonate"] in IMPERSONATE_TARGETS
    assert kw["proxies"] == {"https": "http://p:1", "http": "http://p:1"}
    assert kw["allow_redirects"] is False
    # caller User-Agent is stripped (curl_cffi sets a matching one); other headers kept
    sent_headers = kw["headers"] or {}
    assert not any(k.lower() == "user-agent" for k in sent_headers)
    assert sent_headers.get("X-K") == "v"


async def test_impersonate_direct_no_proxy(monkeypatch):
    calls: list[dict] = []
    _install_fake_curl_cffi(monkeypatch, calls=calls)
    client = _ImpersonateClient(proxy=None, timeout=5.0, headers=None)
    await client.request("GET", "https://example.com")
    assert calls[0]["proxies"] is None


async def test_impersonate_maps_request_error_to_connect_error(monkeypatch):
    _install_fake_curl_cffi(monkeypatch, raise_kind="request")
    client = _ImpersonateClient(proxy=None, timeout=5.0, headers=None)
    with pytest.raises(httpx.ConnectError):
        await client.request("GET", "https://example.com")


async def test_impersonate_maps_timeout(monkeypatch):
    _install_fake_curl_cffi(monkeypatch, raise_kind="timeout")
    client = _ImpersonateClient(proxy=None, timeout=5.0, headers=None)
    with pytest.raises(httpx.TimeoutException):
        await client.request("GET", "https://example.com")


async def test_impersonate_non_curl_error_propagates(monkeypatch):
    _install_fake_curl_cffi(monkeypatch, raise_kind="value")
    client = _ImpersonateClient(proxy=None, timeout=5.0, headers=None)
    with pytest.raises(ValueError):
        await client.request("GET", "https://example.com")


async def test_impersonate_stream_yields_full_body(monkeypatch):
    _install_fake_curl_cffi(monkeypatch, status=200, content=b"abcdef")
    client = _ImpersonateClient(proxy=None, timeout=5.0, headers=None)
    async with client.stream("GET", "https://example.com") as resp:
        body = b""
        async for chunk in resp.aiter_bytes():
            body += chunk
    assert resp.status_code == 200
    assert body == b"abcdef"


async def test_impersonate_response_preserves_multi_headers(monkeypatch):
    class _MultiHeaders:
        def multi_items(self):
            return [("Set-Cookie", "a"), ("Set-Cookie", "b"), ("Content-Type", "x")]

    _install_fake_curl_cffi(monkeypatch, headers=_MultiHeaders())
    client = _ImpersonateClient(proxy=None, timeout=5.0, headers=None)
    resp = await client.request("GET", "https://example.com")
    assert resp.headers.get_list("Set-Cookie") == ["a", "b"]


# --- end-to-end through AsyncProxyHttpClient (the actual feature wiring) ---


async def test_client_impersonate_routes_end_to_end(monkeypatch):
    calls: list[dict] = []
    _install_fake_curl_cffi(monkeypatch, status=200, content=b"XYZ", calls=calls)
    outcomes: list[tuple] = []
    client = AsyncProxyHttpClient(
        proxies=["http://p:1"],
        on_request_outcome=lambda *a: outcomes.append(a),
    )
    resp = await client.get("https://example.com", impersonate=True)
    assert resp.status_code == 200
    assert resp.content == b"XYZ"
    # went through curl_cffi (fake recorded the call) with the proxy from the pool
    assert calls and calls[0]["proxies"] == {"https": "http://p:1", "http": "http://p:1"}
    assert outcomes[-1][2] == "success"


async def test_client_impersonate_error_retries_and_reports_failure(monkeypatch):
    from resilient_httpx.exceptions import MaxRetriesExceeded
    from resilient_httpx.retry import RetryPolicy

    _install_fake_curl_cffi(monkeypatch, raise_kind="request")
    outcomes: list[tuple] = []
    client = AsyncProxyHttpClient(
        proxies=["http://p:1"],
        retry=RetryPolicy(max_attempts=2),
        blacklist_threshold=99,  # keep the proxy available so both attempts run
        on_request_outcome=lambda *a: outcomes.append(a),
    )
    with pytest.raises(MaxRetriesExceeded):
        await client.get("https://example.com", impersonate=True)
    # mapped ConnectError was counted as a proxy failure (retry/blacklist path)
    assert [o for o in outcomes if o[2] == "failed"]


async def test_client_impersonate_stream_end_to_end(monkeypatch):
    _install_fake_curl_cffi(monkeypatch, status=200, content=b"stream-body")
    client = AsyncProxyHttpClient(proxies=["http://p:1"])
    async with client.stream("GET", "https://example.com", impersonate=True) as resp:
        body = b"".join([chunk async for chunk in resp.aiter_bytes()])
    assert resp.status_code == 200
    assert body == b"stream-body"


async def test_client_default_path_unaffected_by_feature():
    # impersonate defaults False -> plain httpx client, no curl_cffi needed
    client = AsyncProxyHttpClient(proxies=["http://p:1"])
    assert isinstance(client._get_client("http://p:1"), httpx.AsyncClient)
