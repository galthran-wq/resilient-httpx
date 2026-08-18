"""Tests for the adaptive impersonate-target health pool.

curl_cffi is faked so a chosen fingerprint returns a scripted status — this lets
us prove a target the WAF 403s gets demoted and rotation moves to clean ones,
plus the per-request override and health snapshot.
"""

from __future__ import annotations

import sys
import types

import pytest

from resilient_httpx.client import AsyncProxyHttpClient


def _install_fake_curl_cffi(monkeypatch, *, status_by_target, default_status=200, calls=None):
    root = types.ModuleType("curl_cffi")
    req = types.ModuleType("curl_cffi.requests")
    exc_mod = types.ModuleType("curl_cffi.requests.exceptions")

    class RequestException(Exception):
        pass

    exc_mod.RequestException = RequestException

    class _FakeResp:
        def __init__(self, status):
            self.status_code = status
            self.content = b"ok"
            self.headers = {"Content-Type": "application/json"}

    class _FakeSession:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def request(self, method, url, **kwargs):
            target = kwargs.get("impersonate")
            if calls is not None:
                calls.append(target)
            return _FakeResp(status_by_target.get(target, default_status))

    req.AsyncSession = _FakeSession
    req.exceptions = exc_mod
    root.requests = req
    monkeypatch.setitem(sys.modules, "curl_cffi", root)
    monkeypatch.setitem(sys.modules, "curl_cffi.requests", req)
    monkeypatch.setitem(sys.modules, "curl_cffi.requests.exceptions", exc_mod)


async def test_bad_target_demoted_after_threshold_then_avoided(monkeypatch):
    calls: list[str] = []
    _install_fake_curl_cffi(monkeypatch, status_by_target={"bad": 403}, calls=calls)
    client = AsyncProxyHttpClient(
        proxies=["http://p:1"],
        impersonate=True,
        impersonate_targets=("g1", "g2", "bad"),
        impersonate_strategy="round-robin",  # deterministic: exercises the blacklist path
        impersonate_blacklist_threshold=2,
        blacklist_threshold=99,  # keep the proxy alive; 403 is not a proxy failure
    )
    for _ in range(15):
        await client.get("https://example.com")

    # 'bad' is used exactly threshold times (2 × 403 -> blacklisted), then skipped
    assert calls.count("bad") == 2
    snap = {row["proxy"]: row for row in client.impersonate_snapshot()}
    assert snap["bad"]["total_fail"] == 2
    assert snap["bad"]["in_rotation"] is False
    # clean targets absorbed the rest and stayed healthy
    assert snap["g1"]["total_ok"] > 0 and snap["g2"]["total_ok"] > 0


async def test_403_is_returned_not_retried(monkeypatch):
    # a 403 (bad target) is demoted but NOT retried by default — caller sees it
    calls: list[str] = []
    _install_fake_curl_cffi(monkeypatch, status_by_target={"only": 403}, calls=calls)
    client = AsyncProxyHttpClient(
        proxies=["http://p:1"], impersonate=True, impersonate_targets=("only",), blacklist_threshold=99
    )
    resp = await client.get("https://example.com")
    assert resp.status_code == 403
    assert calls == ["only"]  # single attempt, no retry storm


async def test_pinned_target_override_bypasses_pool(monkeypatch):
    calls: list[str] = []
    _install_fake_curl_cffi(monkeypatch, status_by_target={}, calls=calls)
    client = AsyncProxyHttpClient(
        proxies=["http://p:1"], impersonate=True, impersonate_targets=("g1", "g2")
    )
    await client.get("https://example.com", impersonate_target="pinned-x")
    assert calls == ["pinned-x"]  # pool ignored; out-of-set pin not scored (no crash)
    snap = {row["proxy"]: row for row in client.impersonate_snapshot()}
    assert snap["g1"]["total_ok"] == 0 and snap["g2"]["total_ok"] == 0


async def test_all_targets_bad_still_issues(monkeypatch):
    # every target 403s -> pool eventually all-blacklisted -> fallback random pick
    # must still issue the request (return the 403) rather than crash/hang
    _install_fake_curl_cffi(monkeypatch, status_by_target={"a": 403, "b": 403}, default_status=403)
    client = AsyncProxyHttpClient(
        proxies=["http://p:1"],
        impersonate=True,
        impersonate_targets=("a", "b"),
        impersonate_blacklist_threshold=1,
        blacklist_threshold=99,
    )
    for _ in range(6):
        resp = await client.get("https://example.com")
        assert resp.status_code == 403


async def test_success_keeps_target_healthy(monkeypatch):
    _install_fake_curl_cffi(monkeypatch, status_by_target={}, default_status=200)
    client = AsyncProxyHttpClient(
        proxies=["http://p:1"], impersonate=True, impersonate_targets=("g1", "g2")
    )
    for _ in range(6):
        await client.get("https://example.com")
    snap = {row["proxy"]: row for row in client.impersonate_snapshot()}
    assert all(snap[t]["in_rotation"] and snap[t]["total_fail"] == 0 for t in ("g1", "g2"))


async def test_snapshot_none_without_targets():
    client = AsyncProxyHttpClient(proxies=["http://p:1"], impersonate_targets=())
    assert client.impersonate_snapshot() is None


async def test_greedy_converges_to_best_target(monkeypatch):
    # 'bad' 403s intermittently is modelled here as always-403; greedy (eps=0)
    # must converge to a 200 target and stop picking 'bad' after the cold-start probe.
    calls: list[str] = []
    _install_fake_curl_cffi(monkeypatch, status_by_target={"bad": 403}, calls=calls)
    client = AsyncProxyHttpClient(
        proxies=["http://p:1"],
        impersonate=True,
        impersonate_targets=("best", "ok", "bad"),
        impersonate_strategy="greedy",
        impersonate_greedy_exploration=0.0,  # deterministic
        blacklist_threshold=99,
    )
    for _ in range(30):
        await client.get("https://example.com")
    # 'bad' only sampled during cold start (once), never chosen greedily after
    assert calls.count("bad") <= 1
    assert calls.count("bad") < calls.count("best") + calls.count("ok")


async def test_impersonate_true_empty_targets_raises_clear_error(monkeypatch):
    _install_fake_curl_cffi(monkeypatch, status_by_target={})
    client = AsyncProxyHttpClient(proxies=["http://p:1"], impersonate_targets=())
    with pytest.raises(ValueError, match="no impersonate_targets"):
        await client.get("https://example.com", impersonate=True)
