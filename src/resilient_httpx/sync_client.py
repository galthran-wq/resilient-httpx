from __future__ import annotations

import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager

import httpx
import structlog
from tenacity import RetryError

from resilient_httpx.exceptions import AllProxiesExhausted, MaxRetriesExceeded
from resilient_httpx.retry import RetryPolicy
from resilient_httpx.sync_pool import SyncProxyPool

_ALL = "_all"
_DEFAULT = "_default"


class _RetriableStatusError(Exception):
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        super().__init__(f"HTTP {response.status_code}")


class ProxyHttpClient:
    def __init__(
        self,
        proxies: list[str] | dict[str, list[str]] | None = None,
        proxy_strategy: str = "round-robin",
        retry: RetryPolicy | None = None,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        blacklist_threshold: int = 3,
        blacklist_ttl: float = 300.0,
        fallback_to_direct: bool = False,
        default_pool: str | None = None,
        on_blacklist: Callable[[str, str], None] | None = None,
        on_unblacklist: Callable[[str, str], None] | None = None,
        on_request_outcome: Callable[[str, str | None, str, float], None] | None = None,
        probe_on_start: bool = False,
        probe_url: str = "http://example.com",
        probe_timeout: float = 5.0,
    ) -> None:
        self._strategy = proxy_strategy
        self._blacklist_threshold = blacklist_threshold
        self._blacklist_ttl = blacklist_ttl
        self._on_blacklist = on_blacklist
        self._on_unblacklist = on_unblacklist
        self._on_request_outcome = on_request_outcome
        self._probe_url = probe_url
        self._probe_timeout = probe_timeout
        self._pools: dict[str, SyncProxyPool] = {}
        self._default_pool_name: str | None = None

        if default_pool is not None and not isinstance(proxies, dict):
            raise ValueError("default_pool requires proxies to be a dict")

        if isinstance(proxies, list):
            self._pools[_DEFAULT] = self._make_pool(_DEFAULT, proxies)
            self._default_pool_name = _DEFAULT
        elif isinstance(proxies, dict):
            reserved = {_ALL, _DEFAULT}
            overlap = reserved.intersection(proxies)
            if overlap:
                raise ValueError(f"Reserved pool name(s): {sorted(overlap)!r}")
            for name, proxy_list in proxies.items():
                self._pools[name] = self._make_pool(name, proxy_list)
            if default_pool is not None:
                if default_pool not in self._pools:
                    raise ValueError(f"Unknown default pool: {default_pool!r}")
                self._default_pool_name = default_pool
            else:
                self._build_combined_pool()

        self._retry = retry or RetryPolicy()
        self._timeout = timeout
        self._headers = headers or {}
        self._fallback = fallback_to_direct
        self._clients: dict[str | None, httpx.Client] = {}
        self._clients_lock = threading.Lock()
        self._log = structlog.get_logger()
        self._probed = False
        self._probe_lock = threading.Lock()
        if probe_on_start:
            self.probe_all()

    def _make_pool(
        self,
        name: str,
        proxies: list[str],
        state: dict[str, object] | None = None,
    ) -> SyncProxyPool:
        on_blacklist = None
        on_unblacklist = None
        if self._on_blacklist is not None:
            on_blacklist = lambda proxy, n=name: self._on_blacklist(n, proxy)
        if self._on_unblacklist is not None:
            on_unblacklist = lambda proxy, n=name: self._on_unblacklist(n, proxy)
        return SyncProxyPool(
            proxies=proxies,
            strategy=self._strategy,
            blacklist_threshold=self._blacklist_threshold,
            blacklist_ttl=self._blacklist_ttl,
            state=state,
            on_blacklist=on_blacklist,
            on_unblacklist=on_unblacklist,
        )

    def _build_combined_pool(self) -> None:
        all_proxies = []
        state = {}
        for name, pool in self._pools.items():
            if name != _ALL:
                for proxy in pool._proxies:
                    all_proxies.append(proxy)
                    if proxy not in state:
                        state[proxy] = pool._state[proxy]
        if all_proxies:
            self._pools[_ALL] = self._make_pool(_ALL, all_proxies, state=state)
            self._default_pool_name = _ALL

    def add_pool(self, name: str, proxies: list[str]) -> None:
        if name in (_ALL, _DEFAULT):
            raise ValueError(f"Reserved pool name: {name!r}")
        self._pools[name] = self._make_pool(name, proxies)
        if self._default_pool_name == _ALL or self._default_pool_name is None:
            self._build_combined_pool()

    def _resolve_pool(self, pool: str | None) -> SyncProxyPool | None:
        if pool is not None:
            if pool not in self._pools:
                raise ValueError(f"Unknown pool: {pool!r}")
            return self._pools[pool]
        if self._default_pool_name is not None:
            return self._pools[self._default_pool_name]
        return None

    def _get_client(self, proxy: str | None) -> httpx.Client:
        with self._clients_lock:
            if proxy not in self._clients:
                self._clients[proxy] = httpx.Client(
                    proxy=proxy,
                    timeout=self._timeout,
                    headers=self._headers,
                )
            return self._clients[proxy]

    def blacklist_snapshot(self) -> dict[str, list[dict]]:
        return {
            name: pool.snapshot()
            for name, pool in self._pools.items()
            if name != _ALL
        }

    def probe_all(self) -> dict[str, list[str]]:
        """Probe each proxy once via probe_url. Returns {pool_name: [dead_proxies]}.

        Idempotent — concurrent callers block on a lock and only the first one runs.
        """
        with self._probe_lock:
            if self._probed:
                return {}
            seen: set[str] = set()
            dead: dict[str, list[str]] = {}
            for name, pool in self._pools.items():
                if name == _ALL:
                    continue
                dead[name] = []
                for proxy in pool._proxies:
                    if proxy in seen:
                        continue
                    seen.add(proxy)
                    client = self._get_client(proxy)
                    try:
                        response = client.get(self._probe_url, timeout=self._probe_timeout)
                        if response.status_code >= 500:
                            raise RuntimeError(f"probe got {response.status_code}")
                    except Exception as exc:
                        pool.blacklist(proxy)
                        dead[name].append(proxy)
                        self._log.warning("probe_failed", proxy=proxy, error=str(exc))
            self._probed = True
            return dead

    def _emit_outcome(self, pool_name: str, proxy: str | None, status: str, duration: float) -> None:
        if self._on_request_outcome is None:
            return
        try:
            self._on_request_outcome(pool_name, proxy, status, duration)
        except Exception:
            pass

    def _request(
        self, method: str, url: str, *, pool: str | None = None, **kwargs,
    ) -> httpx.Response:
        active_pool = self._resolve_pool(pool)
        pool_name = pool or self._default_pool_name or _DEFAULT
        retrying = self._retry.build_sync_retrying(
            extra_exceptions=[_RetriableStatusError],
        )

        try:
            for attempt in retrying:
                with attempt:
                    proxy = None
                    if active_pool:
                        proxy = active_pool.get_proxy()
                        if proxy is None:
                            if self._fallback:
                                self._log.warning(
                                    "all_proxies_blacklisted",
                                    fallback="direct",
                                )
                            else:
                                raise AllProxiesExhausted()

                    self._log.debug(
                        "request_attempt",
                        method=method,
                        url=url,
                        proxy=proxy,
                        attempt=attempt.retry_state.attempt_number,
                    )

                    client = self._get_client(proxy)
                    started = time.monotonic()
                    try:
                        response = client.request(method, url, **kwargs)
                    except tuple(self._retry.retry_on_exception):
                        duration = time.monotonic() - started
                        self._emit_outcome(pool_name, proxy, "failed", duration)
                        if active_pool and proxy:
                            blacklisted = active_pool.report_failure(proxy)
                            if blacklisted:
                                self._log.warning(
                                    "proxy_blacklisted", proxy=proxy
                                )
                        raise

                    if response.status_code in self._retry.retry_on:
                        duration = time.monotonic() - started
                        self._emit_outcome(pool_name, proxy, "failed", duration)
                        if active_pool and proxy:
                            blacklisted = active_pool.report_failure(proxy)
                            if blacklisted:
                                self._log.warning(
                                    "proxy_blacklisted", proxy=proxy
                                )
                        raise _RetriableStatusError(response)

                    duration = time.monotonic() - started
                    self._emit_outcome(pool_name, proxy, "success", duration)
                    if active_pool and proxy:
                        active_pool.report_success(proxy)
                    return response
        except AllProxiesExhausted:
            self._log.error("all_proxies_exhausted")
            raise
        except RetryError as exc:
            last = exc.last_attempt.exception()
            self._log.error("max_retries_exceeded", last_error=str(last))
            raise MaxRetriesExceeded(str(last)) from last

    @contextmanager
    def _stream(
        self, method: str, url: str, *, pool: str | None = None, **kwargs,
    ) -> Iterator[httpx.Response]:
        active_pool = self._resolve_pool(pool)
        pool_name = pool or self._default_pool_name or _DEFAULT
        retrying = self._retry.build_sync_retrying(
            extra_exceptions=[_RetriableStatusError],
        )

        open_stream = None
        proxy = None
        started = 0.0
        try:
            try:
                for attempt in retrying:
                    with attempt:
                        proxy = None
                        if active_pool:
                            proxy = active_pool.get_proxy()
                            if proxy is None:
                                if self._fallback:
                                    self._log.warning(
                                        "all_proxies_blacklisted",
                                        fallback="direct",
                                    )
                                else:
                                    raise AllProxiesExhausted()

                        self._log.debug(
                            "stream_attempt",
                            method=method,
                            url=url,
                            proxy=proxy,
                            attempt=attempt.retry_state.attempt_number,
                        )

                        client = self._get_client(proxy)
                        started = time.monotonic()
                        cm = client.stream(method, url, **kwargs)
                        try:
                            response = cm.__enter__()
                        except tuple(self._retry.retry_on_exception):
                            duration = time.monotonic() - started
                            self._emit_outcome(pool_name, proxy, "failed", duration)
                            if active_pool and proxy:
                                blacklisted = active_pool.report_failure(proxy)
                                if blacklisted:
                                    self._log.warning(
                                        "proxy_blacklisted", proxy=proxy
                                    )
                            raise

                        if response.status_code in self._retry.retry_on:
                            cm.__exit__(None, None, None)
                            duration = time.monotonic() - started
                            self._emit_outcome(pool_name, proxy, "failed", duration)
                            if active_pool and proxy:
                                blacklisted = active_pool.report_failure(proxy)
                                if blacklisted:
                                    self._log.warning(
                                        "proxy_blacklisted", proxy=proxy
                                    )
                            raise _RetriableStatusError(response)

                        open_stream = cm
            except AllProxiesExhausted:
                self._log.error("all_proxies_exhausted")
                raise
            except RetryError as exc:
                last = exc.last_attempt.exception()
                self._log.error("max_retries_exceeded", last_error=str(last))
                raise MaxRetriesExceeded(str(last)) from last

            try:
                yield response
            except Exception as exc:
                duration = time.monotonic() - started
                self._emit_outcome(pool_name, proxy, "failed", duration)
                if active_pool and proxy and isinstance(exc, httpx.HTTPError):
                    blacklisted = active_pool.report_failure(proxy)
                    if blacklisted:
                        self._log.warning("proxy_blacklisted", proxy=proxy)
                raise
            else:
                duration = time.monotonic() - started
                self._emit_outcome(pool_name, proxy, "success", duration)
                if active_pool and proxy:
                    active_pool.report_success(proxy)
        finally:
            if open_stream is not None:
                open_stream.__exit__(None, None, None)

    def stream(
        self, method: str, url: str, *, pool: str | None = None, **kwargs,
    ) -> Iterator[httpx.Response]:
        return self._stream(method, url, pool=pool, **kwargs)

    def get(self, url: str, *, pool: str | None = None, **kwargs) -> httpx.Response:
        return self._request("GET", url, pool=pool, **kwargs)

    def post(self, url: str, *, pool: str | None = None, **kwargs) -> httpx.Response:
        return self._request("POST", url, pool=pool, **kwargs)

    def put(self, url: str, *, pool: str | None = None, **kwargs) -> httpx.Response:
        return self._request("PUT", url, pool=pool, **kwargs)

    def patch(self, url: str, *, pool: str | None = None, **kwargs) -> httpx.Response:
        return self._request("PATCH", url, pool=pool, **kwargs)

    def delete(self, url: str, *, pool: str | None = None, **kwargs) -> httpx.Response:
        return self._request("DELETE", url, pool=pool, **kwargs)

    def close(self) -> None:
        with self._clients_lock:
            for client in self._clients.values():
                client.close()
            self._clients.clear()

    def __enter__(self) -> ProxyHttpClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()
