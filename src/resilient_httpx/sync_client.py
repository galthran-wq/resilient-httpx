from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

import httpx
import structlog
import threading
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
    ) -> None:
        self._strategy = proxy_strategy
        self._blacklist_threshold = blacklist_threshold
        self._blacklist_ttl = blacklist_ttl
        self._pools: dict[str, SyncProxyPool] = {}
        self._default_pool_name: str | None = None
        self._pool_lock: threading.Lock | None = None

        if default_pool is not None and not isinstance(proxies, dict):
            raise ValueError("default_pool requires proxies to be a dict")

        if isinstance(proxies, list):
            self._pools[_DEFAULT] = self._make_pool(proxies)
            self._default_pool_name = _DEFAULT
        elif isinstance(proxies, dict):
            if default_pool is None:
                self._pool_lock = threading.Lock()
            reserved = {_ALL, _DEFAULT}
            overlap = reserved.intersection(proxies)
            if overlap:
                raise ValueError(f"Reserved pool name(s): {sorted(overlap)!r}")
            for name, proxy_list in proxies.items():
                self._pools[name] = self._make_pool(proxy_list)
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
        self._client_lock = threading.Lock()
        self._log = structlog.get_logger()

    def _make_pool(
        self,
        proxies: list[str],
        state: dict[str, object] | None = None,
    ) -> SyncProxyPool:
        return SyncProxyPool(
            proxies=proxies,
            strategy=self._strategy,
            blacklist_threshold=self._blacklist_threshold,
            blacklist_ttl=self._blacklist_ttl,
            state=state,
            lock=self._pool_lock,
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
            self._pools[_ALL] = self._make_pool(all_proxies, state=state)
            self._default_pool_name = _ALL

    def add_pool(self, name: str, proxies: list[str]) -> None:
        if name in (_ALL, _DEFAULT):
            raise ValueError(f"Reserved pool name: {name!r}")
        self._pools[name] = self._make_pool(proxies)
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
        with self._client_lock:
            client = self._clients.get(proxy)
            if client is None:
                client = httpx.Client(
                    proxy=proxy,
                    timeout=self._timeout,
                    headers=self._headers,
                )
                self._clients[proxy] = client
            return client

    def _request(
        self, method: str, url: str, *, pool: str | None = None, **kwargs,
    ) -> httpx.Response:
        active_pool = self._resolve_pool(pool)
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
                    try:
                        response = client.request(method, url, **kwargs)
                    except tuple(self._retry.retry_on_exception):
                        if active_pool and proxy:
                            blacklisted = active_pool.report_failure(proxy)
                            if blacklisted:
                                self._log.warning(
                                    "proxy_blacklisted", proxy=proxy
                                )
                        raise

                    if response.status_code in self._retry.retry_on:
                        if active_pool and proxy:
                            blacklisted = active_pool.report_failure(proxy)
                            if blacklisted:
                                self._log.warning(
                                    "proxy_blacklisted", proxy=proxy
                                )
                        raise _RetriableStatusError(response)

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
        retrying = self._retry.build_sync_retrying(
            extra_exceptions=[_RetriableStatusError],
        )

        open_stream = None
        proxy = None
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
                        cm = client.stream(method, url, **kwargs)
                        try:
                            response = cm.__enter__()
                        except tuple(self._retry.retry_on_exception):
                            if active_pool and proxy:
                                blacklisted = active_pool.report_failure(proxy)
                                if blacklisted:
                                    self._log.warning(
                                        "proxy_blacklisted", proxy=proxy
                                    )
                            raise

                        if response.status_code in self._retry.retry_on:
                            cm.__exit__(None, None, None)
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
                if active_pool and proxy and isinstance(exc, httpx.HTTPError):
                    blacklisted = active_pool.report_failure(proxy)
                    if blacklisted:
                        self._log.warning("proxy_blacklisted", proxy=proxy)
                raise
            else:
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
        with self._client_lock:
            clients = list(self._clients.values())
            self._clients.clear()
        for client in clients:
            client.close()

    def __enter__(self) -> ProxyHttpClient:
        return self

    def __exit__(self, *args) -> None:
        self.close()
