from __future__ import annotations

import asyncio
import random
import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager

import httpx
import structlog
from tenacity import RetryError

from resilient_httpx.exceptions import AllProxiesExhausted, MaxRetriesExceeded
from resilient_httpx.pool import HealthPool
from resilient_httpx.retry import RetryPolicy

_ALL = "_all"
_DEFAULT = "_default"

# Recent-browser TLS/JA3 fingerprints for curl_cffi. The client keeps a health
# pool over these and prefers the ones currently passing (a WAF may reclassify a
# specific fingerprint into a "spam" pool and 403 it — see impersonate_bad_statuses;
# such a target is demoted and rotation moves to a clean one, self-healing on TTL).
# A diverse, modern set (chrome/firefox/safari/android) so a subset getting flagged
# still leaves clean options. All are valid curl_cffi 0.16.x targets — keep them
# current as browsers advance (stale fingerprints are the ones that get flagged).
IMPERSONATE_TARGETS = (
    "chrome136",
    "chrome142",
    "chrome145",
    "chrome146",
    "firefox144",
    "firefox147",
    "safari184",
    "chrome131_android",
)


class _ImpersonateClient:
    """An httpx.AsyncClient-compatible shim (``request`` / ``stream`` / ``aclose``)
    backed by curl_cffi with a per-request randomised browser TLS fingerprint.

    A fresh curl_cffi session per request guarantees the JA3 actually rotates
    (a pooled connection would keep its first handshake). curl_cffi transport
    errors are re-raised as the matching httpx exception, so the surrounding
    pool retry/blacklist logic in AsyncProxyHttpClient is unchanged. Responses
    are returned as real ``httpx.Response`` objects (fully read into memory), so
    callers keep using ``.status_code`` / ``.headers`` / ``.aiter_bytes()`` /
    ``.raise_for_status()`` exactly as with httpx.
    """

    def __init__(
        self,
        proxy: str | None,
        timeout: float,
        headers: dict[str, str] | None,
        targets: tuple[str, ...] = IMPERSONATE_TARGETS,
    ) -> None:
        self._proxy = proxy
        self._timeout = timeout
        # Drop any caller User-Agent: curl_cffi's impersonation sets a UA that
        # matches the fingerprint; a mismatched UA is itself a bot signal.
        self._headers = {k: v for k, v in (headers or {}).items() if k.lower() != "user-agent"}
        self._targets = targets

    def _proxies(self) -> dict[str, str] | None:
        if not self._proxy:
            return None
        return {"https": self._proxy, "http": self._proxy}

    async def _do(self, method: str, url: str, kwargs: dict) -> httpx.Response:
        from curl_cffi.requests import AsyncSession
        from curl_cffi.requests.exceptions import RequestException

        headers = dict(self._headers)
        extra = kwargs.get("headers")
        if extra:
            headers.update({k: v for k, v in dict(extra).items() if k.lower() != "user-agent"})
        # The client's target pool selects adaptively and passes the choice in;
        # fall back to a random pick when called directly (no pool).
        target = kwargs.get("impersonate_target")
        if not target:
            if not self._targets:
                raise ValueError("impersonate requested but no impersonate_targets configured")
            target = random.choice(self._targets)
        try:
            async with AsyncSession() as session:
                resp = await session.request(
                    method,
                    url,
                    impersonate=target,
                    proxies=self._proxies(),
                    timeout=kwargs.get("timeout", self._timeout),
                    allow_redirects=False,
                    headers=headers or None,
                    params=kwargs.get("params"),
                    json=kwargs.get("json"),
                    data=kwargs.get("data") if kwargs.get("data") is not None else kwargs.get("content"),
                )
        except Exception as exc:  # noqa: BLE001 -- map curl_cffi transport errors to httpx
            # Map every curl_cffi-originated error (RequestException family AND
            # lower-level CurlError, matched by module) onto the httpx exception
            # the pool's retry/blacklist logic understands; re-raise anything
            # else (e.g. a programming bug) unchanged.
            is_curl = isinstance(exc, RequestException) or type(exc).__module__.startswith("curl_cffi")
            if not is_curl:
                raise
            if "timeout" in type(exc).__name__.lower():
                raise httpx.TimeoutException(str(exc)) from exc
            raise httpx.ConnectError(str(exc)) from exc
        try:
            raw_headers = list(resp.headers.multi_items())
        except AttributeError:
            raw_headers = list(dict(resp.headers).items())
        return httpx.Response(
            status_code=resp.status_code,
            headers=raw_headers,
            content=resp.content,
            request=httpx.Request(method, url),
        )

    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        return await self._do(method, url, kwargs)

    @asynccontextmanager
    async def stream(self, method: str, url: str, **kwargs) -> AsyncIterator[httpx.Response]:
        response = await self._do(method, url, kwargs)
        try:
            yield response
        finally:
            await response.aclose()

    async def aclose(self) -> None:
        return None


class _RetriableStatusError(Exception):
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        super().__init__(f"HTTP {response.status_code}")


class AsyncProxyHttpClient:
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
        impersonate: bool = False,
        impersonate_targets: tuple[str, ...] = IMPERSONATE_TARGETS,
        impersonate_strategy: str = "greedy",
        impersonate_bad_statuses: tuple[int, ...] = (403,),
        impersonate_blacklist_threshold: int = 2,
        impersonate_blacklist_ttl: float = 600.0,
        impersonate_greedy_exploration: float = 0.1,
    ) -> None:
        self._strategy = proxy_strategy
        self._impersonate = impersonate
        self._impersonate_targets = impersonate_targets
        self._impersonate_bad_statuses = frozenset(impersonate_bad_statuses)
        # Health pool over the fingerprints: a target that returns a bad status
        # (default 403 — WAF fingerprint block) is demoted; rotation prefers
        # clean targets and self-heals on TTL. Lower threshold + longer TTL than
        # proxies: a flagged fingerprint should drop out fast and stay out.
        self._target_pool: HealthPool | None = (
            HealthPool(
                list(impersonate_targets),
                strategy=impersonate_strategy,
                blacklist_threshold=impersonate_blacklist_threshold,
                blacklist_ttl=impersonate_blacklist_ttl,
                greedy_exploration=impersonate_greedy_exploration,
            )
            if impersonate_targets
            else None
        )
        self._blacklist_threshold = blacklist_threshold
        self._blacklist_ttl = blacklist_ttl
        self._on_blacklist = on_blacklist
        self._on_unblacklist = on_unblacklist
        self._on_request_outcome = on_request_outcome
        self._probe_on_start = probe_on_start
        self._probe_url = probe_url
        self._probe_timeout = probe_timeout
        self._pools: dict[str, HealthPool] = {}
        self._default_pool_name: str | None = None
        self._probed = False
        self._probe_lock = asyncio.Lock()

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
        self._clients: dict[tuple[str | None, bool], httpx.AsyncClient | _ImpersonateClient] = {}
        self._log = structlog.get_logger()

    def _make_pool(
        self,
        name: str,
        proxies: list[str],
        state: dict[str, object] | None = None,
    ) -> HealthPool:
        on_blacklist = None
        on_unblacklist = None
        if self._on_blacklist is not None:
            on_blacklist = lambda proxy, n=name: self._on_blacklist(n, proxy)
        if self._on_unblacklist is not None:
            on_unblacklist = lambda proxy, n=name: self._on_unblacklist(n, proxy)
        return HealthPool(
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

    def _resolve_pool(self, pool: str | None) -> HealthPool | None:
        if pool is not None:
            if pool not in self._pools:
                raise ValueError(f"Unknown pool: {pool!r}")
            return self._pools[pool]
        if self._default_pool_name is not None:
            return self._pools[self._default_pool_name]
        return None

    def _get_client(
        self, proxy: str | None, impersonate: bool = False
    ) -> httpx.AsyncClient | _ImpersonateClient:
        key = (proxy, impersonate)
        if key not in self._clients:
            if impersonate:
                self._clients[key] = _ImpersonateClient(
                    proxy=proxy,
                    timeout=self._timeout,
                    headers=self._headers,
                    targets=self._impersonate_targets,
                )
            else:
                self._clients[key] = httpx.AsyncClient(
                    proxy=proxy,
                    timeout=self._timeout,
                    headers=self._headers,
                )
        return self._clients[key]

    def blacklist_snapshot(self) -> dict[str, list[dict]]:
        return {
            name: pool.snapshot()
            for name, pool in self._pools.items()
            if name != _ALL
        }

    def impersonate_snapshot(self) -> list[dict] | None:
        """Per-target health (ok/fail counts, blacklist state) for the impersonate
        target pool, or None when impersonation is not configured."""
        return self._target_pool.snapshot() if self._target_pool is not None else None

    async def _pick_target(self, imp: bool, pinned: str | None) -> str | None:
        if not imp:
            return None
        if pinned is not None:
            return pinned
        if self._target_pool is not None:
            chosen = await self._target_pool.acquire()
            if chosen is not None:
                return chosen
        # All targets blacklisted (or no pool): fall back to a random pick so a
        # transient all-flagged window still issues the request (and its outcome
        # re-scores the pool).
        return random.choice(self._impersonate_targets) if self._impersonate_targets else None

    async def _report_target(self, target: str | None, status_code: int) -> None:
        # A caller-pinned target may be outside the configured set (not a pool
        # member) — use it but don't try to score it.
        if target is None or self._target_pool is None or target not in self._impersonate_targets:
            return
        if status_code in self._impersonate_bad_statuses:
            await self._target_pool.report_failure(target)
        else:
            await self._target_pool.report_success(target)

    async def probe_all(self) -> dict[str, list[str]]:
        """Probe each proxy once via probe_url. Returns {pool_name: [dead_proxies]}.

        Idempotent — concurrent callers block on a lock and only the first one runs.
        """
        async with self._probe_lock:
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
                        response = await client.get(self._probe_url, timeout=self._probe_timeout)
                        if response.status_code >= 500:
                            raise RuntimeError(f"probe got {response.status_code}")
                    except Exception as exc:
                        await pool.blacklist(proxy)
                        dead[name].append(proxy)
                        self._log.warning("probe_failed", proxy=proxy, error=str(exc))
            self._probed = True
            return dead

    async def _ensure_probed(self) -> None:
        if self._probe_on_start and not self._probed:
            await self.probe_all()

    async def _request(
        self, method: str, url: str, *, pool: str | None = None, impersonate: bool | None = None, **kwargs,
    ) -> httpx.Response:
        await self._ensure_probed()
        imp = self._impersonate if impersonate is None else impersonate
        pinned_target = kwargs.pop("impersonate_target", None)
        active_pool = self._resolve_pool(pool)
        pool_name = pool or self._default_pool_name or _DEFAULT
        retrying = self._retry.build_retrying(
            extra_exceptions=[_RetriableStatusError],
        )

        try:
            async for attempt in retrying:
                with attempt:
                    proxy = None
                    if active_pool:
                        proxy = await active_pool.get_proxy()
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

                    client = self._get_client(proxy, imp)
                    target = await self._pick_target(imp, pinned_target)
                    call_kwargs = kwargs if target is None else {**kwargs, "impersonate_target": target}
                    started = time.monotonic()
                    try:
                        response = await client.request(method, url, **call_kwargs)
                    except tuple(self._retry.retry_on_exception):
                        duration = time.monotonic() - started
                        self._emit_outcome(pool_name, proxy, "failed", duration)
                        if active_pool and proxy:
                            blacklisted = await active_pool.report_failure(proxy)
                            if blacklisted:
                                self._log.warning(
                                    "proxy_blacklisted", proxy=proxy
                                )
                        raise

                    await self._report_target(target, response.status_code)

                    if response.status_code in self._retry.retry_on:
                        duration = time.monotonic() - started
                        self._emit_outcome(pool_name, proxy, "failed", duration)
                        if active_pool and proxy:
                            blacklisted = await active_pool.report_failure(proxy)
                            if blacklisted:
                                self._log.warning(
                                    "proxy_blacklisted", proxy=proxy
                                )
                        raise _RetriableStatusError(response)

                    duration = time.monotonic() - started
                    self._emit_outcome(pool_name, proxy, "success", duration)
                    if active_pool and proxy:
                        await active_pool.report_success(proxy)
                    return response
        except AllProxiesExhausted:
            self._log.error("all_proxies_exhausted")
            raise
        except RetryError as exc:
            last = exc.last_attempt.exception()
            self._log.error("max_retries_exceeded", last_error=str(last))
            raise MaxRetriesExceeded(str(last)) from last

    def _emit_outcome(self, pool_name: str, proxy: str | None, status: str, duration: float) -> None:
        if self._on_request_outcome is None:
            return
        try:
            self._on_request_outcome(pool_name, proxy, status, duration)
        except Exception:
            pass

    @asynccontextmanager
    async def _stream(
        self, method: str, url: str, *, pool: str | None = None, impersonate: bool | None = None, **kwargs,
    ) -> AsyncIterator[httpx.Response]:
        await self._ensure_probed()
        imp = self._impersonate if impersonate is None else impersonate
        pinned_target = kwargs.pop("impersonate_target", None)
        active_pool = self._resolve_pool(pool)
        pool_name = pool or self._default_pool_name or _DEFAULT
        retrying = self._retry.build_retrying(
            extra_exceptions=[_RetriableStatusError],
        )

        open_stream = None
        proxy = None
        started = 0.0
        try:
            try:
                async for attempt in retrying:
                    with attempt:
                        proxy = None
                        if active_pool:
                            proxy = await active_pool.get_proxy()
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

                        client = self._get_client(proxy, imp)
                        target = await self._pick_target(imp, pinned_target)
                        call_kwargs = kwargs if target is None else {**kwargs, "impersonate_target": target}
                        started = time.monotonic()
                        cm = client.stream(method, url, **call_kwargs)
                        try:
                            response = await cm.__aenter__()
                        except tuple(self._retry.retry_on_exception):
                            duration = time.monotonic() - started
                            self._emit_outcome(pool_name, proxy, "failed", duration)
                            if active_pool and proxy:
                                blacklisted = await active_pool.report_failure(proxy)
                                if blacklisted:
                                    self._log.warning(
                                        "proxy_blacklisted", proxy=proxy
                                    )
                            raise

                        await self._report_target(target, response.status_code)

                        if response.status_code in self._retry.retry_on:
                            await cm.__aexit__(None, None, None)
                            duration = time.monotonic() - started
                            self._emit_outcome(pool_name, proxy, "failed", duration)
                            if active_pool and proxy:
                                blacklisted = await active_pool.report_failure(proxy)
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
                    blacklisted = await active_pool.report_failure(proxy)
                    if blacklisted:
                        self._log.warning("proxy_blacklisted", proxy=proxy)
                raise
            else:
                duration = time.monotonic() - started
                self._emit_outcome(pool_name, proxy, "success", duration)
                if active_pool and proxy:
                    await active_pool.report_success(proxy)
        finally:
            if open_stream is not None:
                await open_stream.__aexit__(None, None, None)

    def stream(
        self, method: str, url: str, *, pool: str | None = None, **kwargs,
    ) -> AsyncIterator[httpx.Response]:
        return self._stream(method, url, pool=pool, **kwargs)

    async def get(self, url: str, *, pool: str | None = None, **kwargs) -> httpx.Response:
        return await self._request("GET", url, pool=pool, **kwargs)

    async def post(self, url: str, *, pool: str | None = None, **kwargs) -> httpx.Response:
        return await self._request("POST", url, pool=pool, **kwargs)

    async def put(self, url: str, *, pool: str | None = None, **kwargs) -> httpx.Response:
        return await self._request("PUT", url, pool=pool, **kwargs)

    async def patch(self, url: str, *, pool: str | None = None, **kwargs) -> httpx.Response:
        return await self._request("PATCH", url, pool=pool, **kwargs)

    async def delete(self, url: str, *, pool: str | None = None, **kwargs) -> httpx.Response:
        return await self._request("DELETE", url, pool=pool, **kwargs)

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()

    async def __aenter__(self) -> AsyncProxyHttpClient:
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()
