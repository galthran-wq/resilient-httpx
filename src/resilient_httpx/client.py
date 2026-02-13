from __future__ import annotations

import httpx
import structlog
from tenacity import RetryError

from resilient_httpx.exceptions import AllProxiesExhausted, MaxRetriesExceeded
from resilient_httpx.pool import ProxyPool
from resilient_httpx.retry import RetryPolicy


class _RetriableStatusError(Exception):
    def __init__(self, response: httpx.Response) -> None:
        self.response = response
        super().__init__(f"HTTP {response.status_code}")


class ProxyHttpClient:
    def __init__(
        self,
        proxies: list[str] | None = None,
        proxy_strategy: str = "round-robin",
        retry: RetryPolicy | None = None,
        timeout: float = 30.0,
        headers: dict[str, str] | None = None,
        blacklist_threshold: int = 3,
        blacklist_ttl: float = 300.0,
        fallback_to_direct: bool = False,
    ) -> None:
        self._pool = (
            ProxyPool(
                proxies=proxies,
                strategy=proxy_strategy,
                blacklist_threshold=blacklist_threshold,
                blacklist_ttl=blacklist_ttl,
            )
            if proxies
            else None
        )
        self._retry = retry or RetryPolicy()
        self._timeout = timeout
        self._headers = headers or {}
        self._fallback = fallback_to_direct
        self._clients: dict[str | None, httpx.AsyncClient] = {}
        self._log = structlog.get_logger()

    def _get_client(self, proxy: str | None) -> httpx.AsyncClient:
        if proxy not in self._clients:
            self._clients[proxy] = httpx.AsyncClient(
                proxy=proxy,
                timeout=self._timeout,
                headers=self._headers,
            )
        return self._clients[proxy]

    async def _request(self, method: str, url: str, **kwargs) -> httpx.Response:
        retrying = self._retry.build_retrying(
            extra_exceptions=[_RetriableStatusError],
        )

        try:
            async for attempt in retrying:
                with attempt:
                    proxy = None
                    if self._pool:
                        proxy = await self._pool.get_proxy()
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
                        response = await client.request(method, url, **kwargs)
                    except tuple(self._retry.retry_on_exception):
                        if self._pool and proxy:
                            blacklisted = await self._pool.report_failure(proxy)
                            if blacklisted:
                                self._log.warning(
                                    "proxy_blacklisted", proxy=proxy
                                )
                        raise

                    if response.status_code in self._retry.retry_on:
                        if self._pool and proxy:
                            blacklisted = await self._pool.report_failure(proxy)
                            if blacklisted:
                                self._log.warning(
                                    "proxy_blacklisted", proxy=proxy
                                )
                        raise _RetriableStatusError(response)

                    if self._pool and proxy:
                        await self._pool.report_success(proxy)
                    return response
        except AllProxiesExhausted:
            self._log.error("all_proxies_exhausted")
            raise
        except RetryError as exc:
            last = exc.last_attempt.exception()
            self._log.error("max_retries_exceeded", last_error=str(last))
            raise MaxRetriesExceeded(str(last)) from last

    async def get(self, url: str, **kwargs) -> httpx.Response:
        return await self._request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs) -> httpx.Response:
        return await self._request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs) -> httpx.Response:
        return await self._request("PUT", url, **kwargs)

    async def patch(self, url: str, **kwargs) -> httpx.Response:
        return await self._request("PATCH", url, **kwargs)

    async def delete(self, url: str, **kwargs) -> httpx.Response:
        return await self._request("DELETE", url, **kwargs)

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()

    async def __aenter__(self) -> ProxyHttpClient:
        return self

    async def __aexit__(self, *args) -> None:
        await self.aclose()
