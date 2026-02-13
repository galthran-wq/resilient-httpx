from __future__ import annotations

from dataclasses import dataclass, field

import httpx
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
    wait_fixed,
    wait_random,
)


def _default_retry_on_exception() -> list[type[Exception]]:
    return [httpx.TimeoutException, httpx.ConnectError]


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    backoff: str = "exponential"
    min_wait: float = 1.0
    max_wait: float = 30.0
    retry_on: list[int] = field(
        default_factory=lambda: [429, 500, 502, 503, 504],
    )
    retry_on_exception: list[type[Exception]] = field(
        default_factory=_default_retry_on_exception,
    )

    def build_retrying(
        self,
        extra_exceptions: list[type[Exception]] | None = None,
    ) -> AsyncRetrying:
        if self.backoff == "fixed":
            wait = wait_fixed(self.min_wait)
        elif self.backoff == "random_jitter":
            wait = wait_random(min=self.min_wait, max=self.max_wait)
        else:
            wait = wait_exponential(min=self.min_wait, max=self.max_wait)

        exceptions = list(self.retry_on_exception)
        if extra_exceptions:
            exceptions.extend(extra_exceptions)

        return AsyncRetrying(
            stop=stop_after_attempt(self.max_attempts),
            wait=wait,
            retry=retry_if_exception_type(tuple(exceptions)),
        )
