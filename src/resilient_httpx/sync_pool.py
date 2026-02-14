from __future__ import annotations

import random
import threading
import time
from dataclasses import dataclass, replace


@dataclass
class _ProxyState:
    fail_count: int = 0
    blacklisted_until: float | None = None


class SyncProxyPool:
    def __init__(
        self,
        proxies: list[str],
        strategy: str = "round-robin",
        blacklist_threshold: int = 3,
        blacklist_ttl: float = 300.0,
        state: dict[str, _ProxyState] | None = None,
    ) -> None:
        self._proxies = list(proxies)
        self._strategy = strategy
        self._threshold = blacklist_threshold
        self._ttl = blacklist_ttl
        existing_state = state or {}
        self._state = {
            p: replace(existing_state[p]) if p in existing_state else _ProxyState()
            for p in self._proxies
        }
        self._index = 0
        self._lock = threading.Lock()

    def _is_available(self, proxy: str) -> bool:
        state = self._state[proxy]
        if state.blacklisted_until is None:
            return True
        if time.monotonic() >= state.blacklisted_until:
            state.blacklisted_until = None
            state.fail_count = self._threshold - 1
            return True
        return False

    def get_proxy(self) -> str | None:
        with self._lock:
            available = [p for p in self._proxies if self._is_available(p)]
            if not available:
                return None

            if self._strategy == "random":
                return random.choice(available)

            while True:
                proxy = self._proxies[self._index % len(self._proxies)]
                self._index += 1
                if proxy in available:
                    return proxy

    def report_failure(self, proxy: str) -> bool:
        with self._lock:
            state = self._state[proxy]
            state.fail_count += 1
            if state.fail_count >= self._threshold:
                state.blacklisted_until = time.monotonic() + self._ttl
                return True
            return False

    def report_success(self, proxy: str) -> None:
        with self._lock:
            state = self._state[proxy]
            state.fail_count = 0
            state.blacklisted_until = None
