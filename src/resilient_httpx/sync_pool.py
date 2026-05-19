from __future__ import annotations

import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass


@dataclass
class _ProxyState:
    fail_count: int = 0
    blacklisted_until: float | None = None
    blacklisted_at: float | None = None
    total_ok: int = 0
    total_fail: int = 0


class SyncProxyPool:
    def __init__(
        self,
        proxies: list[str],
        strategy: str = "round-robin",
        blacklist_threshold: int = 3,
        blacklist_ttl: float = 300.0,
        state: dict[str, _ProxyState] | None = None,
        on_blacklist: Callable[[str], None] | None = None,
        on_unblacklist: Callable[[str], None] | None = None,
    ) -> None:
        self._proxies = list(proxies)
        self._strategy = strategy
        self._threshold = blacklist_threshold
        self._ttl = blacklist_ttl
        existing_state = state or {}
        self._state = {
            p: existing_state[p] if p in existing_state else _ProxyState()
            for p in self._proxies
        }
        self._index = 0
        self._lock = threading.Lock()
        self._on_blacklist = on_blacklist
        self._on_unblacklist = on_unblacklist

    def _take_if_available(self, proxy: str) -> tuple[bool, bool]:
        """Mutating helper: returns (is_available, just_recovered). Must hold _lock."""
        state = self._state[proxy]
        if state.blacklisted_until is None:
            return True, False
        if time.monotonic() >= state.blacklisted_until:
            state.blacklisted_until = None
            state.blacklisted_at = None
            # Reset to 0 — a single transient failure on a recovered proxy
            # must not immediately re-blacklist it.
            state.fail_count = 0
            return True, True
        return False, False

    def _is_in_rotation(self, proxy: str) -> bool:
        state = self._state[proxy]
        if state.blacklisted_until is None:
            return True
        return time.monotonic() >= state.blacklisted_until

    def get_proxy(self) -> str | None:
        unblacklisted: list[str] = []
        with self._lock:
            available: list[str] = []
            for p in self._proxies:
                ok, recovered = self._take_if_available(p)
                if ok:
                    available.append(p)
                if recovered:
                    unblacklisted.append(p)
            if not available:
                chosen = None
            elif self._strategy == "random":
                chosen = random.choice(available)
            else:
                while True:
                    proxy = self._proxies[self._index % len(self._proxies)]
                    self._index += 1
                    if proxy in available:
                        chosen = proxy
                        break

        for p in unblacklisted:
            self._safe_cb(self._on_unblacklist, p)
        return chosen

    def report_failure(self, proxy: str) -> bool:
        blacklisted = False
        with self._lock:
            state = self._state[proxy]
            state.fail_count += 1
            state.total_fail += 1
            if state.fail_count >= self._threshold and state.blacklisted_until is None:
                now = time.monotonic()
                state.blacklisted_at = now
                state.blacklisted_until = now + self._ttl
                blacklisted = True
        if blacklisted:
            self._safe_cb(self._on_blacklist, proxy)
        return blacklisted

    def report_success(self, proxy: str) -> None:
        with self._lock:
            state = self._state[proxy]
            state.fail_count = 0
            state.blacklisted_until = None
            state.blacklisted_at = None
            state.total_ok += 1

    def blacklist(self, proxy: str) -> None:
        """Force-blacklist a proxy (used by probe_on_start). Fires on_blacklist."""
        fired = False
        with self._lock:
            state = self._state[proxy]
            if state.blacklisted_until is None:
                now = time.monotonic()
                state.fail_count = self._threshold
                state.blacklisted_at = now
                state.blacklisted_until = now + self._ttl
                fired = True
        if fired:
            self._safe_cb(self._on_blacklist, proxy)

    def snapshot(self) -> list[dict]:
        # Read-only — no mutation, no callbacks. May briefly show a recovered
        # proxy as still-blacklisted until the next get_proxy() actually
        # rotates it back in; that's intentional (matches what a request
        # would see at this instant).
        now = time.monotonic()
        out = []
        for p, s in self._state.items():
            ttl_remaining = None
            in_rotation = True
            if s.blacklisted_until is not None:
                ttl_remaining = max(0.0, s.blacklisted_until - now)
                in_rotation = now >= s.blacklisted_until
            out.append({
                "proxy": p,
                "fail_count": s.fail_count,
                "in_rotation": in_rotation,
                "blacklisted_at_monotonic": s.blacklisted_at,
                "ttl_remaining_seconds": ttl_remaining,
                "total_ok": s.total_ok,
                "total_fail": s.total_fail,
            })
        return out

    @staticmethod
    def _safe_cb(cb: Callable[[str], None] | None, proxy: str) -> None:
        if cb is None:
            return
        try:
            cb(proxy)
        except Exception:
            pass

    @property
    def active_count(self) -> int:
        return sum(1 for p in self._proxies if self._is_in_rotation(p))

    @property
    def total_count(self) -> int:
        return len(self._proxies)
