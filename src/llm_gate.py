"""Process-wide LLM admission control for vLLM (and similar) backends.

All intelligence and pipeline-fallback calls share one semaphore sized to the
backend's parallel-slot budget. A consecutive-failure circuit breaker opens
so a sick backend can recover — this is a health signal, shared by everyone,
unchanged by the tenant-fairness logic below.

Tenant fairness: acquire()/slot() take an optional tenant_key. When a slot
frees and more than one tenant currently has a caller waiting, the next slot
goes to whichever tenant is next in round-robin rotation, not whoever asked
first — so one tenant's backlog cannot monopolize the shared gate while
another tenant has a request waiting. Callers that never pass tenant_key
(pipeline_fallback.py today) land in one shared "unknown" bucket and behave
exactly as before: first-come-first-served against whatever's left after
named tenants take their turn.

When there's no contention (the common case — usually 0-1 tenants waiting at
once), a caller is admitted immediately, same latency as the pre-fairness
gate. The wait/rotation machinery only engages when two or more tenants are
genuinely contending for the same slot at the same moment.

ponytail: no per-tenant cooldown/backoff after repeated failures for one
tenant — round-robin admission alone already stops a struggling tenant from
starving others, which is the actual fairness requirement. Add a per-tenant
next_eligible_at map if a persistently-failing tenant needs to stop even
*trying* (not just stop *monopolizing*) — no evidence that's needed yet.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Iterator

import config

logger = __import__("logging").getLogger("benchmark.llm_gate")

UNKNOWN_TENANT = "unknown"


class LlmGateFull(RuntimeError):
    """No LLM slot available right now — caller should back off."""

    def __init__(self, retry_after_s: float = 5.0, message: str = "LLM gate is full") -> None:
        super().__init__(message)
        self.retry_after_s = max(1.0, float(retry_after_s))


class LlmCircuitOpen(RuntimeError):
    """Too many recent LLM failures — stop sending until cooldown."""

    def __init__(
        self, retry_after_s: float = 30.0, message: str = "LLM circuit is open"
    ) -> None:
        super().__init__(message)
        self.retry_after_s = max(1.0, float(retry_after_s))


class LlmGate:
    def __init__(
        self,
        *,
        size: int | None = None,
        failure_threshold: int | None = None,
        cooldown_s: float | None = None,
        wait_timeout_s: float | None = None,
        max_tracked_tenants: int | None = None,
        latency_window: int = 50,
    ) -> None:
        self.size = max(1, int(size if size is not None else config.VLLM_GATE_SIZE))
        self.failure_threshold = max(
            2,
            int(
                failure_threshold
                if failure_threshold is not None
                else config.VLLM_CIRCUIT_FAILURES
            ),
        )
        configured_cooldown = (
            cooldown_s if cooldown_s is not None else config.VLLM_CIRCUIT_COOLDOWN_S
        )
        self.cooldown_s = max(0.05, float(configured_cooldown))
        self.wait_timeout_s = max(
            1.0,
            float(
                wait_timeout_s
                if wait_timeout_s is not None
                else config.VLLM_GATE_WAIT_TIMEOUT_S
            ),
        )
        self.max_tracked_tenants = max(
            1,
            int(
                max_tracked_tenants
                if max_tracked_tenants is not None
                else config.TENANT_GATE_MAX_TRACKED
            ),
        )
        # cond guards everything below, including the plain LlmGate fields —
        # acquire() needs to wait/wake on the same lock the old code used.
        self._cond = threading.Condition()
        self._lock = self._cond  # acquire()/stats() below use "with self._lock:"
        self._in_flight = 0
        self._accepted = 0
        self._rejected = 0
        self._successes = 0
        self._failures = 0
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._half_open_probe = False
        self._latencies_ms: deque[float] = deque(maxlen=max(10, latency_window))

        # Fairness state. _rotation is every tenant_key ever seen, in a fixed
        # round-robin cycle; _cursor is where the next scan starts.
        self._rotation: list[str] = []
        self._known: set[str] = set()
        self._waiting: dict[str, int] = {}
        self._cursor = 0

    @property
    def circuit(self) -> str:
        with self._lock:
            return self._circuit_unlocked(time.monotonic())

    def _circuit_unlocked(self, now: float) -> str:
        if self._opened_at <= 0:
            return "closed"
        elapsed = now - self._opened_at
        if elapsed >= self.cooldown_s:
            return "half_open"
        return "open"

    def _register_tenant_unlocked(self, tenant_key: str) -> None:
        if tenant_key in self._known:
            return
        if len(self._known) >= self.max_tracked_tenants:
            logger.warning(
                "Tenant gate at max_tracked_tenants=%d — %r folded into %r",
                self.max_tracked_tenants, tenant_key, UNKNOWN_TENANT,
            )
            return
        self._known.add(tenant_key)
        self._rotation.append(tenant_key)
        logger.info("Tenant queue created: %s", tenant_key)

    def _next_ready_tenant_unlocked(self) -> str | None:
        """First tenant at/after the cursor with a waiter, without side
        effects — cursor only moves when the tenant this function names
        actually claims the slot (done by the caller, in acquire(), not
        here). Returns None if nobody is currently waiting."""
        n = len(self._rotation)
        if n == 0:
            return None
        for i in range(n):
            idx = (self._cursor + i) % n
            candidate = self._rotation[idx]
            if self._waiting.get(candidate, 0) > 0:
                return candidate
        return None

    def _advance_cursor_past_unlocked(self, tenant_key: str) -> None:
        try:
            idx = self._rotation.index(tenant_key)
        except ValueError:
            return
        self._cursor = (idx + 1) % len(self._rotation)

    def acquire(self, tenant_key: str | None = None) -> None:
        key = (tenant_key or UNKNOWN_TENANT).strip() or UNKNOWN_TENANT
        deadline = time.monotonic() + self.wait_timeout_s
        with self._cond:
            self._register_tenant_unlocked(key)
            self._waiting[key] = self._waiting.get(key, 0) + 1
            try:
                while True:
                    now = time.monotonic()
                    state = self._circuit_unlocked(now)

                    if state == "open":
                        remaining = max(1.0, self.cooldown_s - (now - self._opened_at))
                        self._rejected += 1
                        raise LlmCircuitOpen(
                            retry_after_s=remaining,
                            message=f"LLM circuit open — retry after {remaining:.0f}s",
                        )

                    if state == "half_open":
                        # A health probe is a single admission event, not a
                        # fairness-sensitive one — whoever's earliest in the
                        # rotation with a waiter gets it, same as before.
                        # A second concurrent probe is rejected immediately
                        # (unchanged contract), not queued behind a timeout.
                        if self._half_open_probe or self._in_flight > 0:
                            self._rejected += 1
                            raise LlmCircuitOpen(
                                retry_after_s=5.0,
                                message="LLM circuit half-open — probe already in flight",
                            )
                        turn = self._next_ready_tenant_unlocked()
                        if turn == key:
                            self._half_open_probe = True
                            self._in_flight += 1
                            self._accepted += 1
                            self._advance_cursor_past_unlocked(key)
                            return
                    elif self._in_flight < self.size:
                        turn = self._next_ready_tenant_unlocked()
                        if turn == key:
                            self._in_flight += 1
                            self._accepted += 1
                            self._advance_cursor_past_unlocked(key)
                            return

                    remaining_wait = deadline - now
                    if remaining_wait <= 0:
                        self._rejected += 1
                        raise LlmGateFull(
                            retry_after_s=5.0,
                            message=(
                                f"LLM gate full ({self._in_flight}/{self.size} "
                                f"in flight, tenant={key} waiting)"
                            ),
                        )
                    # Re-check at least once a second even without a notify,
                    # so a rotation change from another thread's claim is
                    # never missed for long.
                    self._cond.wait(min(remaining_wait, 1.0))
            finally:
                self._waiting[key] = max(0, self._waiting.get(key, 1) - 1)

    def release(self) -> None:
        with self._cond:
            if self._in_flight > 0:
                self._in_flight -= 1
            self._cond.notify_all()

    def record_success(self, latency_ms: float | None = None) -> None:
        with self._cond:
            self._successes += 1
            self._consecutive_failures = 0
            self._opened_at = 0.0
            self._half_open_probe = False
            if latency_ms is not None:
                self._latencies_ms.append(float(latency_ms))
            self._cond.notify_all()

    def record_failure(self) -> None:
        with self._cond:
            self._failures += 1
            self._consecutive_failures += 1
            self._half_open_probe = False
            if self._consecutive_failures >= self.failure_threshold and self._opened_at <= 0:
                self._opened_at = time.monotonic()
                logger.warning(
                    "LLM circuit OPEN after %d consecutive failures (cooldown %.0fs)",
                    self._consecutive_failures,
                    self.cooldown_s,
                )
            self._cond.notify_all()

    @contextmanager
    def slot(self, tenant_key: str | None = None) -> Iterator[None]:
        started = time.perf_counter()
        self.acquire(tenant_key)
        try:
            yield
        except Exception:
            self.record_failure()
            raise
        else:
            self.record_success((time.perf_counter() - started) * 1000.0)
        finally:
            self.release()

    def stats(self) -> dict:
        with self._lock:
            now = time.monotonic()
            state = self._circuit_unlocked(now)
            lat = list(self._latencies_ms)
            p50 = p95 = None
            if lat:
                ordered = sorted(lat)
                p50 = round(ordered[len(ordered) // 2], 1)
                p95 = round(
                    ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))],
                    1,
                )
            by_tenant = {
                t: {"waiting": self._waiting.get(t, 0)} for t in self._known
            }
            return {
                "size": self.size,
                "in_flight": self._in_flight,
                "accepted": self._accepted,
                "rejected": self._rejected,
                "successes": self._successes,
                "failures": self._failures,
                "consecutive_failures": self._consecutive_failures,
                "circuit": state,
                "cooldown_s": self.cooldown_s,
                "failure_threshold": self.failure_threshold,
                "latency_ms_p50": p50,
                "latency_ms_p95": p95,
                "latency_samples": len(lat),
                "active_tenants": len(self._known),
                "by_tenant": by_tenant,
            }


_GATE: LlmGate | None = None
_GATE_LOCK = threading.Lock()


def get_llm_gate() -> LlmGate:
    global _GATE
    with _GATE_LOCK:
        if _GATE is None:
            _GATE = LlmGate()
            logger.info(
                "LLM gate ready: size=%d circuit_failures=%d cooldown=%.0fs",
                _GATE.size,
                _GATE.failure_threshold,
                _GATE.cooldown_s,
            )
        return _GATE


def reset_llm_gate_for_tests() -> LlmGate:
    """Test helper — replace the singleton."""
    global _GATE
    with _GATE_LOCK:
        _GATE = LlmGate()
        return _GATE
