"""Process-wide Ollama admission control.

All intelligence and pipeline-fallback calls share one semaphore sized to the
GPU's NUM_PARALLEL slots. When the gate is full the caller fails immediately
(HTTP 429) instead of waiting minutes inside Ollama's FIFO. A consecutive-
failure circuit breaker opens so a sick GPU can recover.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import contextmanager
from typing import Iterator

import config

logger = __import__("logging").getLogger("benchmark.ollama_gate")


class OllamaGateFull(RuntimeError):
    """No GPU slot available right now — caller should back off."""

    def __init__(self, retry_after_s: float = 5.0, message: str = "Ollama gate is full") -> None:
        super().__init__(message)
        self.retry_after_s = max(1.0, float(retry_after_s))


class OllamaCircuitOpen(RuntimeError):
    """Too many recent Ollama failures — stop sending until cooldown."""

    def __init__(self, retry_after_s: float = 30.0, message: str = "Ollama circuit is open") -> None:
        super().__init__(message)
        self.retry_after_s = max(1.0, float(retry_after_s))


class OllamaGate:
    def __init__(
        self,
        *,
        size: int | None = None,
        failure_threshold: int | None = None,
        cooldown_s: float | None = None,
        latency_window: int = 50,
    ) -> None:
        self.size = max(1, int(size if size is not None else config.OLLAMA_GATE_SIZE))
        self.failure_threshold = max(
            2,
            int(
                failure_threshold
                if failure_threshold is not None
                else config.OLLAMA_CIRCUIT_FAILURES
            ),
        )
        configured_cooldown = (
            cooldown_s if cooldown_s is not None else config.OLLAMA_CIRCUIT_COOLDOWN_S
        )
        self.cooldown_s = max(0.05, float(configured_cooldown))
        self._lock = threading.Lock()
        self._in_flight = 0
        self._accepted = 0
        self._rejected = 0
        self._successes = 0
        self._failures = 0
        self._consecutive_failures = 0
        self._opened_at = 0.0
        self._half_open_probe = False
        self._latencies_ms: deque[float] = deque(maxlen=max(10, latency_window))

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

    def acquire(self) -> None:
        now = time.monotonic()
        with self._lock:
            state = self._circuit_unlocked(now)
            if state == "open":
                remaining = max(1.0, self.cooldown_s - (now - self._opened_at))
                self._rejected += 1
                raise OllamaCircuitOpen(
                    retry_after_s=remaining,
                    message=f"Ollama circuit open — retry after {remaining:.0f}s",
                )
            if state == "half_open":
                if self._half_open_probe or self._in_flight > 0:
                    self._rejected += 1
                    raise OllamaCircuitOpen(
                        retry_after_s=5.0,
                        message="Ollama circuit half-open — probe already in flight",
                    )
                self._half_open_probe = True
            elif self._in_flight >= self.size:
                self._rejected += 1
                raise OllamaGateFull(
                    retry_after_s=5.0,
                    message=(
                        f"Ollama gate full ({self._in_flight}/{self.size} in flight)"
                    ),
                )
            self._in_flight += 1
            self._accepted += 1

    def release(self) -> None:
        with self._lock:
            if self._in_flight > 0:
                self._in_flight -= 1

    def record_success(self, latency_ms: float | None = None) -> None:
        with self._lock:
            self._successes += 1
            self._consecutive_failures = 0
            self._opened_at = 0.0
            self._half_open_probe = False
            if latency_ms is not None:
                self._latencies_ms.append(float(latency_ms))

    def record_failure(self) -> None:
        with self._lock:
            self._failures += 1
            self._consecutive_failures += 1
            self._half_open_probe = False
            if self._consecutive_failures >= self.failure_threshold and self._opened_at <= 0:
                self._opened_at = time.monotonic()
                logger.warning(
                    "Ollama circuit OPEN after %d consecutive failures (cooldown %.0fs)",
                    self._consecutive_failures,
                    self.cooldown_s,
                )

    @contextmanager
    def slot(self) -> Iterator[None]:
        started = time.perf_counter()
        self.acquire()
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
                p95 = round(ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))], 1)
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
            }


_GATE: OllamaGate | None = None
_GATE_LOCK = threading.Lock()


def get_ollama_gate() -> OllamaGate:
    global _GATE
    with _GATE_LOCK:
        if _GATE is None:
            _GATE = OllamaGate()
            logger.info(
                "Ollama gate ready: size=%d circuit_failures=%d cooldown=%.0fs",
                _GATE.size,
                _GATE.failure_threshold,
                _GATE.cooldown_s,
            )
        return _GATE


def reset_ollama_gate_for_tests() -> OllamaGate:
    """Test helper — replace the singleton."""
    global _GATE
    with _GATE_LOCK:
        _GATE = OllamaGate()
        return _GATE
