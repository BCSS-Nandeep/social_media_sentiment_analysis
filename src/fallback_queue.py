"""Bounded FIFO worker queue for Ollama pipeline fallback calls."""
from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Any, Callable


class FallbackQueueFull(RuntimeError):
    """The fallback queue did not accept work within its enqueue deadline."""


@dataclass
class _WorkItem:
    future: Future
    operation: Callable
    args: tuple
    kwargs: dict[str, Any]


class BoundedWorkQueue:
    """Fixed workers over a bounded, process-wide FIFO queue."""

    def __init__(
        self,
        *,
        workers: int,
        capacity: int,
        thread_name_prefix: str,
    ) -> None:
        self.workers = max(1, int(workers))
        self.capacity = max(1, int(capacity))
        self._pending: deque[_WorkItem] = deque()
        self._condition = threading.Condition()
        self._stopped = False
        self._active = 0
        self._accepted = 0
        self._rejected = 0
        self._cancelled = 0
        self._completed = 0
        self._threads = [
            threading.Thread(
                target=self._worker,
                name=f"{thread_name_prefix}-{index + 1}",
                daemon=True,
            )
            for index in range(self.workers)
        ]
        for thread in self._threads:
            thread.start()

    def submit(
        self,
        operation: Callable,
        *args,
        timeout_s: float = 0.0,
        **kwargs,
    ) -> Future:
        return self.submit_many(
            [(operation, args, kwargs)], timeout_s=timeout_s
        )[0]

    def submit_many(
        self,
        jobs: list[tuple[Callable, tuple, dict[str, Any]]],
        *,
        timeout_s: float = 0.0,
    ) -> list[Future]:
        """Atomically enqueue every job, or reject the whole batch."""
        if not jobs:
            return []
        if len(jobs) > self.capacity:
            with self._condition:
                self._rejected += len(jobs)
            raise FallbackQueueFull(
                "Ollama fallback batch exceeds queue capacity"
            )

        deadline = time.monotonic() + max(0.0, float(timeout_s))
        with self._condition:
            while len(self._pending) + len(jobs) > self.capacity:
                if self._stopped:
                    raise RuntimeError("Fallback queue has been shut down")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._rejected += len(jobs)
                    raise FallbackQueueFull(
                        "Ollama fallback queue is at capacity"
                    )
                self._condition.wait(remaining)
            if self._stopped:
                raise RuntimeError("Fallback queue has been shut down")

            futures: list[Future] = []
            for operation, args, kwargs in jobs:
                future = Future()
                item = _WorkItem(future, operation, args, kwargs)
                future.add_done_callback(
                    lambda done, queued=item: self._remove_cancelled(
                        done, queued
                    )
                )
                self._pending.append(item)
                futures.append(future)
            self._accepted += len(jobs)
            self._condition.notify_all()
            return futures

    def _remove_cancelled(self, future: Future, item: _WorkItem) -> None:
        if not future.cancelled():
            return
        with self._condition:
            try:
                self._pending.remove(item)
            except ValueError:
                return
            self._cancelled += 1
            self._condition.notify_all()

    def _worker(self) -> None:
        while True:
            with self._condition:
                while not self._pending:
                    if self._stopped:
                        return
                    self._condition.wait()
                item = self._pending.popleft()
                if not item.future.set_running_or_notify_cancel():
                    self._cancelled += 1
                    self._condition.notify_all()
                    continue
                self._active += 1
                self._condition.notify_all()
            try:
                result = item.operation(*item.args, **item.kwargs)
            except BaseException as exc:
                item.future.set_exception(exc)
            else:
                item.future.set_result(result)
            finally:
                with self._condition:
                    self._active -= 1
                    self._completed += 1
                    self._condition.notify_all()

    def stats(self) -> dict[str, int]:
        with self._condition:
            return {
                "workers": self.workers,
                "capacity": self.capacity,
                "queued": len(self._pending),
                "active": self._active,
                "accepted": self._accepted,
                "rejected": self._rejected,
                "cancelled": self._cancelled,
                "completed": self._completed,
            }

    def shutdown(self) -> None:
        with self._condition:
            if self._stopped:
                return
            self._stopped = True
            self._condition.notify_all()
        for thread in self._threads:
            thread.join()
