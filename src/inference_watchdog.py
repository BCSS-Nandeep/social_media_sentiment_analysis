"""Hard process deadline for uninterruptible model inference."""
from __future__ import annotations

import logging
import os
import threading
from collections.abc import Callable
from typing import TypeVar


logger = logging.getLogger("sentiment_api.watchdog")
T = TypeVar("T")


def abort_stuck_process() -> None:
    """Terminate so the process manager can release resources and restart."""
    logger.critical(
        "Inference exceeded the hard deadline; terminating process for PM2 restart"
    )
    logging.shutdown()
    os._exit(70)


class InferenceWatchdog:
    """Arm a daemon timer and cancel it when inference returns normally."""

    def __init__(
        self, timeout_s: float, abort: Callable[[], None] = abort_stuck_process
    ) -> None:
        self.timeout_s = float(timeout_s)
        self.abort = abort
        self._timer: threading.Timer | None = None
        self._lock = threading.Lock()
        self._armed = False

    def arm(self) -> None:
        with self._lock:
            if self.timeout_s <= 0 or self._timer is not None:
                return
            self._armed = True
            self._timer = threading.Timer(self.timeout_s, self._fire)
            self._timer.daemon = True
            self._timer.start()

    def _fire(self) -> None:
        with self._lock:
            if not self._armed:
                return
            self._armed = False
        self.abort()

    def cancel(self) -> None:
        with self._lock:
            self._armed = False
            timer, self._timer = self._timer, None
        if timer is not None:
            timer.cancel()
            if timer is not threading.current_thread():
                timer.join()

    def __enter__(self) -> "InferenceWatchdog":
        self.arm()
        return self

    def __exit__(self, *_exc_info) -> None:
        self.cancel()


def guarded_model_call(operation: Callable[[], T], timeout_s: float) -> T:
    """Run one blocking model/CUDA operation under the hard process deadline."""
    with InferenceWatchdog(timeout_s):
        return operation()
