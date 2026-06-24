from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Optional

from zilli.loops.base import Trigger

logger = logging.getLogger("zilli.loops.trigger")


class FixedIntervalTrigger(Trigger):
    def __init__(self, interval_seconds: float, jitter: float = 0.0):
        if interval_seconds <= 0:
            raise ValueError("interval_seconds must be positive")
        self._interval = interval_seconds
        self._jitter = jitter

    async def wait(self) -> bool:
        import random
        jitter = random.uniform(-self._jitter, self._jitter) if self._jitter else 0.0
        await asyncio.sleep(max(0, self._interval + jitter))
        return True

    async def reset(self) -> None:
        pass


class EventTrigger(Trigger):
    def __init__(self, check_fn: Callable[[], bool], poll_interval: float = 1.0, timeout: Optional[float] = None):
        self._check = check_fn
        self._poll = poll_interval
        self._timeout = timeout
        self._start: float = 0.0

    async def wait(self) -> bool:
        self._start = time.monotonic()
        while True:
            if self._check():
                return True
            if self._timeout and (time.monotonic() - self._start) >= self._timeout:
                logger.warning("EventTrigger timed out after %.1fs", self._timeout)
                return False
            await asyncio.sleep(self._poll)

    async def reset(self) -> None:
        self._start = 0.0


class DynamicIntervalTrigger(Trigger):
    def __init__(self, min_interval: float = 60.0, max_interval: float = 3600.0,
                 interval_fn: Optional[Callable[[dict], float]] = None):
        self._min = min_interval
        self._max = max_interval
        self._interval_fn = interval_fn
        self._last_state: dict = {}

    async def wait(self) -> bool:
        if self._interval_fn:
            interval = self._interval_fn(self._last_state)
        else:
            interval = self._min
        interval = max(self._min, min(interval, self._max))
        await asyncio.sleep(interval)
        return True

    async def reset(self) -> None:
        self._last_state = {}

    def update_state(self, state: dict) -> None:
        self._last_state = state
