from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger("zilli.core.runner")


@dataclass
class TaskStep:
    name: str
    fn: Callable[[], Any]
    depends_on: list[str] = field(default_factory=list)
    retries: int = 2
    timeout_s: float = 60.0


@dataclass
class StepResult:
    name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0


class TaskRunner:
    def __init__(self, max_concurrency: int = 4):
        self._sem = asyncio.Semaphore(max_concurrency)

    async def run(self, steps: list[TaskStep]) -> list[StepResult]:
        results: dict[str, StepResult] = {}
        done: set[str] = set()

        while len(done) < len(steps):
            ready = [s for s in steps if s.name not in done and all(d in done for d in s.depends_on)]
            if not ready:
                blocked = [s.name for s in steps if s.name not in done]
                logger.warning("Deadlock detected: %s", blocked)
                break

            tasks = [self._run_one(s) for s in ready]
            for r in await asyncio.gather(*tasks):
                results[r.name] = r
                done.add(r.name)

        return [results[s.name] for s in steps]

    async def _run_one(self, step: TaskStep) -> StepResult:
        start = time.monotonic()
        for attempt in range(step.retries + 1):
            try:
                async with self._sem:
                    if asyncio.iscoroutinefunction(step.fn):
                        result = await asyncio.wait_for(step.fn(), timeout=step.timeout_s)
                    else:
                        result = await asyncio.wait_for(asyncio.to_thread(step.fn), timeout=step.timeout_s)
                return StepResult(
                    name=step.name, success=True, result=result,
                    duration_ms=(time.monotonic() - start) * 1000,
                )
            except Exception as e:
                if attempt < step.retries:
                    await asyncio.sleep(1 * (2 ** attempt))
                    continue
                return StepResult(
                    name=step.name, success=False, error=str(e),
                    duration_ms=(time.monotonic() - start) * 1000,
                )


__all__ = ["TaskRunner", "TaskStep", "StepResult"]
