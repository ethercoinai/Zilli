from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Generic, Optional, TypeVar

from zilli.loops.base import (
    EscalationHandler,
    LoopCycle,
    LoopResult,
    Trigger,
    VerificationResult,
    Verifier,
)
from zilli.loops.memory import CycleMemory

logger = logging.getLogger("zilli.loops.runner")

T = TypeVar("T")


class LoopRunner(Generic[T]):
    def __init__(
        self,
        process_fn: Callable[[T], Any],
        verifier: Verifier,
        trigger: Trigger,
        max_retries: int = 3,
        memory: Optional[CycleMemory] = None,
        escalation_handler: Optional[EscalationHandler] = None,
        correction_fn: Optional[Callable[[T, Any, str], T]] = None,
        name: str = "",
    ):
        self._process = process_fn
        self._verifier = verifier
        self._trigger = trigger
        self._max_retries = max_retries
        self._memory = memory or CycleMemory()
        self._escalation = escalation_handler
        self._correction = correction_fn
        self._name = name or "loop"
        self._cycle_count = 0

    async def run(self, input_data: T) -> LoopResult[T]:
        start = time.monotonic()
        cycles: list[LoopCycle[T]] = []
        current_input = input_data

        for attempt in range(self._max_retries + 1):
            self._cycle_count += 1
            cycle_start = time.monotonic()

            cycle = LoopCycle[T](
                id=self._cycle_count,
                input_data=current_input,
            )

            try:
                output = await self._process(current_input)
                cycle.output = output

                if self._memory and self._memory.last():
                    context = self._memory.summary()
                    logger.debug("Cycle %d memory context: %s", cycle.id, context)

                result = await self._verifier.verify(current_input, output)
                cycle.verification = result

                if result.passed:
                    cycle.duration_ms = (time.monotonic() - cycle_start) * 1000
                    cycles.append(cycle)
                    self._memory.add_from_cycle(cycle)
                    logger.info(
                        "%s cycle %d passed (%.0fms)",
                        self._name, cycle.id, cycle.duration_ms,
                    )
                    return LoopResult(
                        success=True,
                        output=output,
                        cycles=cycles,
                        total_duration_ms=(time.monotonic() - start) * 1000,
                        total_retries=attempt,
                    )

                if self._correction:
                    current_input = self._correction(current_input, output, result.evidence)
                    cycle.metadata["corrected_input"] = True

                logger.warning(
                    "%s cycle %d failed (attempt %d/%d): %s",
                    self._name, cycle.id, attempt + 1, self._max_retries + 1,
                    result.evidence,
                )

            except Exception as e:
                cycle.error = str(e)
                cycle.verification = VerificationResult(passed=False, evidence=str(e))
                logger.error("%s cycle %d error: %s", self._name, cycle.id, e)

            cycle.duration_ms = (time.monotonic() - cycle_start) * 1000
            cycles.append(cycle)
            self._memory.add_from_cycle(cycle)

            if attempt < self._max_retries:
                logger.info("%s retrying in 1s...", self._name)
                await asyncio.sleep(1)

        total_ms = (time.monotonic() - start) * 1000
        escalation_data: Optional[dict] = None
        if self._escalation:
            try:
                await self._escalation.escalate(cycles[-1], cycles)
                escalation_data = {"handler": self._escalation.__class__.__name__}
            except Exception as e:
                logger.error("Escalation handler failed: %s", e)

        return LoopResult(
            success=False,
            output=cycles[-1].output if cycles else None,
            cycles=cycles,
            escalated=True,
            escalation_data=escalation_data,
            total_duration_ms=total_ms,
            total_retries=self._max_retries,
        )

    async def run_forever(self, input_data: T) -> None:
        logger.info("%s starting forever loop", self._name)
        while True:
            result = await self.run(input_data)
            if result.success:
                logger.info("%s cycle completed successfully", self._name)
            else:
                logger.warning("%s cycle failed after retries", self._name)
            if not await self._trigger.wait():
                logger.info("%s trigger stopped", self._name)
                break

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def memory(self) -> CycleMemory:
        return self._memory
