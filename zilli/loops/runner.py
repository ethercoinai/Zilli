from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Callable, Generic, Optional, TypeVar

from zilli.loops.base import (
    CostEntry,
    EscalationHandler,
    LoopCycle,
    LoopResult,
    Trigger,
    VerificationResult,
    Verifier,
)
from zilli.loops.context_curator import ContextCurator, Trajectory
from zilli.loops.failure_analyzer import WeaknessMiner
from zilli.loops.harness_orchestrator import HarnessOrchestrator
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
        max_context_cycles: int = 10,
        context_curator: Optional[ContextCurator] = None,
        weakness_miner: Optional[WeaknessMiner] = None,
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
        self._max_context_cycles = max_context_cycles
        self._context_curator = context_curator
        self._weakness_miner = weakness_miner

    async def run(self, input_data: T) -> LoopResult[T]:
        start = time.monotonic()
        cycles: list[LoopCycle[T]] = []
        current_input = input_data
        total_cost = CostEntry()

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
                    context = self._memory.summary(exclude_failed=attempt > 1)
                    logger.debug("Cycle %d memory context: %s", cycle.id, context)

                result = await self._verifier.verify(current_input, output)
                cycle.verification = result
                cycle.duration_ms = (time.monotonic() - cycle_start) * 1000

                if result.passed:
                    cycles.append(cycle)
                    self._memory.add_from_cycle(cycle)
                    self._accumulate_cost(total_cost, cycle)
                    self._curate_trajectory(cycle, "success")
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
                        total_cost=total_cost,
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
                self._curate_trajectory(cycle, "failure")

            cycle.duration_ms = (time.monotonic() - cycle_start) * 1000
            cycles.append(cycle)
            self._memory.add_from_cycle(cycle)
            self._accumulate_cost(total_cost, cycle)

            if attempt < self._max_retries:
                wait = 0.5 if attempt == 0 else 1.0
                logger.info("%s retrying in %.1fs...", self._name, wait)
                await asyncio.sleep(wait)

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
            total_cost=total_cost,
        )

    def _accumulate_cost(self, total: CostEntry, cycle: LoopCycle) -> None:
        total.tokens_input += cycle.cost.tokens_input
        total.tokens_output += cycle.cost.tokens_output
        total.cost_usd += cycle.cost.cost_usd
        total.api_calls += cycle.cost.api_calls

    async def run_forever(self, input_data: T) -> None:
        logger.info("%s starting forever loop", self._name)
        consecutive_failures = 0
        while True:
            result = await self.run(input_data)
            if result.success:
                consecutive_failures = 0
                logger.info("%s cycle completed (cost=$%.4f)", self._name, result.total_cost.cost_usd)
            else:
                consecutive_failures += 1
                logger.warning("%s cycle failed (%dx consecutive)", self._name, consecutive_failures)
                self._mine_failures(result)
            if consecutive_failures >= self._max_retries + 1:
                logger.error("%s too many consecutive failures, stopping", self._name)
                if self._weakness_miner:
                    summary = self._weakness_miner.summary()
                    if summary["clusters"]:
                        logger.error("Failure clusters: %s", summary["clusters"])
                break
            if not await self._trigger.wait():
                logger.info("%s trigger stopped", self._name)
                break

    def _mine_failures(self, result: LoopResult) -> None:
        if not self._weakness_miner:
            return
        traces = []
        for cycle in result.cycles:
            if cycle.verification and not cycle.verification.passed:
                traces.append({
                    "task_id": str(getattr(cycle.input_data, "task_id", cycle.id)),
                    "verifier_outcome": (cycle.verification.evidence or "failed")[:80],
                    "causal_status": cycle.error or "verification_failed",
                    "mechanism": "loop_runner",
                    "trace": (cycle.verification.evidence or "")[:500],
                    "timestamp": time.time(),
                })
        if traces:
            self._weakness_miner.ingest(traces)

    def _curate_trajectory(self, cycle: LoopCycle, outcome: str) -> None:
        if not self._context_curator:
            return
        trajectory = Trajectory(
            task_id=str(getattr(cycle.input_data, "task_id", cycle.id)),
            actions=cycle.metadata.get("actions", []),
            outcome=outcome,
            verifier_evidence=cycle.verification.evidence if cycle.verification else "",
            duration_ms=cycle.duration_ms or 0.0,
        )
        if outcome == "failure" and self._cycle_count % 3 == 0:
            self._context_curator.reflect([trajectory])

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    @property
    def memory(self) -> CycleMemory:
        return self._memory


class MetaLoopRunner:
    """Bilevel meta-loop that harness-evolves a wrapped LoopRunner.

    Two modes:
      PARAM_TUNE (default) — adjusts numeric parameters like the original
      HARNESS_EVOLVE — runs Self-Harness: weakness mining → code-level
                       proposals → held-in/held-out validation

    In HARNESS_EVOLVE mode, failed cycles produce execution traces that
    are fed to the HarnessOrchestrator, which proposes edits to the
    editable harness surfaces.
    """

    MODE_PARAM_TUNE = "param_tune"
    MODE_HARNESS_EVOLVE = "harness_evolve"

    def __init__(
        self,
        inner_runner: LoopRunner,
        meta_verifier: Optional[Verifier] = None,
        max_meta_iterations: int = 5,
        improvement_threshold: float = 0.05,
        mode: str = MODE_HARNESS_EVOLVE,
        harness_orchestrator: Optional[HarnessOrchestrator] = None,
    ):
        self._inner = inner_runner
        self._meta_verifier = meta_verifier
        self._max_meta = max_meta_iterations
        self._threshold = improvement_threshold
        self._mode = mode
        self._orchestrator = harness_orchestrator
        self._history: list[LoopResult] = []
        self._tuning_log: list[dict] = []
        self._evolved_versions: list[str] = []
        self._traces: list[dict] = []

    async def run(self, input_data: Any) -> LoopResult:
        best_result: Optional[LoopResult] = None
        params = self._default_params()

        for iteration in range(self._max_meta):
            logger.info("MetaLoop iteration %d/%d (mode=%s)", iteration + 1, self._max_meta, self._mode)
            result = await self._inner.run(input_data)
            self._history.append(result)
            self._tuning_log.append({
                "iteration": iteration,
                "mode": self._mode,
                "params": params,
                "result": result.success,
            })

            # Track best
            if best_result is None or (result.success and not best_result.success):
                best_result = result
            elif result.success and best_result.success and result.total_retries < best_result.total_retries:
                best_result = result

            # Collect traces from failed cycles for harness evolution
            for cycle in result.cycles:
                if cycle.verification and not cycle.verification.passed:
                    self._traces.append({
                        "task_id": str(getattr(cycle.input_data, "task_id", cycle.id)),
                        "verifier_outcome": cycle.verification.evidence[:80] if cycle.verification.evidence else "failed",
                        "causal_status": cycle.error or "verification_failed",
                        "mechanism": "loop_runner",
                        "trace": cycle.verification.evidence[:500] if cycle.verification.evidence else "",
                        "timestamp": time.time(),
                        "metadata": {
                            "cycle_id": cycle.id,
                            "duration_ms": cycle.duration_ms,
                        },
                    })

            # Harness evolution stage
            if self._mode == self.MODE_HARNESS_EVOLVE and self._orchestrator:
                candidate = await self._orchestrator.run_cycle(self._traces)
                if candidate and candidate.accepted:
                    self._evolved_versions.append(candidate.version)
                    logger.info("Harness evolved to %s", candidate.version)
                    # Reset trace buffer after accepted evolution
                    self._traces = []

            # Parameter tuning always runs
            if result.success:
                pass_rate = self._inner.memory.success_rate(n=20)
                if pass_rate >= 0.9:
                    logger.info("MetaLoop converged (pass_rate=%.2f)", pass_rate)
                    break

            params = self._tune(params, result)
            self._apply_params(params)

        return best_result or LoopResult(success=False, output=None, cycles=[])

    def _default_params(self) -> dict:
        return {
            "max_retries": self._inner._max_retries,
        }

    def _tune(self, params: dict, result: LoopResult) -> dict:
        p = dict(params)
        fail_rate = 1.0 - self._inner.memory.success_rate(n=10)

        if fail_rate > 0.5:
            p["max_retries"] = min(p.get("max_retries", 3) + 1, 10)
        elif fail_rate < 0.2 and p.get("max_retries", 3) > 1:
            p["max_retries"] = max(p["max_retries"] - 1, 1)

        return p

    def _apply_params(self, params: dict) -> None:
        self._inner._max_retries = params.get("max_retries", self._inner._max_retries)

    @property
    def tuning_log(self) -> list[dict]:
        return self._tuning_log

    @property
    def evolved_versions(self) -> list[str]:
        return self._evolved_versions

    @property
    def mode(self) -> str:
        return self._mode
