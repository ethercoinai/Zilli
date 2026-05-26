import asyncio
import time
import uuid
import logging
from typing import List, Any, Callable, Dict, Optional, Set
from dataclasses import dataclass, field
from enum import Enum, auto


logger = logging.getLogger("zilli.scheduler")


class RolloutStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    TIMEOUT = auto()
    FAILED = auto()
    CANCELLED = auto()


@dataclass
class RolloutResult:
    task_id: str
    trajectory: List[Dict]
    reward: float
    tokens: int
    completed: bool
    error: Optional[str] = None
    status: RolloutStatus = RolloutStatus.COMPLETED
    elapsed_sec: float = 0.0
    retry_count: int = 0


class AsyncRolloutScheduler:
    def __init__(self, window_sec: int = 60, max_retries: int = 2,
                 progress_callback: Optional[Callable] = None):
        self.window = window_sec
        self.max_retries = max_retries
        self.progress_callback = progress_callback
        self.pending_rollouts: Dict[str, asyncio.Task] = {}
        self._cancelled: Set[str] = set()
        self._total_scheduled = 0
        self._total_completed = 0
        self._total_errors = 0

    async def schedule(
        self,
        rollout_fn: Callable,
        tasks: List[Any],
        timeout_per_task: int = 300,
    ) -> List[RolloutResult]:
        results: List[RolloutResult] = []
        batch_start = time.time()

        async def run_single(task, retry_count: int = 0) -> RolloutResult:
            task_id = str(uuid.uuid4())
            task_start = time.time()

            if task_id in self._cancelled:
                return RolloutResult(
                    task_id=task_id, trajectory=[], reward=-1.0,
                    tokens=0, completed=False, status=RolloutStatus.CANCELLED,
                    elapsed_sec=time.time() - task_start, retry_count=retry_count,
                )

            self._total_scheduled += 1
            try:
                result = await asyncio.wait_for(
                    rollout_fn(task), timeout=timeout_per_task
                )
                elapsed = time.time() - task_start
                self._total_completed += 1
                return RolloutResult(
                    task_id=task_id,
                    trajectory=result.get("trajectory", []),
                    reward=result.get("reward", 0.0),
                    tokens=result.get("tokens", 0),
                    completed=True,
                    status=RolloutStatus.COMPLETED,
                    elapsed_sec=elapsed,
                    retry_count=retry_count,
                )
            except asyncio.TimeoutError:
                self._total_errors += 1
                if retry_count < self.max_retries:
                    logger.info("Retrying task (attempt %d/%d)", retry_count + 1, self.max_retries)
                    return await run_single(task, retry_count + 1)
                return RolloutResult(
                    task_id=task_id, trajectory=[], reward=-1.0,
                    tokens=0, completed=False, status=RolloutStatus.TIMEOUT,
                    error="timeout", elapsed_sec=time.time() - task_start,
                    retry_count=retry_count,
                )
            except Exception as e:
                self._total_errors += 1
                if retry_count < self.max_retries:
                    logger.info("Retrying task after error (attempt %d/%d)", retry_count + 1, self.max_retries)
                    return await run_single(task, retry_count + 1)
                return RolloutResult(
                    task_id=task_id, trajectory=[], reward=-1.0,
                    tokens=0, completed=False, status=RolloutStatus.FAILED,
                    error=str(e), elapsed_sec=time.time() - task_start,
                    retry_count=retry_count,
                )

        pending = [asyncio.ensure_future(run_single(t)) for t in tasks]
        if not pending:
            return results

        if self.progress_callback:
            self.progress_callback(0, len(tasks), "started")

        done, pending_futures = await asyncio.wait(
            pending,
            timeout=self.window,
            return_when=asyncio.ALL_COMPLETED,
        )

        for fut in done:
            try:
                results.append(fut.result())
            except Exception as e:
                self._total_errors += 1

        for fut in pending_futures:
            try:
                result = await asyncio.wait_for(fut, timeout=10)
            except (asyncio.CancelledError, asyncio.TimeoutError, Exception):
                result = RolloutResult(
                    task_id=str(uuid.uuid4()), trajectory=[], reward=-1.0,
                    tokens=0, completed=False, status=RolloutStatus.CANCELLED,
                    error="window_closed",
                )
            results.append(result)

        if self.progress_callback:
            self.progress_callback(len(results), len(tasks), "completed")

        total_elapsed = time.time() - batch_start
        logger.info(
            "Batch: %d/%d done, %d errors, %.1fs elapsed",
            self._total_completed, self._total_scheduled, self._total_errors, total_elapsed,
        )
        return results

    def cancel(self, task_id: str):
        self._cancelled.add(task_id)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_scheduled": self._total_scheduled,
            "total_completed": self._total_completed,
            "total_errors": self._total_errors,
            "pending_rollouts": len(self.pending_rollouts),
            "cancelled_count": len(self._cancelled),
        }


__all__ = ["AsyncRolloutScheduler", "RolloutResult", "RolloutStatus"]
