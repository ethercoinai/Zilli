import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set

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
            original_task_id = str(task.get("id", "")) if isinstance(task, dict) else ""
            task_id = original_task_id or str(uuid.uuid4())
            task_start = time.time()

            if task_id in self._cancelled:
                return RolloutResult(
                    task_id=task_id, trajectory=[], reward=-1.0,
                    tokens=0, completed=False, status=RolloutStatus.CANCELLED,
                    elapsed_sec=time.time() - task_start, retry_count=retry_count,
                )

            if retry_count == 0:
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

        tasks_map: dict[asyncio.Task, Any] = {asyncio.create_task(run_single(t)): t for t in tasks}
        pending_set: set[asyncio.Task] = set(tasks_map.keys())
        if not pending_set:
            return results

        if self.progress_callback:
            self.progress_callback(0, len(tasks), "started")

        done, pending_set = await asyncio.wait(
            pending_set,
            timeout=self.window,
            return_when=asyncio.ALL_COMPLETED,
        )

        for task in done:
            try:
                results.append(task.result())
            except Exception:  # noqa: BLE001
                self._total_errors += 1

        for task in pending_set:
            task.cancel()
            original = tasks_map[task]
            orig_id = str(original.get("id", "")) if isinstance(original, dict) else str(uuid.uuid4())
            result = RolloutResult(
                task_id=orig_id, trajectory=[], reward=-1.0,
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
