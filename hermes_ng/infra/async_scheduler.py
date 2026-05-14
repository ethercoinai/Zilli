import asyncio
import time
import uuid
from typing import List, Any, Callable, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class RolloutResult:
    task_id: str
    trajectory: List[Dict]
    reward: float
    tokens: int
    completed: bool
    error: Optional[str] = None


class AsyncRolloutScheduler:
    def __init__(self, window_sec: int = 60):
        self.window = window_sec
        self.pending_rollouts: Dict[str, asyncio.Task] = {}

    async def schedule(
        self,
        rollout_fn: Callable,
        tasks: List[Any],
        timeout_per_task: int = 300,
    ) -> List[RolloutResult]:
        results: List[RolloutResult] = []
        batch_start = time.time()

        async def run_single(task) -> RolloutResult:
            task_id = str(uuid.uuid4())
            try:
                result = await asyncio.wait_for(
                    rollout_fn(task), timeout=timeout_per_task
                )
                return RolloutResult(
                    task_id=task_id,
                    trajectory=result.get("trajectory", []),
                    reward=result.get("reward", 0.0),
                    tokens=result.get("tokens", 0),
                    completed=True,
                )
            except asyncio.TimeoutError:
                return RolloutResult(
                    task_id=task_id,
                    trajectory=[],
                    reward=-1.0,
                    tokens=0,
                    completed=False,
                    error="timeout",
                )
            except Exception as e:
                return RolloutResult(
                    task_id=task_id,
                    trajectory=[],
                    reward=-1.0,
                    tokens=0,
                    completed=False,
                    error=str(e),
                )

        pending = [run_single(t) for t in tasks]

        done, pending_futures = await asyncio.wait(
            pending,
            timeout=self.window,
            return_when=asyncio.FIRST_COMPLETED,
        )

        for fut in done:
            try:
                results.append(fut.result())
            except Exception as e:
                pass

        for fut in pending_futures:
            try:
                result = await fut
            except (asyncio.CancelledError, Exception):
                result = RolloutResult(
                    task_id=str(uuid.uuid4()),
                    trajectory=[],
                    reward=-1.0,
                    tokens=0,
                    completed=False,
                    error="window_closed",
                )
            results.append(result)

        return results
