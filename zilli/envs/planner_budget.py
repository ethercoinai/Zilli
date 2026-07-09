from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Any

logger = logging.getLogger("zilli.envs.planner_budget")


class PlannerBudget:
    """Hard enforcement of planner call ratio.

    Maintains a sliding window of the last N calls and blocks planner
    invocation if the planner ratio exceeds the configured limit.

    PRD requirement: SOTA model calls < 5%, cost < 10%.
    """

    def __init__(self, window_size: int = 1000, max_planner_ratio: float = 0.05,
                 budget_file: str | None = None):
        self._window_size = window_size
        self._max_ratio = max_planner_ratio
        self._calls: deque[str] = deque(maxlen=window_size)
        self._lock = threading.Lock()
        self._budget_file = budget_file

    def record_call(self, role: str) -> None:
        with self._lock:
            self._calls.append(role)
        self._maybe_persist()

    def may_use_planner(self) -> bool:
        with self._lock:
            if not self._calls:
                return True
            total = len(self._calls)
            planner_count = sum(1 for c in self._calls if c == "planner")
            ratio = planner_count / total if total > 0 else 0.0
            if ratio >= self._max_ratio:
                logger.warning(
                    "Planner budget exceeded: ratio=%.1f%% (max=%.1f%%)",
                    ratio * 100, self._max_ratio * 100,
                )
                return False
            return True

    @property
    def planner_ratio(self) -> float:
        with self._lock:
            total = len(self._calls)
            if total == 0:
                return 0.0
            planner_count = sum(1 for c in self._calls if c == "planner")
            return planner_count / total

    def stats(self) -> dict[str, Any]:
        return {
            "window_size": self._window_size,
            "max_planner_ratio": self._max_ratio,
            "current_ratio": round(self.planner_ratio, 4),
            "total_calls": len(self._calls),
            "planner_calls": sum(1 for c in self._calls if c == "planner"),
            "executor_calls": sum(1 for c in self._calls if c == "executor"),
        }

    def _maybe_persist(self) -> None:
        if not self._budget_file:
            return
        try:
            import json
            from pathlib import Path
            data = {"calls": list(self._calls), "max_ratio": self._max_ratio,
                    "window_size": self._window_size}
            Path(self._budget_file).write_text(json.dumps(data))
        except Exception as e:
            logger.debug("Failed to persist planner budget: %s", e)

    @classmethod
    def load(cls, path: str, **overrides: Any) -> "PlannerBudget":
        try:
            import json
            from pathlib import Path
            data = json.loads(Path(path).read_text())
            budget = cls(
                window_size=data.get("window_size", 1000),
                max_planner_ratio=data.get("max_ratio", 0.05),
                budget_file=path,
            )
            calls = data.get("calls", [])
            budget._calls = deque(calls, maxlen=budget._window_size)
            return budget
        except Exception:
            return cls(**overrides)
