from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("zilli.routing.frequency_controller")


@dataclass
class PlannerCallRecord:
    total_calls: int = 0
    planner_calls: int = 0
    executor_calls: int = 0
    window_start: float = 0.0
    planner_blocked: int = 0


class PlannerFrequencyController:
    def __init__(
        self,
        max_planner_ratio: float = 0.05,
        window_seconds: int = 3600,
        persist_path: str = "",
    ):
        self.max_ratio = max_planner_ratio
        self.window_seconds = window_seconds
        self._path = Path(persist_path) if persist_path else Path.home() / ".zilli_planner_freq.json"
        self._record = PlannerCallRecord()
        self._load()

    def _load(self):
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text())
                self._record = PlannerCallRecord(**data)
                if time.time() - self._record.window_start > self.window_seconds:
                    self._reset()
        except Exception as e:
            logger.debug("Failed to load planner frequency: %s", e)
            self._reset()

    def _save(self):
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._path.write_text(json.dumps({
                "total_calls": self._record.total_calls,
                "planner_calls": self._record.planner_calls,
                "executor_calls": self._record.executor_calls,
                "window_start": self._record.window_start,
                "planner_blocked": self._record.planner_blocked,
            }))
        except Exception as e:
            logger.debug("Failed to save planner frequency: %s", e)

    def _reset(self):
        self._record = PlannerCallRecord(window_start=time.time())
        self._save()

    def record_planner_call(self) -> bool:
        if time.time() - self._record.window_start > self.window_seconds:
            self._reset()
        self._record.total_calls += 1
        self._record.planner_calls += 1
        current_ratio = self._record.planner_calls / max(self._record.total_calls, 1)
        if current_ratio > self.max_ratio:
            self._record.planner_calls -= 1
            self._record.planner_blocked += 1
            self._save()
            return False
        self._save()
        return True

    def record_executor_call(self) -> None:
        if time.time() - self._record.window_start > self.window_seconds:
            self._reset()
        self._record.total_calls += 1
        self._record.executor_calls += 1
        self._save()

    def current_ratio(self) -> float:
        if self._record.total_calls == 0:
            return 0.0
        return self._record.planner_calls / self._record.total_calls

    def stats(self) -> dict:
        return {
            "total_calls": self._record.total_calls,
            "planner_calls": self._record.planner_calls,
            "executor_calls": self._record.executor_calls,
            "planner_ratio": round(self.current_ratio(), 4),
            "max_ratio": self.max_ratio,
            "planner_blocked": self._record.planner_blocked,
            "window_seconds": self.window_seconds,
            "window_remaining": max(0, self.window_seconds - (time.time() - self._record.window_start)),
        }
