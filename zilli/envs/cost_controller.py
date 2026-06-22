from __future__ import annotations

import fcntl
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

from zilli.adaptive.sota_scheduler import DynamicSOTAScheduler
from zilli.models.registry import ModelRegistry

if TYPE_CHECKING:
    from zilli.configs import ZilliConfig

logger = logging.getLogger("zilli.cost_controller")

_BUDGET_FILE = Path.home() / ".zilli_budget.json"
_DEFAULT_BUDGET = 500.0


@dataclass
class BudgetSnapshot:
    remaining_budget: float
    total_calls: int
    calls_this_hour: int
    hourly_quota: float
    emergency_mode: bool
    timestamp: float


class CostController:
    def __init__(
        self,
        budget_file: Optional[str] = None,
        monthly_budget: float = _DEFAULT_BUDGET,
        model_registry: Optional[ModelRegistry] = None,
        config: Optional["ZilliConfig"] = None,
    ):
        self._file = Path(budget_file) if budget_file else _BUDGET_FILE
        self.model_registry = model_registry or ModelRegistry()
        self._config = config

        budget = monthly_budget
        if config:
            profile = config.to_model_profile()
            budget = profile.monthly_budget_usd
        elif model_registry:
            budget = model_registry.profile.monthly_budget_usd

        self._scheduler = DynamicSOTAScheduler(
            monthly_budget_usd=budget,
            model_registry=model_registry,
            config=config,
        )
        self._dirty = False
        self._last_save_time = 0.0
        self._load()

    def _load(self):
        if self._file.exists():
            try:
                data = json.loads(self._file.read_text())
                self._scheduler.remaining_budget = data.get("remaining_budget", _DEFAULT_BUDGET)
                self._scheduler.total_calls = data.get("total_calls", 0)
                self._scheduler.hourly_quota = data.get("hourly_quota", self._scheduler.hourly_quota)
                logger.info("Budget loaded: %.2f remaining (%d calls)",
                            self._scheduler.remaining_budget, self._scheduler.total_calls)
            except (json.JSONDecodeError, KeyError) as e:
                logger.warning("Failed to load budget file, using defaults: %s", e)

    def _save(self):
        self._file.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "remaining_budget": self._scheduler.remaining_budget,
            "total_calls": self._scheduler.total_calls,
            "hourly_quota": self._scheduler.hourly_quota,
            "updated_at": time.time(),
        }
        try:
            with open(self._file, "w") as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                f.write(json.dumps(data, indent=2))
                f.flush()
                fcntl.flock(f, fcntl.LOCK_UN)
        except OSError as e:
            logger.warning("Failed to save budget file: %s", e)
        self._dirty = False

    def should_use_planner(self, task_type: str,
                           executor_state: Optional[Dict[str, Any]] = None) -> bool:
        return self._scheduler.should_call_sota(
            task_type, executor_state or {"max_prob": 0.5},
        )

    def _debounced_save(self):
        now = time.time()
        if now - self._last_save_time < 5.0:
            return
        self._last_save_time = now
        self._save()

    def record_planner_call(self, task_type: str, success: bool):
        self._scheduler.record_call("planner", task_type, success)
        self._dirty = True
        self._debounced_save()

    def record_executor_call(self, task_type: str, success: bool):
        self._scheduler.record_without_sota(task_type, success)
        self._dirty = True
        self._debounced_save()

    def snapshot(self) -> BudgetSnapshot:
        s = self._scheduler
        return BudgetSnapshot(
            remaining_budget=round(s.remaining_budget, 2),
            total_calls=s.total_calls,
            calls_this_hour=s.calls_this_hour,
            hourly_quota=round(s.hourly_quota, 2),
            emergency_mode=s.remaining_budget < 0.1 * s.monthly_budget,
            timestamp=time.time(),
        )

    def reset_monthly(self):
        self._scheduler.remaining_budget = self._scheduler.monthly_budget
        self._scheduler.total_calls = 0
        self._scheduler.calls_this_hour = 0
        self._save()

    def reset_hourly(self):
        self._scheduler.reset_hourly_counter()

    @property
    def scheduler(self) -> DynamicSOTAScheduler:
        return self._scheduler


__all__ = ["CostController", "BudgetSnapshot"]
