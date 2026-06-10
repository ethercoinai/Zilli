import random
from collections import defaultdict
from typing import Dict

import numpy as np


class DynamicSOTAScheduler:
    def __init__(self, monthly_budget_usd: float = 500.0,
                 cost_per_call: Dict[str, float] = None):
        self.cost_per_call = cost_per_call or {"default": 0.04}
        self.monthly_budget = monthly_budget_usd
        self.remaining_budget = monthly_budget_usd
        self.total_calls = 0

        self.task_stats = defaultdict(lambda: {
            "failure_rate": 0.5,
            "samples": 0,
            "threshold": 0.7,
            "success_with_sota": 0.0,
            "success_without_sota": 0.0,
        })

        self.threshold_arms = [0.5, 0.6, 0.7, 0.8, 0.9]
        self.beta_params = {t: {"alpha": 1.0, "beta": 1.0}
                            for t in self.threshold_arms}

        min_cost = min(self.cost_per_call.values())
        self.hourly_quota = (monthly_budget_usd / 30 / 24) / min_cost if min_cost > 0 else 100
        self.calls_this_hour = 0

    def _executor_confidence(self, executor_state: Dict) -> float:
        return executor_state.get("max_prob", 0.5)

    def _performance_gap(self, task_type: str) -> float:
        stats = self.task_stats[task_type]
        return stats["success_with_sota"] - stats["success_without_sota"]

    def _sample_threshold(self, task_type: str) -> float:
        samples = []
        for t in self.threshold_arms:
            a = self.beta_params[t]["alpha"]
            b = self.beta_params[t]["beta"]
            samples.append(np.random.beta(a, b))
        return self.threshold_arms[int(np.argmax(samples))]

    def _update_bandit(self, used_threshold: float, reward: float):
        self.beta_params[used_threshold]["alpha"] += reward
        self.beta_params[used_threshold]["beta"] += (1 - reward)

    def should_call_sota(self, task_type: str, executor_state: Dict) -> bool:
        difficulty = self.task_stats[task_type]["failure_rate"]
        gap = self._performance_gap(task_type)
        conf = self._executor_confidence(executor_state)
        threshold = self._sample_threshold(task_type)

        if self.remaining_budget < 0.1 * self.monthly_budget:
            return difficulty > 0.8

        if self.calls_this_hour > 1.2 * self.hourly_quota:
            threshold = 0.9

        if difficulty > 0.7 and conf < 0.7:
            return True
        if gap > 0.2 and conf < 0.8:
            return True
        if gap < 0.05 and conf > 0.9:
            return False
        if random.random() < 0.05:
            return True

        return conf < threshold

    def record_call(self, model_name: str, task_type: str,
                    actual_success: bool):
        cost = self.cost_per_call.get(model_name, self.cost_per_call["default"])
        self.remaining_budget -= cost
        self.total_calls += 1
        self.calls_this_hour += 1

        stats = self.task_stats[task_type]
        n = stats["samples"] + 1
        stats["samples"] = n

        if actual_success:
            stats["success_with_sota"] = (
                stats["success_with_sota"] * (n - 1) + 1
            ) / n
            stats["failure_rate"] = stats["failure_rate"] * (n - 1) / n
        else:
            stats["success_with_sota"] = (
                stats["success_with_sota"] * (n - 1)
            ) / n
            stats["failure_rate"] = (
                stats["failure_rate"] * (n - 1) + 1
            ) / n

    def record_without_sota(self, task_type: str, actual_success: bool):
        stats = self.task_stats[task_type]
        n = stats["samples"] + 1
        stats["samples"] = n
        if actual_success:
            stats["success_without_sota"] = (
                stats["success_without_sota"] * (n - 1) + 1
            ) / n
        else:
            stats["success_without_sota"] = (
                stats["success_without_sota"] * (n - 1)
            ) / n

    def reset_hourly_counter(self):
        self.calls_this_hour = 0

    def stats(self) -> Dict:
        return {
            "remaining_budget": self.remaining_budget,
            "total_calls": self.total_calls,
            "calls_this_hour": self.calls_this_hour,
            "hourly_quota": self.hourly_quota,
            "task_types": dict(self.task_stats),
        }


__all__ = ["DynamicSOTAScheduler"]
