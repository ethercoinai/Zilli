from __future__ import annotations

import pytest

from zilli.envs.planner_budget import PlannerBudget


def test_allows_executor_calls():
    budget = PlannerBudget(window_size=100, max_planner_ratio=0.05)
    for _ in range(100):
        budget.record_call("executor")
    assert budget.planner_ratio == 0.0
    assert budget.may_use_planner() is True


def test_blocks_planner_when_ratio_exceeded():
    budget = PlannerBudget(window_size=20, max_planner_ratio=0.1)
    for _ in range(18):
        budget.record_call("executor")
    budget.record_call("planner")
    budget.record_call("planner")
    assert budget.planner_ratio == pytest.approx(0.1, abs=0.01)
    assert budget.may_use_planner() is False


def test_allows_planner_below_ratio():
    budget = PlannerBudget(window_size=100, max_planner_ratio=0.2)
    for _ in range(10):
        budget.record_call("executor")
    budget.record_call("planner")
    assert budget.planner_ratio == pytest.approx(1 / 11, abs=0.01)
    assert budget.may_use_planner() is True


def test_rolling_window_evicts_old_calls():
    budget = PlannerBudget(window_size=10, max_planner_ratio=0.2)
    for _ in range(10):
        budget.record_call("planner")
    assert budget.planner_ratio == 1.0
    for _ in range(10):
        budget.record_call("executor")
    assert budget.planner_ratio <= 0.5


def test_stats():
    budget = PlannerBudget(window_size=100, max_planner_ratio=0.05)
    budget.record_call("executor")
    budget.record_call("planner")
    stats = budget.stats()
    assert stats["total_calls"] == 2
    assert stats["planner_calls"] == 1
    assert stats["executor_calls"] == 1
    assert stats["max_planner_ratio"] == 0.05
