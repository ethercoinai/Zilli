from __future__ import annotations

import os
import tempfile

from zilli.routing.frequency_controller import PlannerFrequencyController


def _make_ctrl(max_planner_ratio=0.05, window=3600):
    f = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    f.close()
    return (
        PlannerFrequencyController(
            max_planner_ratio=max_planner_ratio,
            window_seconds=window,
            persist_path=f.name,
        ),
        f.name,
    )


def test_init_resets_on_expired_window():
    ctrl, path = _make_ctrl(max_planner_ratio=0.05)
    try:
        assert ctrl.current_ratio() == 0.0
        assert ctrl.stats()["total_calls"] == 0
    finally:
        os.unlink(path)


def test_records_and_blocks():
    ctrl, path = _make_ctrl(max_planner_ratio=0.1)
    try:
        for _ in range(9):
            ctrl.record_executor_call()
        assert ctrl.record_planner_call() is True
        assert ctrl.record_planner_call() is False
        assert ctrl.stats()["planner_blocked"] >= 1
    finally:
        os.unlink(path)


def test_persists_and_loads():
    ctrl, path = _make_ctrl(max_planner_ratio=0.5)
    try:
        ctrl.record_executor_call()
        ctrl.record_planner_call()

        ctrl2 = PlannerFrequencyController(
            max_planner_ratio=0.5,
            window_seconds=3600,
            persist_path=path,
        )
        assert ctrl2.stats()["total_calls"] == 2
        assert ctrl2.stats()["planner_calls"] == 1
    finally:
        os.unlink(path)


def test_records_executor_only():
    ctrl, path = _make_ctrl(max_planner_ratio=0.5)
    try:
        for _ in range(100):
            ctrl.record_executor_call()
        assert ctrl.current_ratio() == 0.0
        assert ctrl.stats()["executor_calls"] == 100
    finally:
        os.unlink(path)
