from __future__ import annotations

import asyncio
from typing import Any

from zilli.loops.failure_analyzer import WeaknessMiner
from zilli.loops.harness_orchestrator import HarnessEdit, HarnessOrchestrator


def _dummy_surface() -> str:
    return "def process(input): return input"


async def _passing_task() -> dict[str, bool]:
    return {"test1": True, "test2": True}


async def _failing_task() -> dict[str, bool]:
    return {"test1": False, "test2": False}


def test_harness_edit_creation():
    edit = HarnessEdit(
        description="Fix context overflow",
        diff="--- a/ctx\n+++ b/ctx\n@@ -1,3 +1,3 @@",
        source_file="context_manager",
        target_pattern="context_overflow",
    )
    assert edit.description == "Fix context overflow"
    assert edit.accepted is False
    assert edit.rejection_reason == ""


def test_orchestrator_no_failures():
    miner = WeaknessMiner(min_cluster_size=1)
    orchestrator = HarnessOrchestrator(
        current_version="1.0.0",
        editable_surfaces={"ctx": _dummy_surface},
        miner=miner,
        held_in_tasks=[],
        held_out_tasks=[],
    )
    result = asyncio.run(orchestrator.run_cycle([]))
    assert result is None


def test_orchestrator_proposes_on_failures():
    miner = WeaknessMiner(min_cluster_size=1)
    traces = [
        {"task_id": "t1", "verifier_outcome": "timeout",
         "causal_status": "wrong_tool", "mechanism": "tool_registry"},
        {"task_id": "t2", "verifier_outcome": "timeout",
         "causal_status": "wrong_tool", "mechanism": "tool_registry"},
    ]

    orchestrator = HarnessOrchestrator(
        current_version="1.0.0",
        editable_surfaces={"tool_registry": _dummy_surface},
        miner=miner,
        held_in_tasks=[_passing_task],
        held_out_tasks=[_passing_task],
        improvement_threshold=0.0,
    )
    result = asyncio.run(orchestrator.run_cycle(traces))
    assert result is not None


def test_orchestrator_rejects_regression():
    miner = WeaknessMiner(min_cluster_size=1)
    traces = [
        {"task_id": "t1", "verifier_outcome": "timeout",
         "causal_status": "wrong_tool", "mechanism": "tool_registry"},
    ]

    orchestrator = HarnessOrchestrator(
        current_version="1.0.0",
        editable_surfaces={"tool_registry": _dummy_surface},
        miner=miner,
        held_in_tasks=[_passing_task],      # held-in passes
        held_out_tasks=[_failing_task],     # held-out fails → regression
        improvement_threshold=0.0,
        max_edits_per_round=5,
    )
    result = asyncio.run(orchestrator.run_cycle(traces))
    assert result is not None
    assert not result.accepted


def test_stats():
    miner = WeaknessMiner(min_cluster_size=1)
    orchestrator = HarnessOrchestrator(
        current_version="1.0.0",
        editable_surfaces={},
        miner=miner,
        held_in_tasks=[],
        held_out_tasks=[],
    )
    stats = orchestrator.stats()
    assert stats["version"] == "1.0.0"
    assert stats["cycles"] == 0
