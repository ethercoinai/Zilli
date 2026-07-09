from __future__ import annotations

from zilli.loops.failure_analyzer import FailureRecord, WeaknessMiner


def test_ingest_and_cluster():
    miner = WeaknessMiner(min_cluster_size=2)
    traces = [
        {"task_id": "t1", "verifier_outcome": "timeout",
         "causal_status": "wrong_tool", "mechanism": "tool_registry"},
        {"task_id": "t2", "verifier_outcome": "timeout",
         "causal_status": "wrong_tool", "mechanism": "tool_registry"},
        {"task_id": "t3", "verifier_outcome": "crash",
         "causal_status": "missing_file", "mechanism": "filesystem"},
    ]
    clusters = miner.cluster_failures(traces)
    assert len(clusters) == 1  # only one cluster meets min_cluster_size=2
    assert clusters[0].pattern_label == "tool_registry: wrong_tool"
    assert clusters[0].count == 2


def test_failure_record_fields():
    record = FailureRecord(
        task_id="test-task",
        verifier_outcome="wrong_answer",
        causal_status="skipped_validation",
        mechanism="verifier",
        trace_excerpt="agent returned incorrect result",
    )
    assert record.task_id == "test-task"
    assert record.verifier_outcome == "wrong_answer"
    assert record.causal_status == "skipped_validation"
    assert record.mechanism == "verifier"


def test_min_cluster_size_filters():
    miner = WeaknessMiner(min_cluster_size=3)
    traces = [
        {"task_id": "t1", "verifier_outcome": "a", "causal_status": "x", "mechanism": "m"},
        {"task_id": "t2", "verifier_outcome": "a", "causal_status": "x", "mechanism": "m"},
    ]
    clusters = miner.cluster_failures(traces)
    assert len(clusters) == 0


def test_summary():
    miner = WeaknessMiner(min_cluster_size=1)
    miner.ingest([
        {"task_id": "t1", "verifier_outcome": "timeout",
         "causal_status": "wrong_tool", "mechanism": "tool"},
    ])
    summary = miner.summary()
    assert summary["total_records"] == 1
    assert summary["top_pattern"] is not None


def test_clear():
    miner = WeaknessMiner(min_cluster_size=1)
    miner.ingest([{"task_id": "t1", "verifier_outcome": "x", "causal_status": "y", "mechanism": "z"}])
    assert miner.total_records == 1
    miner.clear()
    assert miner.total_records == 0
