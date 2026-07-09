from __future__ import annotations

from zilli.models.profiler import ModelProfiler


def test_initial_elo():
    profiler = ModelProfiler()
    assert profiler.get_elo("unknown-model") == 1200.0


def test_comparison_updates_elo():
    profiler = ModelProfiler()
    profiler.record_comparison("model-a", "model-b")
    elo_a = profiler.get_elo("model-a")
    elo_b = profiler.get_elo("model-b")
    assert elo_a > 1200.0
    assert elo_b < 1200.0


def test_leaderboard_is_sorted():
    profiler = ModelProfiler()
    profiler.record_comparison("winner", "loser")
    board = profiler.leaderboard()
    assert len(board) == 2
    assert board[0]["name"] == "winner"
    assert board[0]["elo"] > board[1]["elo"]


def test_capability_radar_data():
    profiler = ModelProfiler()
    profiler.record_task_result("model-a", "reasoning", 0.9)
    radar = profiler.get_radar_data("model-a")
    assert set(radar.keys()) == {
        "reasoning", "code", "math", "creativity", "instruction", "safety"
    }
    assert radar["reasoning"] > 0.5


def test_record_task_result_updates_capability():
    profiler = ModelProfiler()
    profiler.record_task_result("model-x", "reasoning", 1.0)
    profiler.record_task_result("model-x", "code", 0.0)
    caps = profiler.get_capabilities("model-x")
    assert caps.reasoning > 0.6
    assert caps.code <= 0.35


def test_stats():
    profiler = ModelProfiler()
    profiler.record_comparison("a", "b")
    stats = profiler.stats()
    assert stats["models_tracked"] == 2
    assert stats["total_comparisons"] >= 2
