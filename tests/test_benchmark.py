import json
import time
from pathlib import Path

import pytest

from zilli.evaluation.distillation_benchmark import (
    BenchmarkEntry,
    BenchmarkTracker,
    run_benchmarked_distillation,
)
from zilli.training.distillation import DistillationCycle, DistillationSample, DistillationScheduler


def _sample(exec_reward=0.5, plan_reward=0.8):
    return DistillationSample(
        executor_action={"tool": "write"},
        planner_action={"tool": "write"},
        executor_log_prob=-1.0,
        planner_log_prob=-1.5,
        executor_reward=exec_reward,
        planner_reward=plan_reward,
        executor_embedding=[0.1, 0.2],
        planner_embedding=[0.3, 0.4],
    )


class TestBenchmarkEntry:
    def test_entry_creation(self):
        entry = BenchmarkEntry(
            timestamp=100.0,
            model_name="executor",
            phase="before_distill",
            loss=0.5,
            kl=0.1,
            exec_reward=0.6,
            plan_reward=0.7,
            sample_count=50,
            wall_time_sec=2.0,
        )
        assert entry.model_name == "executor"
        assert entry.phase == "before_distill"

    def test_to_dict(self):
        entry = BenchmarkEntry(
            timestamp=100.0,
            model_name="test",
            phase="after_distill",
            loss=0.5, kl=0.1,
            exec_reward=0.6, plan_reward=0.7,
            sample_count=10, wall_time_sec=1.0,
        )
        d = entry.to_dict()
        assert d["loss"] == 0.5
        assert d["phase"] == "after_distill"
        assert d["sample_count"] == 10


class TestBenchmarkTracker:
    def test_record_before(self, tmp_path):
        tracker = BenchmarkTracker(log_dir=str(tmp_path))
        scheduler = DistillationScheduler(log_dir=str(tmp_path))
        scheduler.add_batch([_sample() for _ in range(10)])
        entry = tracker.record_before(scheduler)
        assert entry.phase == "before_distill"
        assert entry.sample_count == 10

    def test_record_before_logs_to_file(self, tmp_path):
        tracker = BenchmarkTracker(log_dir=str(tmp_path))
        scheduler = DistillationScheduler(log_dir=str(tmp_path))
        scheduler.add_batch([_sample() for _ in range(5)])
        tracker.record_before(scheduler)
        log_file = Path(tmp_path) / "benchmark_entries.jsonl"
        assert log_file.exists()
        lines = log_file.read_text().strip().split("\n")
        assert len(lines) == 1

    def test_record_after(self, tmp_path):
        tracker = BenchmarkTracker(log_dir=str(tmp_path))
        cycle = DistillationCycle(
            cycle_id=1, start_time=time.time(), end_time=time.time() + 1,
            samples=50, total_loss=0.5, bc_loss=0.3, rl_loss=0.1, reg_loss=0.1,
            kl_divergence=0.05, avg_executor_reward=0.7, avg_planner_reward=0.8,
        )
        entry = tracker.record_after(cycle, model_name="executor")
        assert entry.phase == "after_distill"
        assert entry.loss == 0.5

    def test_record_ab_result(self, tmp_path):
        tracker = BenchmarkTracker(log_dir=str(tmp_path))
        entry = tracker.record_ab_result(
            variant_name="exp_A", loss=0.4, kl=0.05,
            sample_count=100, wall_time_sec=5.0,
            metadata={"lr": 1e-4},
        )
        assert entry.phase == "ab_test"
        assert entry.model_name == "exp_A"
        assert entry.metadata.get("lr") == 1e-4

    def test_to_arena_match(self, tmp_path):
        tracker = BenchmarkTracker(log_dir=str(tmp_path))
        before = BenchmarkEntry(timestamp=1, model_name="pre", phase="before_distill", loss=0.8, kl=0.2, exec_reward=0, plan_reward=0, sample_count=10, wall_time_sec=0)
        after = BenchmarkEntry(timestamp=2, model_name="post", phase="after_distill", loss=0.4, kl=0.1, exec_reward=1, plan_reward=1, sample_count=10, wall_time_sec=2)
        match = tracker.to_arena_match("old_model", "new_model", before, after)
        assert match["champion"] == "old_model"
        assert match["challenger"] == "new_model"
        assert match["loss_delta"] == pytest.approx(-0.4)
        assert match["kl_delta"] == pytest.approx(-0.1)

    def test_log_arena_match_creates_file(self, tmp_path):
        tracker = BenchmarkTracker(log_dir=str(tmp_path))
        match = {"match_id": "test_1", "type": "distillation_benchmark"}
        tracker.log_arena_match(match)
        log_file = Path(tmp_path) / "distill_benchmarks.jsonl"
        assert log_file.exists()
        with open(log_file) as f:
            assert json.load(f)["match_id"] == "test_1"

    def test_get_recent(self, tmp_path):
        tracker = BenchmarkTracker(log_dir=str(tmp_path))
        scheduler = DistillationScheduler(log_dir=str(tmp_path))
        scheduler.add_batch([_sample() for _ in range(3)])
        tracker.record_before(scheduler)
        recent = tracker.get_recent()
        assert len(recent) == 1
        assert recent[0]["phase"] == "before_distill"


class TestRunBenchmarkedDistillation:
    def test_benchmarked_distillation_no_arena(self, tmp_path):
        scheduler = DistillationScheduler(log_dir=str(tmp_path))
        scheduler.add_batch([_sample() for _ in range(20)])
        tracker = BenchmarkTracker(log_dir=str(tmp_path))
        cycle = run_benchmarked_distillation(scheduler, tracker)
        assert cycle is not None

    def test_benchmarked_distillation_creates_logs(self, tmp_path):
        scheduler = DistillationScheduler(log_dir=str(tmp_path))
        scheduler.add_batch([_sample() for _ in range(20)])
        tracker = BenchmarkTracker(log_dir=str(tmp_path))
        run_benchmarked_distillation(scheduler, tracker)
        bench_file = Path(tmp_path) / "distill_benchmarks.jsonl"
        assert bench_file.exists()

    def test_benchmarked_distillation_with_arena(self, tmp_path):
        from zilli.training.champion_challenger import ArenaStatus, ChampionChallenger
        scheduler = DistillationScheduler(log_dir=str(tmp_path))
        scheduler.add_batch([_sample() for _ in range(20)])
        tracker = BenchmarkTracker(log_dir=str(tmp_path))
        arena = ChampionChallenger(log_dir=str(tmp_path))
        arena.register_model("pre_distill", "v1", ArenaStatus.CHAMPION)
        arena.register_model("post_distill", "v2")
        cycle = run_benchmarked_distillation(
            scheduler, tracker, arena=arena,
            champion_name="pre_distill", challenger_name="post_distill",
        )
        assert cycle is not None
