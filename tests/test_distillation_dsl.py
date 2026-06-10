import random

import pytest

from zilli.distillation.dsl import (
    ABTestGroup,
    ExperimentParams,
    ExperimentResult,
    compare,
    export_results,
    run_experiment,
)
from zilli.training.distillation import DistillationCycle, DistillationSample


def _make_sample(exec_reward: float = 0.5, plan_reward: float = 0.8) -> DistillationSample:
    return DistillationSample(
        executor_action={"tool": "write", "key": "x", "value": "1"},
        planner_action={"tool": "write", "key": "x", "value": "1"},
        executor_log_prob=random.uniform(-2.0, -0.5),
        planner_log_prob=random.uniform(-2.0, -0.5),
        executor_reward=exec_reward,
        planner_reward=plan_reward,
        state_embedding=[random.random() for _ in range(4)],
        executor_embedding=[random.random() for _ in range(4)],
        planner_embedding=[random.random() for _ in range(4)],
    )


def _make_samples(n: int = 50) -> list:
    return [_make_sample() for _ in range(n)]


class TestExperimentParams:
    def test_label_default(self):
        p = ExperimentParams(name="baseline")
        assert "baseline" in p.label()

    def test_label_with_tags(self):
        p = ExperimentParams(name="test", tags={"lr": "0.01", "opt": "adam"})
        label = p.label()
        assert "test" in label
        assert "lr=0.01" in label
        assert "opt=adam" in label

    def test_to_scheduler_kwargs(self):
        p = ExperimentParams(name="x", lambda_bc=0.5, lora_threshold=500)
        kw = p.to_scheduler_kwargs()
        assert kw["lambda_bc"] == 0.5
        assert kw["lora_threshold"] == 500
        assert "lora_callback" not in kw

    def test_to_scheduler_kwargs_excludes_callbacks(self):
        p = ExperimentParams(name="x")
        kw = p.to_scheduler_kwargs()
        assert "lora_callback" not in kw
        assert "full_sft_callback" not in kw


class TestRunExperiment:
    def test_run_experiment_returns_result(self):
        samples = _make_samples(20)
        p = ExperimentParams(name="test_run", lora_threshold=100)
        result = run_experiment(p, samples)
        assert result.params.name == "test_run"
        assert result.total_samples == 20
        assert isinstance(result.avg_loss, float)
        assert result.avg_kl >= 0
        assert result.wall_time_sec > 0

    def test_run_experiment_with_empty_samples(self):
        p = ExperimentParams(name="empty")
        result = run_experiment(p, [])
        assert result.total_samples == 0
        assert result.avg_loss == 0.0

    def test_run_experiment_tracks_lora(self):
        called = False
        def cb(samples):
            nonlocal called
            called = True
            return {"status": "ok"}
        samples = _make_samples(2000)
        p = ExperimentParams(name="lora_test", lora_threshold=1500)
        result = run_experiment(p, samples, lora_callback=cb)
        assert result.lora_count >= 0

    def test_run_experiment_with_callback_error(self):
        def failing_cb(samples):
            raise RuntimeError("callback failed")
        samples = _make_samples(50)
        p = ExperimentParams(name="fail_cb")
        result = run_experiment(p, samples, lora_callback=failing_cb)
        assert result.total_samples == 50


class TestABTestGroup:
    def test_add_variant(self):
        g = ABTestGroup(name="test")
        g.add(ExperimentParams(name="A", lambda_bc=1.0))
        g.add(ExperimentParams(name="B", lambda_bc=0.5))
        assert len(g.variants) == 2

    def test_chainable_add(self):
        g = ABTestGroup(name="chain")
        g.add(ExperimentParams(name="A")).add(ExperimentParams(name="B"))
        assert len(g.variants) == 2


class TestCompare:
    def test_compare_two_results(self):
        def _fake_cycle(loss: float) -> DistillationCycle:
            c = DistillationCycle(cycle_id=1, start_time=0, end_time=1, samples=10)
            c.total_loss = loss
            c.kl_divergence = 0.1
            c.avg_executor_reward = 0.5
            c.avg_planner_reward = 0.7
            return c

        r1 = ExperimentResult(
            params=ExperimentParams(name="A"),
            cycles=[_fake_cycle(1.0)],
            total_samples=10, avg_loss=1.0, avg_kl=0.1,
            avg_exec_reward=0.5, avg_plan_reward=0.7,
            wall_time_sec=1.0, lora_count=0,
        )
        r2 = ExperimentResult(
            params=ExperimentParams(name="B"),
            cycles=[_fake_cycle(0.5)],
            total_samples=10, avg_loss=0.5, avg_kl=0.05,
            avg_exec_reward=0.8, avg_plan_reward=0.9,
            wall_time_sec=1.5, lora_count=0,
        )
        comparison = compare([r1, r2])
        assert comparison["baseline"] == "A"
        assert comparison["best"] == "B"
        assert len(comparison["comparisons"]) == 1
        assert comparison["comparisons"][0]["loss_delta"] == pytest.approx(-0.5)

    def test_compare_requires_two(self):
        r = ExperimentResult(
            params=ExperimentParams(name="only"),
            cycles=[], total_samples=0, avg_loss=0, avg_kl=0,
            avg_exec_reward=0, avg_plan_reward=0,
            wall_time_sec=0, lora_count=0,
        )
        result = compare([r])
        assert "error" in result


class TestExportResults:
    def test_export_results_creates_file(self, tmp_path):
        from zilli.distillation.dsl import ABIteration
        r = ExperimentResult(
            params=ExperimentParams(name="A"),
            cycles=[], total_samples=10, avg_loss=0.5, avg_kl=0.1,
            avg_exec_reward=0.6, avg_plan_reward=0.7,
            wall_time_sec=2.0, lora_count=0,
        )
        iteration = ABIteration(
            group=ABTestGroup(name="test"),
            results=[r],
            best=r,
        )
        path = str(tmp_path / "ab_result.json")
        export_results(iteration, path)
        import json
        with open(path) as f:
            data = json.load(f)
        assert data["group"] == "test"
        assert len(data["variants"]) == 1


class TestExperimentResult:
    def test_summary_contains_keys(self):
        r = ExperimentResult(
            params=ExperimentParams(name="test"),
            cycles=[], total_samples=100, avg_loss=0.5, avg_kl=0.1,
            avg_exec_reward=0.6, avg_plan_reward=0.7,
            wall_time_sec=3.0, lora_count=1,
        )
        s = r.summary()
        assert s["experiment"] == "test"
        assert s["total_samples"] == 100
        assert s["avg_loss"] == 0.5
        assert s["lora_events"] == 1
