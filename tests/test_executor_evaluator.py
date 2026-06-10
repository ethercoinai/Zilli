from unittest.mock import AsyncMock, MagicMock

import pytest

from zilli.evaluation.executor_only_evaluator import EvalResult, ExecutorOnlyEvaluator


class TestEvalResult:
    def test_dataclass_defaults(self):
        r = EvalResult(task_name="test", success=True, cost_usd=0.0,
                       inference_tokens=100, latency_sec=1.0)
        assert r.task_name == "test"
        assert r.success is True

    def test_dataclass_asdict(self):
        from dataclasses import asdict
        r = EvalResult(task_name="t", success=True, cost_usd=0.01,
                       inference_tokens=50, latency_sec=0.5)
        d = asdict(r)
        assert d["task_name"] == "t"


class TestExecutorOnlyEvaluator:
    @pytest.fixture
    def mock_model(self):
        m = MagicMock()
        m.generate.return_value = ("response text", 100, {})
        return m

    @pytest.fixture
    def mock_sandbox(self):
        s = AsyncMock()
        s.reset = AsyncMock()
        s.execute = AsyncMock(return_value={"done": True, "output": "ok"})
        return s

    @pytest.fixture
    def tasks(self):
        return [
            {
                "name": "test_task",
                "prompt": "do something",
                "verification_fn": lambda s: s.get("done", False),
            }
        ]

    @pytest.mark.asyncio
    async def test_run_single_task_success(self, mock_model, mock_sandbox, tasks):
        evaluator = ExecutorOnlyEvaluator(mock_model, tasks, mock_sandbox)
        results = await evaluator.run_single_task(tasks[0], repeat=1)
        assert len(results) == 1
        assert results[0].success is True
        assert results[0].task_name == "test_task"
        assert results[0].inference_tokens == 100

    @pytest.mark.asyncio
    async def test_run_single_task_multiple_repeats(self, mock_model, mock_sandbox, tasks):
        evaluator = ExecutorOnlyEvaluator(mock_model, tasks, mock_sandbox)
        results = await evaluator.run_single_task(tasks[0], repeat=3)
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_run_single_task_exception_handling(self, mock_model, mock_sandbox, tasks):
        mock_model.generate.side_effect = Exception("model failure")
        evaluator = ExecutorOnlyEvaluator(mock_model, tasks, mock_sandbox)
        results = await evaluator.run_single_task(tasks[0], repeat=1)
        assert len(results) == 1
        assert results[0].success is False
        assert results[0].inference_tokens == 0

    @pytest.mark.asyncio
    async def test_run_single_task_cost(self, mock_model, mock_sandbox, tasks):
        evaluator = ExecutorOnlyEvaluator(mock_model, tasks, mock_sandbox,
                                          cost_per_1k_tokens=1.0)
        results = await evaluator.run_single_task(tasks[0], repeat=1)
        assert results[0].cost_usd == 0.1

    @pytest.mark.asyncio
    async def test_evaluate_report_structure(self, mock_model, mock_sandbox, tasks):
        evaluator = ExecutorOnlyEvaluator(mock_model, tasks, mock_sandbox)
        report = await evaluator.evaluate(duration_hours=0.001, repeat_per_task=1)
        assert "success_rate" in report
        assert "avg_cost_usd" in report
        assert "avg_latency_sec" in report
        assert "avg_tokens" in report
        assert "total_cost_usd" in report
        assert "total_runs" in report
        assert report["success_rate"] == 1.0
