from __future__ import annotations

import asyncio

from zilli.core.runner import TaskRunner, TaskStep


class TestTaskRunner:
    def test_run_single_step(self):
        runner = TaskRunner()
        steps = [TaskStep(name="hello", fn=lambda: "world")]
        results = asyncio.run(runner.run(steps))
        assert len(results) == 1
        assert results[0].success
        assert results[0].result == "world"

    def test_run_sequential(self):
        runner = TaskRunner()
        order: list[str] = []

        def a():
            order.append("a")

        def b():
            order.append("b")

        def c():
            order.append("c")

        steps = [
            TaskStep(name="a", fn=a),
            TaskStep(name="b", fn=b, depends_on=["a"]),
            TaskStep(name="c", fn=c, depends_on=["b"]),
        ]
        results = asyncio.run(runner.run(steps))
        assert all(r.success for r in results)
        assert order == ["a", "b", "c"]

    def test_run_parallel(self):
        runner = TaskRunner(max_concurrency=4)
        steps = [
            TaskStep(name="a", fn=lambda: "a"),
            TaskStep(name="b", fn=lambda: "b"),
            TaskStep(name="c", fn=lambda: "c"),
        ]
        results = asyncio.run(runner.run(steps))
        assert all(r.success for r in results)
        assert {r.result for r in results} == {"a", "b", "c"}

    def test_retry_on_failure(self):
        runner = TaskRunner()
        call_count = [0]

        def flaky():
            call_count[0] += 1
            if call_count[0] < 2:
                raise RuntimeError("fail")
            return "ok"

        steps = [TaskStep(name="f", fn=flaky, retries=2)]
        results = asyncio.run(runner.run(steps))
        assert results[0].success
        assert results[0].result == "ok"
        assert call_count[0] == 2

    def test_all_fail(self):
        runner = TaskRunner()

        def fail():
            raise RuntimeError("always fails")

        steps = [TaskStep(name="f", fn=fail, retries=1, timeout_s=1)]
        results = asyncio.run(runner.run(steps))
        assert not results[0].success
        assert "always fails" in results[0].error

    def test_dag_dependency_shared_state(self):
        runner = TaskRunner()
        results: dict[str, str] = {}

        def parse():
            results["parse"] = "parsed"

        def validate():
            results["validate"] = "validated"

        def generate():
            results["generate"] = "generated"

        steps = [
            TaskStep(name="parse", fn=parse),
            TaskStep(name="validate", fn=validate, depends_on=["parse"]),
            TaskStep(name="generate", fn=generate, depends_on=["validate"]),
        ]
        r = asyncio.run(runner.run(steps))
        assert all(x.success for x in r)
        assert list(results.keys()) == ["parse", "validate", "generate"]
