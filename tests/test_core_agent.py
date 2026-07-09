from __future__ import annotations

from zilli.core.agent import Agent, AgentResult, _extract_python, _fallback_for_task


class TestAgent:
    def test_extract_python(self):
        code = _extract_python("Some text\n```python\nprint('hello')\n```\nmore")
        assert code == "print('hello')"

    def test_extract_python_no_fence(self):
        code = _extract_python("print('hello')")
        assert code == "print('hello')"

    def test_fallback_fib(self):
        code = _fallback_for_task("write fibonacci")
        assert "a, b = 0, 1" in code
        assert "for i in range" in code

    def test_fallback_hello(self):
        code = _fallback_for_task("say hello")
        assert "Hello from Zilli" in code

    def test_run_simple_task(self):
        import asyncio
        agent = Agent()
        result = asyncio.run(agent.run("print hello world"))
        assert result.success
        assert "Hello" in result.output or "hello" in result.output.lower()

    def test_run_fib(self):
        import asyncio
        agent = Agent()
        result = asyncio.run(agent.run("fibonacci"))
        assert result.success
        assert "0 1 1 2 3" in result.output

    def test_run_failure_then_retry(self):
        import asyncio

        class FailOnceAgent(Agent):
            def __init__(self):
                super().__init__(max_retries=2)
                self._call_count = 0

            async def _execute_code(self, code: str) -> tuple[bool, str, str]:
                self._call_count += 1
                if self._call_count == 1:
                    return False, "", "Intentional failure"
                return await super()._execute_code(code)

        a = FailOnceAgent()
        result = asyncio.run(a.run("print hello"))
        assert result.success

    def test_run_all_fail(self):
        import asyncio

        class AlwaysFailAgent(Agent):
            async def _execute_code(self, code: str) -> tuple[bool, str, str]:
                return False, "", "Always fails"

        agent = AlwaysFailAgent(max_retries=1)
        result = asyncio.run(agent.run("anything"))
        assert not result.success
        assert "Always fails" in (result.error or "")

    def test_result_dataclass(self):
        r = AgentResult(success=True, output="ok", code_used="print('ok')")
        assert r.success
        assert r.output == "ok"
