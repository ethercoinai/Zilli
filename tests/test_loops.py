from unittest.mock import AsyncMock, MagicMock

import pytest

from zilli.loops.base import VerificationResult
from zilli.loops.memory import CycleMemory, MemoryEntry
from zilli.loops.runner import LoopRunner
from zilli.loops.trigger import DynamicIntervalTrigger, EventTrigger, FixedIntervalTrigger
from zilli.loops.verification import (
    CompositeVerifier,
    ExternalModelVerifier,
    PredicateVerifier,
    TestSuiteVerifier,
)


class TestFixedIntervalTrigger:
    @pytest.mark.asyncio
    async def test_positive_interval_required(self):
        with pytest.raises(ValueError, match="positive"):
            FixedIntervalTrigger(0)

    @pytest.mark.asyncio
    async def test_wait(self):
        t = FixedIntervalTrigger(0.01)
        result = await t.wait()
        assert result is True


class TestEventTrigger:
    @pytest.mark.asyncio
    async def test_event_true(self):
        t = EventTrigger(lambda: True)
        assert await t.wait() is True

    @pytest.mark.asyncio
    async def test_event_timeout(self):
        t = EventTrigger(lambda: False, poll_interval=0.01, timeout=0.05)
        assert await t.wait() is False


class TestDynamicIntervalTrigger:
    @pytest.mark.asyncio
    async def test_default_interval(self):
        t = DynamicIntervalTrigger(min_interval=0.01)
        result = await t.wait()
        assert result is True

    @pytest.mark.asyncio
    async def test_interval_fn(self):
        t = DynamicIntervalTrigger(min_interval=0.01, interval_fn=lambda s: 0.05 if s.get("busy") else 0.01)
        t.update_state({"busy": True})
        result = await t.wait()
        assert result is True


class TestPredicateVerifier:
    @pytest.mark.asyncio
    async def test_pass(self):
        v = PredicateVerifier(lambda inp, out: out == "ok")
        r = await v.verify("do", "ok")
        assert r.passed is True

    @pytest.mark.asyncio
    async def test_fail(self):
        v = PredicateVerifier(lambda inp, out: out == "ok")
        r = await v.verify("do", "bad")
        assert r.passed is False

    @pytest.mark.asyncio
    async def test_exception(self):
        v = PredicateVerifier(lambda inp, out: (_ for _ in ()).throw(ValueError("bad")))
        r = await v.verify("", "")
        assert r.passed is False
        assert "bad" in r.evidence


class TestCompositeVerifier:
    @pytest.mark.asyncio
    async def test_all_pass(self):
        v = CompositeVerifier([
            PredicateVerifier(lambda i, o: True),
            PredicateVerifier(lambda i, o: True),
        ])
        r = await v.verify("", "")
        assert r.passed is True

    @pytest.mark.asyncio
    async def test_any_pass_when_require_all_false(self):
        v = CompositeVerifier([
            PredicateVerifier(lambda i, o: False),
            PredicateVerifier(lambda i, o: True),
        ], require_all=False)
        r = await v.verify("", "")
        assert r.passed is True

    @pytest.mark.asyncio
    async def test_one_fails_require_all(self):
        v = CompositeVerifier([
            PredicateVerifier(lambda i, o: True),
            PredicateVerifier(lambda i, o: False),
        ])
        r = await v.verify("", "")
        assert r.passed is False


class TestTestSuiteVerifier:
    @pytest.mark.asyncio
    async def test_success(self):
        v = TestSuiteVerifier("echo ok")
        r = await v.verify("", "")
        assert r.passed is True

    @pytest.mark.asyncio
    async def test_failure(self):
        v = TestSuiteVerifier("false")
        r = await v.verify("", "")
        assert r.passed is False

    @pytest.mark.asyncio
    async def test_timeout(self):
        v = TestSuiteVerifier("sleep 10", timeout=0.05)
        r = await v.verify("", "")
        assert r.passed is False
        assert "Timed out" in r.evidence


class TestExternalModelVerifier:
    @pytest.mark.asyncio
    async def test_pass_on_pass(self):
        model = AsyncMock()
        model.generate.return_value = MagicMock(text="PASS: looks good", error=None)
        v = ExternalModelVerifier(model)
        r = await v.verify("req", "out")
        assert r.passed is True

    @pytest.mark.asyncio
    async def test_fail_on_fail(self):
        model = AsyncMock()
        model.generate.return_value = MagicMock(text="FAIL: missing", error=None)
        v = ExternalModelVerifier(model)
        r = await v.verify("req", "out")
        assert r.passed is False

    @pytest.mark.asyncio
    async def test_ambiguous_defaults_to_fail(self):
        model = AsyncMock()
        model.generate.return_value = MagicMock(text="maybe ok", error=None)
        v = ExternalModelVerifier(model)
        r = await v.verify("req", "out")
        assert r.passed is False


class TestCycleMemory:
    def test_add_and_recall(self):
        m = CycleMemory(max_entries=10)
        m.add(MemoryEntry(cycle_id=1, timestamp=100, input_data="a", output="b", passed=True))
        assert len(m.recent()) == 1
        assert m.recent()[0].cycle_id == 1

    def test_max_entries(self):
        m = CycleMemory(max_entries=3)
        for i in range(5):
            m.add(MemoryEntry(cycle_id=i, timestamp=float(i), input_data=str(i), output=str(i), passed=True))
        assert len(m.recent()) == 3
        assert m.recent()[0].cycle_id == 2

    def test_failures_filter(self):
        m = CycleMemory(max_entries=10)
        m.add(MemoryEntry(cycle_id=1, timestamp=1, input_data="", output="", passed=True))
        m.add(MemoryEntry(cycle_id=2, timestamp=2, input_data="", output="", passed=False))
        m.add(MemoryEntry(cycle_id=3, timestamp=3, input_data="", output="", passed=False))
        assert len(m.failures()) == 2

    def test_success_rate(self):
        m = CycleMemory(max_entries=10)
        m.add(MemoryEntry(cycle_id=1, timestamp=1, input_data="", output="", passed=True))
        m.add(MemoryEntry(cycle_id=2, timestamp=2, input_data="", output="", passed=False))
        assert m.success_rate() == 0.5

    def test_success_rate_empty(self):
        m = CycleMemory(max_entries=10)
        assert m.success_rate() == 1.0

    def test_last(self):
        m = CycleMemory(max_entries=10)
        assert m.last() is None
        m.add(MemoryEntry(cycle_id=1, timestamp=1, input_data="", output="", passed=True))
        assert m.last() is not None
        assert m.last().cycle_id == 1

    def test_clear(self):
        m = CycleMemory(max_entries=10)
        m.add(MemoryEntry(cycle_id=1, timestamp=1, input_data="", output="", passed=True))
        m.clear()
        assert len(m.recent()) == 0

    def test_persistence(self, tmp_path):
        p = tmp_path / "mem.json"
        m1 = CycleMemory(max_entries=10, persist_path=str(p))
        m1.add(MemoryEntry(cycle_id=1, timestamp=1, input_data="x", output="y", passed=True))
        assert p.exists()

        m2 = CycleMemory(max_entries=10, persist_path=str(p))
        assert len(m2.recent()) == 1


class TestLoopRunner:
    @pytest.mark.asyncio
    async def test_success_first_attempt(self):
        process = AsyncMock(return_value="result")
        verifier = AsyncMock()
        verifier.verify.return_value = VerificationResult(passed=True)
        trigger = AsyncMock()
        trigger.wait.return_value = True

        runner = LoopRunner(process, verifier, trigger, max_retries=3)
        result = await runner.run("input")
        assert result.success is True
        assert result.output == "result"
        assert result.total_retries == 0
        assert len(result.cycles) == 1
        process.assert_called_once_with("input")

    @pytest.mark.asyncio
    async def test_retry_then_succeed(self):
        process = AsyncMock(return_value="result")
        verifier = AsyncMock()
        verifier.verify.side_effect = [
            VerificationResult(passed=False, evidence="try again"),
            VerificationResult(passed=True),
        ]
        trigger = AsyncMock()

        runner = LoopRunner(process, verifier, trigger, max_retries=3)
        result = await runner.run("input")
        assert result.success is True
        assert result.total_retries == 1
        assert len(result.cycles) == 2

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self):
        process = AsyncMock(return_value="result")
        verifier = AsyncMock()
        verifier.verify.return_value = VerificationResult(passed=False, evidence="fail")
        trigger = AsyncMock()

        runner = LoopRunner(process, verifier, trigger, max_retries=2)
        result = await runner.run("input")
        assert result.success is False
        assert result.escalated is True
        assert len(result.cycles) == 3

    @pytest.mark.asyncio
    async def test_process_exception_triggers_retry(self):
        process = AsyncMock(side_effect=[ValueError("crash"), "ok"])
        verifier = AsyncMock()
        verifier.verify.return_value = VerificationResult(passed=True)
        trigger = AsyncMock()

        runner = LoopRunner(process, verifier, trigger, max_retries=2)
        result = await runner.run("input")
        assert result.success is True
        assert result.total_retries == 1

    @pytest.mark.asyncio
    async def test_correction_fn_applied_on_failure(self):
        process = AsyncMock(return_value="result")
        verifier = AsyncMock()
        verifier.verify.side_effect = [
            VerificationResult(passed=False, evidence="fix this"),
            VerificationResult(passed=True),
        ]
        trigger = AsyncMock()
        correction = MagicMock(return_value="corrected_input")

        runner = LoopRunner(process, verifier, trigger, max_retries=2, correction_fn=correction)
        result = await runner.run("original")
        assert result.success is True
        correction.assert_called_once()
        # process should be called with corrected input on second try
        assert process.call_args_list[1][0][0] == "corrected_input"

    @pytest.mark.asyncio
    async def test_escalation_handler_called(self):
        process = AsyncMock(return_value="result")
        verifier = AsyncMock()
        verifier.verify.return_value = VerificationResult(passed=False, evidence="fail")
        trigger = AsyncMock()
        escalation = AsyncMock()

        runner = LoopRunner(process, verifier, trigger, max_retries=1, escalation_handler=escalation)
        result = await runner.run("input")
        assert result.escalated is True
        escalation.escalate.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_cycle_count_persists_across_runs(self):
        process = AsyncMock(return_value="ok")
        verifier = AsyncMock()
        verifier.verify.return_value = VerificationResult(passed=True)
        trigger = AsyncMock()

        runner = LoopRunner(process, verifier, trigger)
        r1 = await runner.run("a")
        r2 = await runner.run("b")
        assert r1.cycles[0].id == 1
        assert r2.cycles[0].id == 2

    @pytest.mark.asyncio
    async def test_memory_persists_across_runs(self):
        process = AsyncMock(return_value="ok")
        verifier = AsyncMock()
        verifier.verify.return_value = VerificationResult(passed=True)
        trigger = AsyncMock()

        runner = LoopRunner(process, verifier, trigger, memory=CycleMemory())
        await runner.run("a")
        await runner.run("b")
        assert len(runner.memory.recent()) == 2
