import tempfile
from pathlib import Path

from zilli.envs.cost_controller import CostController


def _make_cc(monthly_budget=100.0):
    tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    tmp.close()
    cc = CostController(budget_file=tmp.name, monthly_budget=monthly_budget)
    cc._file = Path(tmp.name)
    return cc


def _cleanup(cc):
    if cc._file.exists():
        cc._file.unlink()


class TestCostController:
    def teardown_method(self):
        # clean up default file if any test wrote to it
        default = Path.home() / ".zilli_budget.json"
        if default.exists():
            default.unlink()

    def test_default_init(self):
        cc = _make_cc(100.0)
        assert cc.scheduler.remaining_budget == 100.0
        assert cc.scheduler.total_calls == 0
        _cleanup(cc)

    def test_should_use_planner_default(self):
        cc = _make_cc(500.0)
        result = cc.should_use_planner("code_gen", {"max_prob": 0.9})
        assert isinstance(result, bool)
        _cleanup(cc)

    def test_record_planner_call(self):
        cc = _make_cc(100.0)
        cc.record_planner_call("code_gen", True)
        assert cc.scheduler.total_calls == 1
        assert cc.scheduler.remaining_budget < 100.0
        _cleanup(cc)

    def test_record_executor_call(self):
        cc = _make_cc(100.0)
        cc.record_executor_call("code_gen", True)
        assert cc.scheduler.total_calls == 0
        assert cc.scheduler.remaining_budget == 100.0
        _cleanup(cc)

    def test_snapshot(self):
        cc = _make_cc(100.0)
        snap = cc.snapshot()
        assert snap.remaining_budget == 100.0
        assert snap.total_calls == 0
        assert snap.emergency_mode is False
        _cleanup(cc)

    def test_emergency_mode(self):
        cc = _make_cc(100.0)
        cc.scheduler.remaining_budget = 5.0
        assert cc.snapshot().emergency_mode is True
        _cleanup(cc)

    def test_reset_monthly(self):
        cc = _make_cc(100.0)
        cc.record_planner_call("test", True)
        cc.reset_monthly()
        assert cc.scheduler.remaining_budget == 100.0
        assert cc.scheduler.total_calls == 0
        _cleanup(cc)

    def test_persistence(self):
        cc = _make_cc(200.0)
        cc.record_planner_call("test", True)
        remaining = cc.scheduler.remaining_budget

        cc2 = CostController(budget_file=str(cc._file), monthly_budget=200.0)
        assert cc2.scheduler.remaining_budget == remaining
        assert cc2.scheduler.total_calls == 1
        _cleanup(cc)
        _cleanup(cc2)

    def test_persistence_corrupt_file(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json")
            tmp = f.name
        try:
            cc = CostController(budget_file=tmp, monthly_budget=300.0)
            assert cc.scheduler.remaining_budget == 300.0
        finally:
            Path(tmp).unlink(missing_ok=True)

    def test_reset_hourly(self):
        cc = _make_cc(100.0)
        cc.scheduler.calls_this_hour = 50
        cc.reset_hourly()
        assert cc.scheduler.calls_this_hour == 0
        _cleanup(cc)
