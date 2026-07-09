from zilli.privacy.classifier import DataClass
from zilli.privacy.sandbox import (
    PrivacyBudget,
    PrivacySandbox,
    SandboxConfig,
    SandboxStatus,
)


class TestPrivacyBudget:
    def test_remaining(self):
        b = PrivacyBudget(total_epsilon=1.0, used_epsilon=0.3)
        assert b.remaining_epsilon == 0.7

    def test_exhausted_by_epsilon(self):
        b = PrivacyBudget(total_epsilon=1.0, used_epsilon=1.0)
        assert b.exhausted

    def test_exhausted_by_queries(self):
        b = PrivacyBudget(total_epsilon=10.0, max_queries=5, query_count=5)
        assert b.exhausted


class TestPrivacySandbox:
    def test_init(self):
        sb = PrivacySandbox()
        assert sb.status == SandboxStatus.CREATED

    def test_activate(self):
        sb = PrivacySandbox()
        sb.activate()
        assert sb.status == SandboxStatus.ACTIVE

    def test_destroy(self):
        sb = PrivacySandbox()
        sb.activate()
        sb.destroy()
        assert sb.status == SandboxStatus.DESTROYED

    def test_execute_when_not_active(self):
        sb = PrivacySandbox()
        result = sb.execute("test", lambda x: "ok")
        assert not result.allowed
        assert "not active" in result.error

    def test_execute_success(self):
        sb = PrivacySandbox()
        sb.activate()
        result = sb.execute("hello world", lambda x: f"processed: {x}")
        assert result.allowed
        assert result.result == "processed: hello world"

    def test_execute_rejects_high_classification(self):
        config = SandboxConfig(data_class_min=DataClass.PUBLIC)
        sb = PrivacySandbox(config=config)
        sb.activate()
        result = sb.execute("This contains SSN 123-45-6789", lambda x: "ok")
        assert not result.allowed

    def test_budget_tracking(self):
        sb = PrivacySandbox()
        sb.activate()
        for _ in range(5):
            sb.execute("test", lambda x: "ok")
        assert sb.budget.query_count == 5

    def test_budget_exhaustion(self):
        config = SandboxConfig(privacy_budget=PrivacyBudget(total_epsilon=0.05, max_queries=3))
        sb = PrivacySandbox(config=config)
        sb.activate()
        for _ in range(3):
            sb.execute("test", lambda x: "ok")
        result = sb.execute("test", lambda x: "ok")
        assert not result.allowed
        assert "exhausted" in result.error

    def test_audit_log(self):
        sb = PrivacySandbox()
        sb.activate()
        sb.execute("test1", lambda x: "ok")
        sb.execute("test2", lambda x: "ok")
        log = sb.get_audit_log()
        assert len(log) == 2

    def test_summary(self):
        sb = PrivacySandbox()
        sb.activate()
        sb.execute("test", lambda x: "ok")
        s = sb.summary()
        assert s["total_executions"] == 1
        assert s["successful"] == 1

    def test_differential_privacy(self):
        sb = PrivacySandbox()
        val_raw = sb._apply_differential_privacy(1.0, sensitivity=1.0, epsilon=0.1)
        assert isinstance(val_raw, float)
        assert val_raw != 1.0

        sb.activate()
        result = sb.execute("compute", lambda x: 42.0, epsilon=0.05)
        assert result.allowed
        assert isinstance(result.result, float)
        assert result.result != 42.0
        assert sb.budget.used_epsilon == 0.05


__all__ = ["TestPrivacyBudget", "TestPrivacySandbox"]
