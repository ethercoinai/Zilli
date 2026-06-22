import json
import tempfile
from pathlib import Path

from zilli.audit.logger import AuditEvent, AuditLevel, AuditLogger


class TestAuditLogger:
    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.logger = AuditLogger(log_dir=self.tmpdir, sanitize=False)

    def _get_log_lines(self) -> list[dict]:
        log_files = sorted(Path(self.tmpdir).glob("audit_*.jsonl"))
        if not log_files:
            return []
        events = []
        with open(log_files[-1]) as f:
            for line in f:
                if line.strip():
                    events.append(json.loads(line))
        return events

    def test_log_event(self):
        event = AuditEvent(
            event_type="test_event",
            level=AuditLevel.INFO,
            message="test message",
        )
        self.logger.log(event)
        logs = self._get_log_lines()
        assert len(logs) == 1
        assert logs[0]["event_type"] == "test_event"
        assert logs[0]["message"] == "test message"

    def test_model_call(self):
        self.logger.model_call(
            model_name="test-model",
            prompt="hello",
            response="world",
            tokens_in=5,
            tokens_out=10,
            duration_ms=100.0,
        )
        logs = self._get_log_lines()
        assert len(logs) == 1
        assert logs[0]["event_type"] == "model_call"
        assert logs[0]["model_name"] == "test-model"

    def test_route_decision(self):
        self.logger.route_decision(
            route_type="full_route",
            request="复杂任务",
            reason="matched pattern",
        )
        logs = self._get_log_lines()
        assert len(logs) == 1
        assert logs[0]["event_type"] == "route_decision"
        assert logs[0]["route_type"] == "full_route"

    def test_data_access(self):
        self.logger.data_access(
            action="read",
            resource="patient_records",
            tenant_id="hospital_a",
        )
        logs = self._get_log_lines()
        assert len(logs) == 1
        assert logs[0]["event_type"] == "data_access"
        assert logs[0]["tenant_id"] == "hospital_a"

    def test_multiple_events(self):
        for i in range(3):
            self.logger.log(AuditEvent(
                event_type="test", level=AuditLevel.INFO, message=f"event {i}",
            ))
        logs = self._get_log_lines()
        assert len(logs) == 3

    def test_audit_levels(self):
        assert AuditLevel.INFO.value == "info"
        assert AuditLevel.ERROR.value == "error"
        assert AuditLevel.CRITICAL.value == "critical"

    def test_sanitize_pii(self):
        from zilli.audit.logger import AuditLogger
        sanitizing_logger = AuditLogger(log_dir=self.tmpdir, sanitize=True)
        sanitizing_logger.model_call(
            model_name="m",
            prompt="Email: test@example.com",
            response="ok",
        )
        logs = sanitizing_logger.get_logs()
        assert len(logs) >= 1
        assert "test@example.com" not in logs[0].get("message", "")

    def test_get_logs_empty(self):
        from zilli.audit.logger import AuditLogger
        empty = AuditLogger(log_dir=self.tmpdir, sanitize=False)
        logs = empty.get_logs(date="2099-01-01")
        assert logs == []
