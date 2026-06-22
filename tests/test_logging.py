import json
import io
import logging
import re
from zilli.infra.logging import StructuredFormatter, get_trace_id, set_trace_id


class TestStructuredFormatter:
    def setup_method(self):
        self.formatter = StructuredFormatter()
        self.stream = io.StringIO()
        self.handler = logging.StreamHandler(self.stream)
        self.handler.setFormatter(self.formatter)
        self.logger = logging.getLogger("test_logging")
        self.logger.addHandler(self.handler)
        self.logger.setLevel(logging.DEBUG)

    def test_json_output(self):
        self.logger.info("hello world")
        output = self.stream.getvalue()
        data = json.loads(output)
        assert data["msg"] == "hello world"
        assert data["level"] == "INFO"
        assert data["logger"] == "test_logging"

    def test_trace_id_present(self):
        tid = set_trace_id("abc123")
        self.logger.info("test")
        output = self.stream.getvalue()
        data = json.loads(output)
        assert data["trace_id"] == "abc123"


class TestTraceId:
    def test_set_and_get(self):
        tid = set_trace_id()
        assert len(tid) == 12
        assert get_trace_id() == tid
        assert re.match(r"^[a-f0-9]{12}$", tid)

    def test_custom_tid(self):
        set_trace_id("my-custom-id")
        assert get_trace_id() == "my-custom-id"
