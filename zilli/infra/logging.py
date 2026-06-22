import json
import logging
import threading
import time
import uuid
from contextvars import ContextVar
from typing import Any, Dict, Optional

_trace_id: ContextVar[str] = ContextVar("trace_id", default="")


def get_trace_id() -> str:
    return _trace_id.get()


def set_trace_id(tid: Optional[str] = None) -> str:
    if tid is None:
        tid = uuid.uuid4().hex[:12]
    _trace_id.set(tid)
    return tid


class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        base = {
            "ts": time.time(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "trace_id": get_trace_id(),
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            base.update(record.extra)
        if record.exc_info and record.exc_info[0]:
            base["exc"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)


def configure_logging(level: str = "INFO"):
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredFormatter())
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), handlers=[handler], force=True)


class TraceLogger:
    def __init__(self, name: str):
        self._logger = logging.getLogger(name)

    def _log(self, level: str, msg: str, **extra: Any):
        record = self._logger.makeRecord(
            self._logger.name, getattr(logging, level.upper(), logging.INFO),
            "", 0, msg, (), None,
        )
        record.__dict__.setdefault("extra", {}).update(extra)
        record.trace_id = get_trace_id()
        self._logger.handle(record)

    def info(self, msg: str, **extra):
        self._log("info", msg, **extra)

    def warning(self, msg: str, **extra):
        self._log("warning", msg, **extra)

    def error(self, msg: str, **extra):
        self._log("error", msg, **extra)

    def debug(self, msg: str, **extra):
        self._log("debug", msg, **extra)


__all__ = ["configure_logging", "StructuredFormatter", "TraceLogger",
           "get_trace_id", "set_trace_id"]
