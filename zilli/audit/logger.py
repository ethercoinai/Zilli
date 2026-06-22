from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from zilli.security.pii import Sanitizer

if TYPE_CHECKING:
    from zilli.configs import ZilliConfig

logger = logging.getLogger("zilli.audit")


class AuditLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class AuditEvent:
    event_type: str
    level: AuditLevel
    message: str
    tenant_id: str = "default"
    model_name: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: float = 0.0
    route_type: str = ""
    success: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)


class AuditLogger:
    def __init__(self, log_dir: str = "./audit_logs", sanitize: bool = True,
                 config: Optional["ZilliConfig"] = None):
        if config is not None:
            cfg = config.audit
            log_dir = cfg.log_dir
            sanitize = cfg.sanitize
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.sanitizer = Sanitizer() if sanitize else None

    def log(self, event: AuditEvent):
        data = asdict(event)

        if self.sanitizer:
            for text_field in ("message",):
                if text_field in data and isinstance(data[text_field], str):
                    data[text_field] = self.sanitizer.sanitize_for_log(data[text_field])

        date_str = time.strftime("%Y-%m-%d", time.localtime(event.timestamp))
        log_path = self.log_dir / f"audit_{date_str}.jsonl"

        try:
            with open(log_path, "a", buffering=1) as f:
                f.write(json.dumps(data, ensure_ascii=False) + "\n")
                f.flush()
        except OSError as e:
            logger.error("Failed to write audit log: %s", e)

    def model_call(
        self,
        model_name: str,
        prompt: str,
        response: str,
        tokens_in: int = 0,
        tokens_out: int = 0,
        duration_ms: float = 0.0,
        tenant_id: str = "default",
        success: bool = True,
    ):
        event = AuditEvent(
            event_type="model_call",
            level=AuditLevel.INFO if success else AuditLevel.ERROR,
            message=f"Model call: {model_name}",
            tenant_id=tenant_id,
            model_name=model_name,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            duration_ms=duration_ms,
            success=success,
            metadata={
                "prompt_len": len(prompt),
                "response_len": len(response),
            },
        )
        self.log(event)

    def route_decision(
        self,
        route_type: str,
        request: str,
        reason: str,
        tenant_id: str = "default",
    ):
        event = AuditEvent(
            event_type="route_decision",
            level=AuditLevel.INFO,
            message=f"Route: {route_type} ({reason})",
            tenant_id=tenant_id,
            route_type=route_type,
            metadata={"request_len": len(request)},
        )
        self.log(event)

    def data_access(
        self,
        action: str,
        resource: str,
        tenant_id: str = "default",
        success: bool = True,
    ):
        event = AuditEvent(
            event_type="data_access",
            level=AuditLevel.INFO if success else AuditLevel.WARNING,
            message=f"Data access: {action} on {resource}",
            tenant_id=tenant_id,
            success=success,
        )
        self.log(event)

    def get_logs(self, date: Optional[str] = None, limit: int = 100) -> list[dict]:
        if date:
            log_path = self.log_dir / f"audit_{date}.jsonl"
        else:
            log_files = sorted(self.log_dir.glob("audit_*.jsonl"), reverse=True)
            log_path = log_files[0] if log_files else None

        if not log_path or not log_path.exists():
            return []

        events: list[dict] = []
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
                    if len(events) >= limit:
                        break
        return events
