from __future__ import annotations

import os
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "pending"
    SCHEDULED = "scheduled"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    SKIPPED = "skipped"


@dataclass
class CeleryConfig:
    broker_url: str = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
    result_backend: str = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/0")
    task_track_started: bool = True
    task_acks_late: bool = True
    worker_prefetch_multiplier: int = 1
    task_soft_time_limit: int = 600
    task_time_limit: int = 900
    task_max_retries: int = 3
    task_default_retry_delay: int = 30
    result_expires: int = 86400  # 24h
    event_serializer: str = "json"
    task_serializer: str = "json"
    result_serializer: str = "json"
    accept_content: list[str] = field(default_factory=lambda: ["json"])


_celery_app = None


def create_celery(config: CeleryConfig | None = None) -> Any:
    global _celery_app
    if _celery_app is not None:
        return _celery_app

    try:
        from celery import Celery
    except ImportError:
        raise ImportError("Install `celery[redis]` to use the workflow engine")

    cfg = config or CeleryConfig()
    app = Celery("zilli", broker=cfg.broker_url, backend=cfg.result_backend)
    app.conf.update(
        task_track_started=cfg.task_track_started,
        task_acks_late=cfg.task_acks_late,
        worker_prefetch_multiplier=cfg.worker_prefetch_multiplier,
        task_soft_time_limit=cfg.task_soft_time_limit,
        task_time_limit=cfg.task_time_limit,
        task_max_retries=cfg.task_max_retries,
        task_default_retry_delay=cfg.task_default_retry_delay,
        result_expires=cfg.result_expires,
        event_serializer=cfg.event_serializer,
        task_serializer=cfg.task_serializer,
        result_serializer=cfg.result_serializer,
        accept_content=cfg.accept_content,
    )

    app.autodiscover_tasks(["zilli.workflow"])
    _celery_app = app
    return app


celery_app = create_celery()
