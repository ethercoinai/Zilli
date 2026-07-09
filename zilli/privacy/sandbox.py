from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Coroutine, Optional

from zilli.privacy.classifier import ClassificationResult, DataClass, DataClassifier

logger = logging.getLogger("zilli.privacy.sandbox")


class SandboxStatus(str, Enum):
    CREATED = "created"
    ACTIVE = "active"
    COMPLETED = "completed"
    FAILED = "failed"
    DESTROYED = "destroyed"


@dataclass
class PrivacyBudget:
    total_epsilon: float = 1.0
    used_epsilon: float = 0.0
    delta: float = 1e-5
    max_queries: int = 1000
    query_count: int = 0

    @property
    def remaining_epsilon(self) -> float:
        return max(0.0, self.total_epsilon - self.used_epsilon)

    @property
    def exhausted(self) -> bool:
        return self.used_epsilon >= self.total_epsilon or self.query_count >= self.max_queries


@dataclass
class SandboxConfig:
    max_memory_mb: int = 512
    max_cpu_time_s: float = 300.0
    network_access: bool = False
    privacy_budget: PrivacyBudget = field(default_factory=PrivacyBudget)
    data_class_min: DataClass = DataClass.PUBLIC
    isolation_level: str = "process"


@dataclass
class SandboxExecution:
    sandbox_id: str
    query_text: str
    classification: ClassificationResult
    allowed: bool
    result: Any = None
    error: Optional[str] = None
    epsilon_cost: float = 0.0
    timestamp: float = 0.0


class PrivacySandbox:
    def __init__(
        self,
        config: Optional[SandboxConfig] = None,
        classifier: Optional[DataClassifier] = None,
    ):
        self.config = config or SandboxConfig()
        self.classifier = classifier or DataClassifier()
        self._sandbox_id = str(uuid.uuid4())[:8]
        self._status = SandboxStatus.CREATED
        self._budget = self.config.privacy_budget
        self._executions: list[SandboxExecution] = []
        self._data_store: dict[str, Any] = {}

    @property
    def status(self) -> SandboxStatus:
        return self._status

    @property
    def budget(self) -> PrivacyBudget:
        return self._budget

    def activate(self) -> None:
        self._status = SandboxStatus.ACTIVE
        logger.info(f"Sandbox {self._sandbox_id} activated")

    def destroy(self) -> None:
        self._status = SandboxStatus.DESTROYED
        self._data_store.clear()
        logger.info(f"Sandbox {self._sandbox_id} destroyed")

    def _check_budget(self, estimated_epsilon: float = 0.01) -> bool:
        if self._budget.exhausted:
            return False
        if self._budget.used_epsilon + estimated_epsilon > self._budget.total_epsilon:
            return False
        return True

    def _apply_differential_privacy(self, value: float, sensitivity: float = 1.0, epsilon: float = 0.01) -> float:
        import numpy as np
        if epsilon <= 0:
            return value
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale)
        return float(value + noise)

    def execute(
        self,
        query_text: str,
        handler: Callable[[str], Coroutine[Any, Any, Any]],
        epsilon: float = 0.01,
    ) -> SandboxExecution:
        if self._status != SandboxStatus.ACTIVE:
            return SandboxExecution(
                sandbox_id=self._sandbox_id,
                query_text=query_text,
                classification=ClassificationResult(data_class=DataClass.PUBLIC),
                allowed=False,
                error=f"Sandbox not active (status={self._status.value})",
            )

        classification = self.classifier.classify(query_text)

        if not self._check_budget(epsilon):
            return SandboxExecution(
                sandbox_id=self._sandbox_id,
                query_text=query_text,
                classification=classification,
                allowed=False,
                error="Privacy budget exhausted",
                timestamp=time.time(),
            )

        class_level = classification.data_class
        if self._data_class_level(class_level) > self._data_class_level(self.config.data_class_min):
            return SandboxExecution(
                sandbox_id=self._sandbox_id,
                query_text=query_text,
                classification=classification,
                allowed=False,
                error=f"Data class {class_level.value} exceeds minimum {self.config.data_class_min.value}",
                timestamp=time.time(),
            )

        if not self.config.network_access and classification.requires_sanitization:
            return SandboxExecution(
                sandbox_id=self._sandbox_id,
                query_text=query_text,
                classification=classification,
                allowed=False,
                error="Network access denied for sensitive data",
                timestamp=time.time(),
            )

        import asyncio
        import concurrent.futures
        try:
            import inspect
            if inspect.iscoroutinefunction(handler):
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(lambda: asyncio.run(handler(query_text)))
                    result = future.result(timeout=self.config.max_cpu_time_s)
            else:
                result = handler(query_text)
        except Exception as e:
            self._status = SandboxStatus.FAILED
            return SandboxExecution(
                sandbox_id=self._sandbox_id,
                query_text=query_text,
                classification=classification,
                allowed=False,
                error=str(e),
                timestamp=time.time(),
            )

        self._budget.used_epsilon += epsilon
        self._budget.query_count += 1

        if isinstance(result, (int, float)):
            result = self._apply_differential_privacy(float(result), epsilon=epsilon)

        execution = SandboxExecution(
            sandbox_id=self._sandbox_id,
            query_text=query_text,
            classification=classification,
            allowed=True,
            result=result,
            epsilon_cost=epsilon,
            timestamp=time.time(),
        )
        self._executions.append(execution)
        return execution

    def _data_class_level(self, dc: DataClass) -> int:
        from zilli.privacy.classifier import CLASS_LEVEL
        return CLASS_LEVEL.get(dc, 0)

    def get_audit_log(self) -> list[dict[str, Any]]:
        return [
            {
                "sandbox_id": e.sandbox_id,
                "query": e.query_text[:100],
                "data_class": e.classification.data_class.value,
                "allowed": e.allowed,
                "epsilon_cost": e.epsilon_cost,
                "timestamp": e.timestamp,
                "error": e.error,
            }
            for e in self._executions
        ]

    def summary(self) -> dict[str, Any]:
        return {
            "sandbox_id": self._sandbox_id,
            "status": self._status.value,
            "budget_remaining_epsilon": self._budget.remaining_epsilon,
            "budget_used_epsilon": self._budget.used_epsilon,
            "query_count": self._budget.query_count,
            "total_executions": len(self._executions),
            "successful": sum(1 for e in self._executions if e.allowed),
            "rejected": sum(1 for e in self._executions if not e.allowed),
        }


__all__ = ["PrivacySandbox", "SandboxConfig", "SandboxStatus", "PrivacyBudget", "SandboxExecution"]
