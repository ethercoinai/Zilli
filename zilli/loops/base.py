from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Generic, Optional, TypeVar

T = TypeVar("T")


@dataclass
class VerificationResult:
    passed: bool
    evidence: str = ""
    details: str = ""
    confidence: float = 1.0


@dataclass
class CostEntry:
    tokens_input: int = 0
    tokens_output: int = 0
    cost_usd: float = 0.0
    api_calls: int = 0


@dataclass
class LoopCycle(Generic[T]):
    id: int
    input_data: T
    output: Any = None
    verification: Optional[VerificationResult] = None
    duration_ms: float = 0.0
    error: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    cost: CostEntry = field(default_factory=CostEntry)


@dataclass
class LoopResult(Generic[T]):
    success: bool
    output: Any = None
    cycles: list[LoopCycle[T]] = field(default_factory=list)
    escalated: bool = False
    escalation_data: Optional[dict[str, Any]] = None
    total_duration_ms: float = 0.0
    total_retries: int = 0
    total_cost: CostEntry = field(default_factory=CostEntry)


class Verifier(ABC):
    @abstractmethod
    async def verify(self, input_data: Any, output: Any) -> VerificationResult:
        ...


class SkillProvider(ABC):
    """Reads reusable skill instructions the loop reads every cycle."""

    @abstractmethod
    async def load_skill(self, name: str) -> str:
        ...


class Trigger(ABC):
    @abstractmethod
    async def wait(self) -> bool:
        ...

    @abstractmethod
    async def reset(self) -> None:
        ...


class EscalationHandler(ABC):
    @abstractmethod
    async def escalate(self, cycle: LoopCycle, history: list[LoopCycle]) -> None:
        ...
