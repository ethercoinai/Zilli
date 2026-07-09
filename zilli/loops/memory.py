from __future__ import annotations

import json
import logging
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from zilli.loops.base import LoopCycle

logger = logging.getLogger("zilli.loops.memory")


@dataclass
class MemoryEntry:
    cycle_id: int
    timestamp: float
    input_data: Any
    output: Any
    passed: bool
    evidence: str = ""
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class CycleMemory:
    def __init__(self, max_entries: int = 1000, persist_path: Optional[str] = None):
        self._entries: deque[MemoryEntry] = deque(maxlen=max_entries)
        self._persist_path = Path(persist_path) if persist_path else None
        self._load()

    def add(self, entry: MemoryEntry) -> None:
        self._entries.append(entry)
        self._save()

    def add_from_cycle(self, cycle: LoopCycle) -> None:
        entry = MemoryEntry(
            cycle_id=cycle.id,
            timestamp=time.time(),
            input_data=cycle.input_data,
            output=cycle.output,
            passed=cycle.verification.passed if cycle.verification else False,
            evidence=cycle.verification.evidence if cycle.verification else "",
            duration_ms=cycle.duration_ms,
            metadata=cycle.metadata,
        )
        self.add(entry)

    def recent(self, n: int = 10) -> list[MemoryEntry]:
        return list(self._entries)[-n:]

    def last(self) -> Optional[MemoryEntry]:
        return self._entries[-1] if self._entries else None

    def failures(self, n: int = 10) -> list[MemoryEntry]:
        return [e for e in self._entries if not e.passed][-n:]

    def success_rate(self, n: int = 50) -> float:
        recent = list(self._entries)[-n:]
        if not recent:
            return 1.0
        return sum(1 for e in recent if e.passed) / len(recent)

    def summary(self, exclude_failed: bool = False) -> str:
        total = len(self._entries)
        passes = sum(1 for e in self._entries if e.passed)
        failures = total - passes
        recent_pass_rate = self.success_rate(n=20)
        return (
            f"Memory: {total} cycles, {passes} passed ({'%.0f' % (passes/total*100 if total else 100)}%), "
            f"{failures} failed, recent_pass_rate={'%.0f' % (recent_pass_rate*100)}%, "
            f"last: {self._entries[-1].cycle_id if self._entries else 'none'}"
        )

    def clear(self) -> None:
        self._entries.clear()
        self._save()

    def _save(self) -> None:
        if not self._persist_path:
            return
        try:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            data = [asdict(e) for e in self._entries]
            with open(self._persist_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.warning("Failed to persist memory: %s", e)

    def _load(self) -> None:
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            with open(self._persist_path) as f:
                data = json.load(f)
            for item in data:
                self._entries.append(MemoryEntry(**item))
        except Exception as e:
            logger.warning("Failed to load memory: %s", e)
