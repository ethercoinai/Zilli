"""ACE-style context engineering: structured context as an evolving playbook.

Based on Zhang et al. 2025 "Agentic Context Engineering (ACE)" and
Lilian Weng's Harness Engineering for Self-Improvement (Jul 2026).

Components:
  1. Generator — produces task trajectories with reference to bullet points
  2. Reflector — distills insights from successful and failed trajectories
  3. Curator — updates structured context with incremental, itemized entries
     Never rewrites a full prompt blob. Outputs (identifier, description) pairs.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger("zilli.loops.context_curator")


@dataclass
class ContextBullet:
    """A single structured context entry."""
    id: str
    description: str
    category: str = "general"
    confidence: float = 0.5
    created_at: float = 0.0
    last_accessed: float = 0.0
    hit_count: int = 0
    source_trajectory: str = ""


@dataclass
class Trajectory:
    """A task execution trajectory for reflection."""
    task_id: str
    actions: list[dict]
    outcome: str  # "success" | "failure"
    verifier_evidence: str = ""
    duration_ms: float = 0.0


class ContextCurator:
    """ACE Curator: maintains a structured context playbook of bullet points.

    The curator never rewrites a full prompt blob. Instead it outputs
    incremental, itemized entries that are merged deterministically.
    """

    def __init__(self, persist_path: str | None = None):
        self._bullets: OrderedDict[str, ContextBullet] = OrderedDict()
        self._path = Path(persist_path) if persist_path else None
        self._load()

    def add_bullet(
        self,
        description: str,
        category: str = "general",
        confidence: float = 0.5,
        source_trajectory: str = "",
    ) -> str:
        bullet_id = uuid.uuid4().hex[:12]
        now = time.time()
        bullet = ContextBullet(
            id=bullet_id,
            description=description,
            category=category,
            confidence=confidence,
            created_at=now,
            last_accessed=now,
            source_trajectory=source_trajectory[:200],
        )
        self._bullets[bullet_id] = bullet
        self._prune()
        self._save()
        return bullet_id

    def get(self, category: str | None = None) -> list[ContextBullet]:
        if category:
            bullets = [b for b in self._bullets.values() if b.category == category]
        else:
            bullets = list(self._bullets.values())
        for b in bullets:
            b.last_accessed = time.time()
            b.hit_count += 1
        return bullets

    def format_context(self, max_bullets: int = 20) -> str:
        """Format context as structured markdown for the agent."""
        bullets = sorted(
            self._bullets.values(),
            key=lambda b: (b.confidence, b.hit_count),
            reverse=True,
        )[:max_bullets]

        lines = ["## Curated Context Playbook"]
        for b in bullets:
            lines.append(f"- [{b.id}] ({b.confidence:.1f}) {b.description}")
        return "\n".join(lines)

    def reflect(self, trajectories: list[Trajectory]) -> list[str]:
        """ACE Reflector: distill insights from trajectories into new bullets.

        Returns IDs of newly created bullets.
        """
        new_ids: list[str] = []
        failures = [t for t in trajectories if t.outcome == "failure"]
        successes = [t for t in trajectories if t.outcome == "success"]

        if successes:
            common = self._extract_common_patterns(successes)
            for pattern in common:
                bid = self.add_bullet(
                    description=pattern,
                    category="success_pattern",
                    confidence=0.6,
                    source_trajectory=f"{len(successes)} successful runs",
                )
                new_ids.append(bid)

        if failures:
            pitfalls = self._extract_common_patterns(failures)
            for pitfall in pitfalls:
                bid = self.add_bullet(
                    description=f"AVOID: {pitfall}",
                    category="pitfall",
                    confidence=0.7,
                    source_trajectory=f"{len(failures)} failed runs",
                )
                new_ids.append(bid)

        self._save()
        return new_ids

    def _extract_common_patterns(self, trajectories: list[Trajectory]) -> list[str]:
        """Simple heuristic: collect unique verifier evidence strings.

        In production, this would call an LLM for semantic distillation.
        """
        patterns: set[str] = set()
        for t in trajectories:
            if t.verifier_evidence:
                patterns.add(t.verifier_evidence[:120])
        return list(patterns)

    def _prune(self, max_bullets: int = 200) -> None:
        """Keep only top-N bullets by confidence + hit count."""
        if len(self._bullets) <= max_bullets:
            return
        sorted_bullets = sorted(
            self._bullets.values(),
            key=lambda b: (b.confidence, b.hit_count),
            reverse=True,
        )[:max_bullets]
        self._bullets = OrderedDict((b.id, b) for b in sorted_bullets)

    def _load(self) -> None:
        if not self._path or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text())
            for item in data:
                bullet = ContextBullet(**item)
                self._bullets[bullet.id] = bullet
        except Exception as e:
            logger.debug("Failed to load context: %s", e)

    def _save(self) -> None:
        if not self._path:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            data = [
                {
                    "id": b.id,
                    "description": b.description,
                    "category": b.category,
                    "confidence": b.confidence,
                    "created_at": b.created_at,
                    "hit_count": b.hit_count,
                    "source_trajectory": b.source_trajectory,
                }
                for b in self._bullets.values()
            ]
            self._path.write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.debug("Failed to persist context: %s", e)

    @property
    def bullet_count(self) -> int:
        return len(self._bullets)

    def clear(self) -> None:
        self._bullets.clear()
