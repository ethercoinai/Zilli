"""Self-Harness: three-stage meta-loop for improving the harness itself.

Based on Zhang et al. 2026 "Self-Harness: Harnesses That Improve Themselves"
and Lilian Weng's Harness Engineering for Self-Improvement (Jul 2026).

Stages:
  1. Weakness Mining — collect execution traces, cluster failures
  2. Bounded Harness Proposal — propose narrow edits to harness code
  3. Proposal Validation — regression test on held-in/held-out splits
"""

from __future__ import annotations

import asyncio
import difflib
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable

from zilli.loops.base import LoopResult, Verifier
from zilli.loops.failure_analyzer import (
    FailureCluster,
    FailureRecord,
    WeaknessMiner,
)

logger = logging.getLogger("zilli.loops.harness_orchestrator")


@dataclass
class HarnessEdit:
    description: str
    diff: str
    source_file: str
    target_pattern: str
    accepted: bool = False
    rejection_reason: str = ""
    validation_metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class HarnessCandidate:
    version: str
    edits: list[HarnessEdit]
    held_in_pass_rate: float = 0.0
    held_out_pass_rate: float = 0.0
    accepted: bool = False


EditingSurface = Callable[[], str]  # returns current harness source
EvaluationTask = Callable[[], Awaitable[dict[str, bool]]]  # returns {task_id: passed}


class HarnessOrchestrator:
    """Three-stage Self-Harness loop.

    Parameters
    ----------
    current_version : str
        Semver of the current harness.
    editable_surfaces : dict[str, EditingSurface]
        Named editable regions of the harness (code as string).
    miner : WeaknessMiner
        Failure clustering engine.
    held_in_tasks : list[EvaluationTask]
        Tasks to verify a weakness is resolved.
    held_out_tasks : list[EvaluationTask]
        Tasks to check for regressions.
    max_edits_per_round : int
        Max harness edits per proposal stage.
    improvement_threshold : float
        Minimum held-in improvement to accept.
    """

    def __init__(
        self,
        current_version: str,
        editable_surfaces: dict[str, EditingSurface],
        miner: WeaknessMiner,
        held_in_tasks: list[EvaluationTask],
        held_out_tasks: list[EvaluationTask],
        max_edits_per_round: int = 5,
        improvement_threshold: float = 0.05,
        harness_dir: str | None = None,
    ):
        self._version = current_version
        self._surfaces = editable_surfaces
        self._miner = miner
        self._held_in = held_in_tasks
        self._held_out = held_out_tasks
        self._max_edits = max_edits_per_round
        self._threshold = improvement_threshold
        self._harness_dir = Path(harness_dir) if harness_dir else Path.cwd()
        self._history: list[HarnessCandidate] = []
        self._rejected_edits: list[HarnessEdit] = []

    async def run_cycle(self, traces: list[dict]) -> HarnessCandidate | None:
        """One full Self-Harness iteration."""
        # Stage 1: Weakness Mining
        clusters = self._miner.cluster_failures(traces)
        if not clusters:
            logger.info("No actionable failure patterns found")
            return None

        # Stage 2: Bounded Proposal
        proposals = await self._propose_edits(clusters)
        if not proposals:
            logger.info("No proposals generated")
            return None

        # Stage 3: Validation
        candidate = HarnessCandidate(
            version=f"{self._version}-{len(self._history) + 1}",
            edits=proposals,
        )
        await self._validate(candidate)

        if candidate.accepted:
            self._history.append(candidate)
            self._version = candidate.version
            logger.info(
                "Accepted harness %s (held_in=%.2f, held_out=%.2f)",
                candidate.version,
                candidate.held_in_pass_rate,
                candidate.held_out_pass_rate,
            )
        else:
            for edit in proposals:
                if not edit.accepted:
                    self._rejected_edits.append(edit)

        return candidate

    async def _propose_edits(
        self, clusters: list[FailureCluster]
    ) -> list[HarnessEdit]:
        """Generate bounded harness edits from failure clusters.

        For each cluster, propose a narrow edit addressing the recurrent
        failure pattern. Edits reference editable surfaces to stay bounded.
        """
        edits: list[HarnessEdit] = []
        for cluster in clusters[:self._max_edits]:
            edit = self._draft_edit(cluster)
            if edit:
                edits.append(edit)
        return edits

    def _draft_edit(self, cluster: FailureCluster) -> HarnessEdit | None:
        """Create a concrete harness edit proposal from a failure cluster.

        In production this would call an LLM; here we implement a
        pattern-based heuristic as a starting scaffold.
        """
        dominant = cluster.pattern_label
        source = cluster.records[0].mechanism if cluster.records else ""

        if "context" in source.lower() or "memory" in source.lower():
            target = "context_manager"
        elif "tool" in source.lower():
            target = "tool_registry"
        elif "verifier" in source.lower() or "verification" in source.lower():
            target = "verifier_config"
        elif "workflow" in source.lower() or "loop" in source.lower():
            target = "workflow_engine"
        else:
            target = "harness_core"

        surface_code = self._surfaces.get(target, lambda: "")()
        old = surface_code[:200] if len(surface_code) > 200 else surface_code

        return HarnessEdit(
            description=f"Address pattern '{dominant}' ({len(cluster.records)} failures)",
            diff=f"--- a/{target}\n+++ b/{target}\n@@ ...\n-{old}\n+[proposed edit for {dominant}]",
            source_file=target,
            target_pattern=dominant,
        )

    async def _validate(self, candidate: HarnessCandidate) -> None:
        """Run held-in and held-out evaluation. Accept if no regression."""
        held_in_results: list[bool] = []
        for task in self._held_in:
            try:
                result = await task()
                held_in_results.extend(result.values())
            except Exception as e:
                logger.warning("held-in task failed: %s", e)

        candidate.held_in_pass_rate = (
            sum(held_in_results) / len(held_in_results)
            if held_in_results else 0.0
        )

        held_out_results: list[bool] = []
        for task in self._held_out:
            try:
                result = await task()
                held_out_results.extend(result.values())
            except Exception as e:
                logger.warning("held-out task failed: %s", e)

        candidate.held_out_pass_rate = (
            sum(held_out_results) / len(held_out_results)
            if held_out_results else 0.0
        )

        prev_pass_rate = 0.0
        if self._history:
            prev_pass_rate = self._history[-1].held_in_pass_rate

        improvement = candidate.held_in_pass_rate - prev_pass_rate
        regression = candidate.held_out_pass_rate < (
            self._history[-1].held_out_pass_rate if self._history else 1.0
        )

        candidate.accepted = (
            improvement >= self._threshold
            and not regression
        )

        for edit in candidate.edits:
            edit.accepted = candidate.accepted
            if not candidate.accepted:
                if regression:
                    edit.rejection_reason = "regression on held-out tasks"
                else:
                    edit.rejection_reason = f"held-in improvement {improvement:.2f} < threshold {self._threshold}"

    @property
    def version(self) -> str:
        return self._version

    @property
    def history(self) -> list[HarnessCandidate]:
        return self._history

    @property
    def rejected_edits(self) -> list[HarnessEdit]:
        return self._rejected_edits

    def stats(self) -> dict[str, Any]:
        accepted = [c for c in self._history if c.accepted]
        return {
            "version": self._version,
            "cycles": len(self._history),
            "accepted": len(accepted),
            "rejected_proposals": len(self._rejected_edits),
            "best_held_in": max((c.held_in_pass_rate for c in self._history), default=0.0),
            "best_held_out": max((c.held_out_pass_rate for c in self._history), default=0.0),
        }
