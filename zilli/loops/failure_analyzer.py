"""Verifier-grounded failure records and weakness mining.

Based on Self-Harness (Zhang et al. 2026):
A failure record must contain three levels of information:
  (a) terminal verifier-level cause (e.g., timeout, missing artifact)
  (b) causal status of the relevant agent behavior
  (c) abstract agent mechanism exposed by the trace
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("zilli.loops.failure_analyzer")


@dataclass
class FailureRecord:
    """Rich failure record with verifier-grounded information.

    Attributes
    ----------
    task_id : str
        Identifier of the task that failed.
    verifier_outcome : str
        What the gate reported (timeout, wrong_answer, crash, missing_artifact, etc.).
    causal_status : str
        What the agent did leading to the failure (e.g., wrong_tool, skipped_step).
    mechanism : str
        Which harness component failed (context, tool, workflow, permission, verifier).
    trace_excerpt : str
        Key excerpts from the execution trajectory.
    timestamp : float
        When the failure occurred.
    metadata : dict
        Additional context (model, iteration, cost, etc.).
    """

    task_id: str
    verifier_outcome: str
    causal_status: str
    mechanism: str
    trace_excerpt: str = ""
    timestamp: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FailureCluster:
    """A cluster of failures sharing the same root cause pattern."""

    pattern_label: str
    mechanism: str
    records: list[FailureRecord]
    count: int = 0
    verifier_outcomes: set[str] = field(default_factory=set)

    def __post_init__(self):
        self.count = len(self.records)
        self.verifier_outcomes = {r.verifier_outcome for r in self.records}


class WeaknessMiner:
    """Mines and clusters failures from execution traces.

    Clusters by (mechanism, causal_status) tuple to group failures
    by root cause rather than surface-level verifier outcome.
    Two runs can share the same verifier outcome (e.g., timeout)
    while having different causal mechanisms.
    """

    def __init__(self, min_cluster_size: int = 2):
        self._min_size = min_cluster_size
        self._all_records: list[FailureRecord] = []

    def ingest(self, traces: list[dict]) -> None:
        """Convert raw execution traces to structured FailureRecords."""
        for t in traces:
            record = FailureRecord(
                task_id=t.get("task_id", "unknown"),
                verifier_outcome=t.get("verifier_outcome", "unknown"),
                causal_status=t.get("causal_status", "unknown"),
                mechanism=t.get("mechanism", "unknown"),
                trace_excerpt=t.get("trace", "")[:500],
                timestamp=t.get("timestamp", 0.0),
                metadata=t.get("metadata", {}),
            )
            self._all_records.append(record)

    def cluster_failures(self, traces: list[dict] | None = None) -> list[FailureCluster]:
        """Cluster failures by (mechanism, causal_status)."""
        if traces is not None:
            self.ingest(traces)

        clusters: dict[str, list[FailureRecord]] = defaultdict(list)
        for record in self._all_records:
            key = f"{record.mechanism}::{record.causal_status}"
            clusters[key].append(record)

        result: list[FailureCluster] = []
        for key, records in clusters.items():
            if len(records) < self._min_size:
                continue
            mechanism, causal = key.split("::", 1)
            result.append(FailureCluster(
                pattern_label=f"{mechanism}: {causal}",
                mechanism=mechanism,
                records=records,
            ))

        result.sort(key=lambda c: c.count, reverse=True)
        return result

    def summary(self) -> dict[str, Any]:
        clusters = self.cluster_failures()
        return {
            "total_records": len(self._all_records),
            "clusters": [
                {
                    "pattern": c.pattern_label,
                    "count": c.count,
                    "outcomes": list(c.verifier_outcomes),
                }
                for c in clusters
            ],
            "top_pattern": clusters[0].pattern_label if clusters else None,
        }

    @property
    def total_records(self) -> int:
        return len(self._all_records)

    def clear(self) -> None:
        self._all_records.clear()
