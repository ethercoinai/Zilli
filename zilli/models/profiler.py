"""Model capability profiling with ELO scoring and radar chart data.

PRD §4.3 requirement: maintain multi-dimensional capability radar chart
for each model (reasoning/code/math/creativity/instruction-following/safety),
ELO score with dynamic decay.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class CapabilityScores:
    reasoning: float = 0.5
    code: float = 0.5
    math: float = 0.5
    creativity: float = 0.5
    instruction_following: float = 0.5
    safety: float = 0.5

    def to_vector(self) -> list[float]:
        return [self.reasoning, self.code, self.math,
                self.creativity, self.instruction_following, self.safety]

    @property
    def labels(self) -> list[str]:
        return ["reasoning", "code", "math", "creativity", "instruction", "safety"]


@dataclass
class ELOEntry:
    rating: float = 1200.0
    games: int = 0
    last_updated: float = 0.0
    volatility: float = 1.0


_DEFAULT_K = 32
_DECAY_PER_DAY = 0.997
_INITIAL_ELO = 1200.0


class ModelProfiler:
    """Maintains ELO ratings and capability scores for each model.

    ELO is updated via pairwise comparisons. Capability scores are
    updated via task-specific evaluations. Both decay over time.
    """

    def __init__(self):
        self._elo: dict[str, ELOEntry] = {}
        self._capabilities: dict[str, CapabilityScores] = {}
        self._task_results: dict[str, list[dict]] = {}

    def get_elo(self, model_name: str) -> float:
        entry = self._elo.get(model_name)
        if entry is None:
            return _INITIAL_ELO
        days_since = (time.time() - entry.last_updated) / 86400
        decayed = entry.rating * (_DECAY_PER_DAY ** days_since)
        return max(decayed, 100.0)

    def record_comparison(self, winner: str, loser: str) -> None:
        elo_a = self.get_elo(winner)
        elo_b = self.get_elo(loser)
        expected_a = 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))
        k_a = _DEFAULT_K / max(1, self._elo.get(winner, ELOEntry()).games)
        k_b = _DEFAULT_K / max(1, self._elo.get(loser, ELOEntry()).games)

        new_a = elo_a + k_a * (1.0 - expected_a)
        new_b = elo_b + k_b * (0.0 - (1.0 - expected_a))

        self._elo[winner] = ELOEntry(
            rating=round(new_a, 1),
            games=self._elo.get(winner, ELOEntry()).games + 1,
            last_updated=time.time(),
        )
        self._elo[loser] = ELOEntry(
            rating=round(new_b, 1),
            games=self._elo.get(loser, ELOEntry()).games + 1,
            last_updated=time.time(),
        )

    def record_task_result(self, model_name: str, task_type: str,
                            score: float, max_score: float = 1.0) -> None:
        if model_name not in self._task_results:
            self._task_results[model_name] = []
        self._task_results[model_name].append({
            "task_type": task_type,
            "score": score,
            "max_score": max_score,
            "timestamp": time.time(),
        })

        cap = self._capabilities.setdefault(model_name, CapabilityScores())
        normalized = score / max_score if max_score > 0 else 0.0
        if task_type in cap.__dataclass_fields__:
            old = getattr(cap, task_type)
            setattr(cap, task_type, old * 0.7 + normalized * 0.3)

    def get_capabilities(self, model_name: str) -> CapabilityScores:
        return self._capabilities.get(model_name, CapabilityScores())

    def get_radar_data(self, model_name: str) -> dict[str, float]:
        cap = self.get_capabilities(model_name)
        return {
            "reasoning": round(cap.reasoning, 3),
            "code": round(cap.code, 3),
            "math": round(cap.math, 3),
            "creativity": round(cap.creativity, 3),
            "instruction": round(cap.instruction_following, 3),
            "safety": round(cap.safety, 3),
        }

    def leaderboard(self, top_n: int = 10) -> list[dict[str, Any]]:
        ratings = [(name, self.get_elo(name)) for name in self._elo]
        ratings.sort(key=lambda x: x[1], reverse=True)
        result = []
        for name, elo in ratings[:top_n]:
            cap = self.get_radar_data(name)
            result.append({
                "name": name,
                "elo": round(elo, 1),
                "games": self._elo.get(name, ELOEntry()).games,
                "capabilities": cap,
            })
        return result

    def stats(self) -> dict[str, Any]:
        return {
            "models_tracked": len(self._elo),
            "total_comparisons": sum(e.games for e in self._elo.values()),
            "leaderboard": self.leaderboard(),
        }
