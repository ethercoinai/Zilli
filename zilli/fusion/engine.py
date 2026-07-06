from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

import numpy as np

logger = logging.getLogger("zilli.fusion.engine")


class FusionStrategy(str, Enum):
    CONFIDENCE_WEIGHTED = "confidence_weighted"
    MAJORITY_VOTE = "majority_vote"
    AVERAGE = "average"
    BEST_CONFIDENCE = "best_confidence"
    STACKING = "stacking"


@dataclass
class ModelOutput:
    model_name: str
    text: str
    confidence: float = 0.5
    latency_ms: float = 0.0
    cost: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FusionResult:
    fused_text: str
    confidence: float
    strategy: FusionStrategy
    contributing_models: list[str]
    disagreement_score: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)


class ResultFusion:
    def __init__(
        self,
        default_strategy: FusionStrategy = FusionStrategy.CONFIDENCE_WEIGHTED,
        disagreement_threshold: float = 0.5,
        min_models: int = 1,
    ):
        self.default_strategy = default_strategy
        self.disagreement_threshold = disagreement_threshold
        self.min_models = min_models
        self._judge_model: Optional[str] = None
        self._history: list[FusionResult] = []

    def set_judge_model(self, model_name: str):
        self._judge_model = model_name

    def fuse(
        self,
        outputs: list[ModelOutput],
        strategy: Optional[FusionStrategy] = None,
        task_type: str = "classification",
    ) -> FusionResult:
        strat = strategy or self.default_strategy

        if len(outputs) < self.min_models:
            if outputs:
                return FusionResult(
                    fused_text=outputs[0].text,
                    confidence=outputs[0].confidence,
                    strategy=strat,
                    contributing_models=[outputs[0].model_name],
                )
            return FusionResult(fused_text="", confidence=0.0, strategy=strat, contributing_models=[])

        if strat == FusionStrategy.CONFIDENCE_WEIGHTED:
            result = self._confidence_weighted(outputs)
        elif strat == FusionStrategy.MAJORITY_VOTE:
            result = self._majority_vote(outputs)
        elif strat == FusionStrategy.AVERAGE:
            result = self._average(outputs)
        elif strat == FusionStrategy.BEST_CONFIDENCE:
            result = self._best_confidence(outputs)
        elif strat == FusionStrategy.STACKING:
            result = self._stacking(outputs)
        else:
            result = self._confidence_weighted(outputs)

        result.disagreement_score = self._compute_disagreement(outputs)
        if result.disagreement_score > self.disagreement_threshold and self._judge_model:
            result.details["high_disagreement"] = True
            result.details["suggested_arbiter"] = self._judge_model

        self._history.append(result)
        return result

    def _confidence_weighted(self, outputs: list[ModelOutput]) -> FusionResult:
        total_conf = sum(o.confidence for o in outputs)
        if total_conf == 0:
            return self._average(outputs)

        weights = [o.confidence / total_conf for o in outputs]
        texts = [o.text for o in outputs]
        numeric_values = []
        all_numeric = True
        for t in texts:
            try:
                numeric_values.append(float(t.strip()))
            except (ValueError, AttributeError):
                all_numeric = False
                break

        if all_numeric and numeric_values:
            fused_val = sum(w * v for w, v in zip(weights, numeric_values))
            return FusionResult(
                fused_text=str(round(fused_val, 4)),
                confidence=total_conf / len(outputs),
                strategy=FusionStrategy.CONFIDENCE_WEIGHTED,
                contributing_models=[o.model_name for o in outputs],
                details={"weighted_sum": fused_val, "weights": weights},
            )

        best_idx = int(np.argmax(weights))
        return FusionResult(
            fused_text=outputs[best_idx].text,
            confidence=outputs[best_idx].confidence,
            strategy=FusionStrategy.CONFIDENCE_WEIGHTED,
            contributing_models=[o.model_name for o in outputs],
            details={"selected_index": best_idx, "weights": weights},
        )

    def _majority_vote(self, outputs: list[ModelOutput]) -> FusionResult:
        votes: dict[str, float] = {}
        for o in outputs:
            key = o.text.strip().lower()
            votes[key] = votes.get(key, 0) + o.confidence

        winner = max(votes, key=votes.get)
        winner_models = [o.model_name for o in outputs if o.text.strip().lower() == winner]

        return FusionResult(
            fused_text=winner,
            confidence=votes[winner] / sum(votes.values()),
            strategy=FusionStrategy.MAJORITY_VOTE,
            contributing_models=winner_models,
            details={"vote_counts": votes},
        )

    def _average(self, outputs: list[ModelOutput]) -> FusionResult:
        texts = [o.text for o in outputs]
        numeric_values = []
        all_numeric = True
        for t in texts:
            try:
                numeric_values.append(float(t.strip()))
            except (ValueError, AttributeError):
                all_numeric = False
                break

        if all_numeric and numeric_values:
            avg = float(np.mean(numeric_values))
            return FusionResult(
                fused_text=str(round(avg, 4)),
                confidence=float(np.mean([o.confidence for o in outputs])),
                strategy=FusionStrategy.AVERAGE,
                contributing_models=[o.model_name for o in outputs],
                details={"mean": avg, "std": float(np.std(numeric_values))},
            )

        best_idx = int(np.argmax([o.confidence for o in outputs]))
        return FusionResult(
            fused_text=outputs[best_idx].text,
            confidence=outputs[best_idx].confidence,
            strategy=FusionStrategy.AVERAGE,
            contributing_models=[o.model_name for o in outputs],
        )

    def _best_confidence(self, outputs: list[ModelOutput]) -> FusionResult:
        best = max(outputs, key=lambda o: o.confidence)
        return FusionResult(
            fused_text=best.text,
            confidence=best.confidence,
            strategy=FusionStrategy.BEST_CONFIDENCE,
            contributing_models=[best.model_name],
            details={"all_confidences": {o.model_name: o.confidence for o in outputs}},
        )

    def _stacking(self, outputs: list[ModelOutput]) -> FusionResult:
        confs = np.array([o.confidence for o in outputs])
        softmax = np.exp(confs) / np.exp(confs).sum()
        texts = [o.text for o in outputs]
        numeric_values = []
        all_numeric = True
        for t in texts:
            try:
                numeric_values.append(float(t.strip()))
            except (ValueError, AttributeError):
                all_numeric = False
                break

        if all_numeric and numeric_values:
            vals = np.array(numeric_values)
            fused = float(np.dot(softmax, vals))
            return FusionResult(
                fused_text=str(round(fused, 4)),
                confidence=float(softmax.max()),
                strategy=FusionStrategy.STACKING,
                contributing_models=[o.model_name for o in outputs],
                details={"softmax_weights": softmax.tolist(), "fused_value": fused},
            )

        best_idx = int(np.argmax(softmax))
        return FusionResult(
            fused_text=outputs[best_idx].text,
            confidence=float(softmax[best_idx]),
            strategy=FusionStrategy.STACKING,
            contributing_models=[outputs[best_idx].model_name],
        )

    def _compute_disagreement(self, outputs: list[ModelOutput]) -> float:
        if len(outputs) < 2:
            return 0.0
        texts = [o.text.strip().lower() for o in outputs]
        unique = set(texts)
        if len(unique) == 1:
            return 0.0
        return 1.0 - (1.0 / len(unique))

    def should_arbitrate(self, result: FusionResult) -> bool:
        return result.disagreement_score > self.disagreement_threshold and self._judge_model is not None

    def summary(self) -> dict[str, Any]:
        if not self._history:
            return {"total_fusions": 0}
        strategies = [r.strategy.value for r in self._history]
        return {
            "total_fusions": len(self._history),
            "avg_disagreement": float(np.mean([r.disagreement_score for r in self._history])),
            "strategy_usage": {s: strategies.count(s) for s in set(strategies)},
            "avg_confidence": float(np.mean([r.confidence for r in self._history])),
        }


__all__ = ["FusionStrategy", "ModelOutput", "FusionResult", "ResultFusion"]
