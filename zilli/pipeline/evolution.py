from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from zilli.adaptive.moo import CandidateSolution, MultiObjectiveOptimizer
from zilli.evaluation.meta_evaluator import EvaluationSample, MetaEvaluator
from zilli.fusion.engine import FusionStrategy, ResultFusion
from zilli.models.config import ModelRole
from zilli.models.registry import ModelRegistry

logger = logging.getLogger("zilli.pipeline.evolution")


class PipelineStage(str, Enum):
    MONITOR = "monitor"
    DETECT = "detect"
    PLAN = "plan"
    EVOLVE = "evolve"
    EVALUATE = "evaluate"
    DEPLOY = "deploy"
    ROLLBACK = "rollback"


@dataclass
class EvolutionEvent:
    stage: PipelineStage
    success: bool
    message: str
    metrics: dict[str, Any] = field(default_factory=dict)
    timestamp: float = 0.0


@dataclass
class PipelineConfig:
    monitor_interval_s: float = 3600.0
    degradation_threshold: float = 0.1
    min_samples_for_detection: int = 20
    auto_rollback: bool = True
    max_evolution_attempts: int = 3
    objective_names: list[str] = field(default_factory=lambda: ["quality", "cost", "latency"])


class EvolutionPipeline:
    def __init__(self, config: Optional[PipelineConfig] = None, registry: Optional[ModelRegistry] = None):
        self.config = config or PipelineConfig()
        self.meta_evaluator = MetaEvaluator()
        self.fusion = ResultFusion(default_strategy=FusionStrategy.CONFIDENCE_WEIGHTED)
        self.optimizer = MultiObjectiveOptimizer(
            objective_names=self.config.objective_names,
            objectives_to_minimize=["cost", "latency"],
        )
        self._registry = registry
        self._current_stage = PipelineStage.MONITOR
        self._events: list[EvolutionEvent] = []
        self._deployed_version: Optional[str] = None
        self._rollback_versions: list[str] = []
        self._degradation_detected = False

    @property
    def current_stage(self) -> PipelineStage:
        return self._current_stage

    @property
    def events(self) -> list[EvolutionEvent]:
        return list(self._events)

    def record_evaluation(self, sample: EvaluationSample) -> None:
        self.meta_evaluator.record(sample)

    def check_health(self) -> dict[str, Any]:
        meta_result = self.meta_evaluator.evaluate()
        drift = self.meta_evaluator.detect_drift()
        self._degradation_detected = drift or not meta_result.reliable
        return {
            "stage": self._current_stage.value,
            "reliable": meta_result.reliable,
            "drift_detected": drift,
            "calibration_error": meta_result.calibration_error,
            "degradation_detected": self._degradation_detected,
            "total_samples": meta_result.sample_count,
        }

    def run_cycle(self) -> list[EvolutionEvent]:
        events: list[EvolutionEvent] = []
        health = self.check_health()

        self._current_stage = PipelineStage.MONITOR
        events.append(EvolutionEvent(
            stage=PipelineStage.MONITOR, success=True,
            message="Health check completed", metrics=health,
        ))

        if self._degradation_detected:
            self._current_stage = PipelineStage.DETECT
            events.append(EvolutionEvent(
                stage=PipelineStage.DETECT, success=True,
                message="Degradation detected, triggering evolution",
                metrics={"drift": health["drift_detected"], "reliable": health["reliable"]},
            ))

            self._current_stage = PipelineStage.PLAN
            plan_event = self._plan_evolution()
            events.append(plan_event)

            if plan_event.success:
                self._current_stage = PipelineStage.EVOLVE
                evolve_event = self._execute_evolution()
                events.append(evolve_event)

                if evolve_event.success:
                    self._current_stage = PipelineStage.EVALUATE
                    eval_event = self._evaluate_evolution()
                    events.append(eval_event)

                    if eval_event.success:
                        self._current_stage = PipelineStage.DEPLOY
                        deploy_event = self._deploy()
                        events.append(deploy_event)
                    elif self.config.auto_rollback:
                        self._current_stage = PipelineStage.ROLLBACK
                        rollback_event = self._rollback()
                        events.append(rollback_event)
        else:
            events.append(EvolutionEvent(
                stage=PipelineStage.DETECT, success=True,
                message="No degradation detected, system healthy",
            ))

        self._events.extend(events)
        return events

    def _plan_evolution(self) -> EvolutionEvent:
        importance = self.meta_evaluator.feature_importance()
        top_features = list(importance.keys())[:3]
        return EvolutionEvent(
            stage=PipelineStage.PLAN, success=True,
            message=f"Evolution plan: top features {top_features}",
            metrics={"top_features": top_features},
        )

    def _execute_evolution(self) -> EvolutionEvent:
        import asyncio

        if self._registry:
            model = asyncio.run(self._registry.get_model_for_role(ModelRole.EXECUTOR))
            if model:
                gen = asyncio.run(model.generate(
                    "Analyze the current evaluation data and suggest one improvement: "
                    f"{self.meta_evaluator.summary()}"
                ))
                suggestion = gen.text or ""
                return EvolutionEvent(
                    stage=PipelineStage.EVOLVE, success=True,
                    message="Model suggested improvement",
                    metrics={"suggestion_len": len(suggestion), "source": "model_api"},
                )

        candidates = self.get_candidate_solutions(["current", "variant_a", "variant_b"])
        if len(candidates) < 2:
            return EvolutionEvent(
                stage=PipelineStage.EVOLVE, success=False,
                message="Insufficient candidates for evolution",
            )

        result = self.optimizer.optimize(candidates, max_generations=3)
        best = result.best_solution or candidates[0]
        return EvolutionEvent(
            stage=PipelineStage.EVOLVE, success=True,
            message=f"Evolved: selected {best.model_name}, pareto size={len(result.pareto_front.solutions)}",
            metrics={
                "selected": best.model_name,
                "pareto_size": len(result.pareto_front.solutions),
            },
        )

    def _evaluate_evolution(self) -> EvolutionEvent:
        meta_result = self.meta_evaluator.evaluate()
        return EvolutionEvent(
            stage=PipelineStage.EVALUATE,
            success=meta_result.reliable,
            message=f"Post-evolution eval: cal_error={meta_result.calibration_error}",
            metrics={"calibration_error": meta_result.calibration_error},
        )

    def _deploy(self) -> EvolutionEvent:
        version_num = 1
        if self._deployed_version and self._deployed_version.startswith("evolved_v"):
            self._rollback_versions.append(self._deployed_version)
            version_num = int(self._deployed_version.split("_v")[1]) + 1
        self._deployed_version = f"evolved_v{version_num}"
        return EvolutionEvent(
            stage=PipelineStage.DEPLOY, success=True,
            message=f"Deployed: {self._deployed_version}",
        )

    def _rollback(self) -> EvolutionEvent:
        if self._rollback_versions:
            prev = self._rollback_versions.pop()
            self._deployed_version = prev
            return EvolutionEvent(
                stage=PipelineStage.ROLLBACK, success=True,
                message=f"Rolled back to {prev}",
            )
        return EvolutionEvent(
            stage=PipelineStage.ROLLBACK, success=False,
            message="No rollback version available",
        )

    def get_candidate_solutions(self, models: list[str]) -> list[CandidateSolution]:
        evaluation = self.meta_evaluator.evaluate()
        samples = self.meta_evaluator._history
        candidates = []
        for model in models:
            model_errors = [abs(s.predicted_score - s.actual_score) for s in samples if s.model_name == model]
            q = float(sum(model_errors) / len(model_errors)) if model_errors else (evaluation.calibration_error or 0.1)
            candidates.append(CandidateSolution(
                model_name=model,
                objectives={
                    "quality": max(0.01, q),
                    "cost": 0.05 if model == "current" else 0.07,
                    "latency": 100.0 if model == "current" else 120.0,
                },
            ))
        return candidates

    def summary(self) -> dict[str, Any]:
        return {
            "current_stage": self._current_stage.value,
            "total_events": len(self._events),
            "deployed_version": self._deployed_version,
            "rollback_count": len(self._rollback_versions),
            "health": self.check_health(),
        }


__all__ = ["EvolutionPipeline", "PipelineConfig", "PipelineStage", "EvolutionEvent"]
