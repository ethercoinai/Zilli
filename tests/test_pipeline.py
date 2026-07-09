from zilli.evaluation.meta_evaluator import EvaluationSample
from zilli.pipeline.evolution import (
    EvolutionPipeline,
    PipelineStage,
)


class TestEvolutionPipeline:
    def test_init(self):
        p = EvolutionPipeline()
        assert p.current_stage == PipelineStage.MONITOR

    def test_record_evaluation(self):
        p = EvolutionPipeline()
        p.record_evaluation(EvaluationSample(
            task_id="t1", features={}, predicted_score=0.8, actual_score=0.8,
        ))
        assert len(p.meta_evaluator._history) == 1

    def test_check_health(self):
        p = EvolutionPipeline()
        for i in range(25):
            p.record_evaluation(EvaluationSample(
                task_id=f"t{i}", features={}, predicted_score=0.8, actual_score=0.8,
            ))
        health = p.check_health()
        assert "reliable" in health
        assert "drift_detected" in health

    def test_run_cycle_healthy(self):
        p = EvolutionPipeline()
        for i in range(25):
            p.record_evaluation(EvaluationSample(
                task_id=f"t{i}", features={}, predicted_score=0.8, actual_score=0.8,
            ))
        events = p.run_cycle()
        assert len(events) >= 2
        assert events[0].stage == PipelineStage.MONITOR

    def test_run_cycle_degradation(self):
        p = EvolutionPipeline()
        for i in range(15):
            p.record_evaluation(EvaluationSample(
                task_id=f"t{i}", features={}, predicted_score=0.8, actual_score=0.8,
            ))
        for i in range(15):
            p.record_evaluation(EvaluationSample(
                task_id=f"t{i+15}", features={}, predicted_score=0.8, actual_score=0.2,
            ))
        events = p.run_cycle()
        stages = [e.stage for e in events]
        assert PipelineStage.DETECT in stages

    def test_summary(self):
        p = EvolutionPipeline()
        s = p.summary()
        assert "current_stage" in s
        assert "health" in s

    def test_get_candidate_solutions(self):
        p = EvolutionPipeline()
        candidates = p.get_candidate_solutions(["m1", "m2", "m3"])
        assert len(candidates) == 3
        assert candidates[0].model_name == "m1"


__all__ = ["TestEvolutionPipeline"]
