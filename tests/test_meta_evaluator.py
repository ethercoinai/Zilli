from zilli.evaluation.meta_evaluator import EvaluationSample, MetaEvaluator


class TestMetaEvaluator:
    def test_empty_history(self):
        me = MetaEvaluator()
        result = me.evaluate()
        assert result.sample_count == 0
        assert not result.reliable

    def test_record_and_evaluate(self):
        me = MetaEvaluator(window_size=50)
        for i in range(20):
            me.record(EvaluationSample(
                task_id=f"t{i}",
                features={"complexity": 0.5},
                predicted_score=0.8,
                actual_score=0.8 + 0.01 * i,
                model_name="test_model",
            ))
        result = me.evaluate()
        assert result.sample_count == 20
        assert result.calibration_error >= 0
        assert isinstance(result.confidence_interval, tuple)

    def test_bias_detection(self):
        me = MetaEvaluator()
        for i in range(30):
            me.record(EvaluationSample(
                task_id=f"t{i}",
                features={},
                predicted_score=0.9,
                actual_score=0.7,
            ))
        result = me.evaluate()
        assert result.bias > 0
        assert abs(result.bias - 0.2) < 0.01

    def test_correct_prediction(self):
        me = MetaEvaluator()
        for i in range(20):
            me.record(EvaluationSample(
                task_id=f"t{i}",
                features={},
                predicted_score=0.9,
                actual_score=0.7,
            ))
        me.evaluate()
        corrected = me.correct_prediction(0.9)
        assert abs(corrected - 0.7) < 0.05

    def test_no_drift_with稳定的数据(self):
        me = MetaEvaluator(window_size=10)
        for i in range(30):
            me.record(EvaluationSample(
                task_id=f"t{i}",
                features={},
                predicted_score=0.8,
                actual_score=0.8,
            ))
        assert not me.detect_drift()

    def test_drift_detection(self):
        me = MetaEvaluator(window_size=10)
        for i in range(15):
            me.record(EvaluationSample(
                task_id=f"t{i}",
                features={},
                predicted_score=0.8,
                actual_score=0.8,
            ))
        for i in range(15):
            me.record(EvaluationSample(
                task_id=f"t{i+15}",
                features={},
                predicted_score=0.8,
                actual_score=0.3,
            ))
        assert me.detect_drift(threshold=0.1)

    def test_feature_importance(self):
        me = MetaEvaluator()
        for i in range(30):
            me.record(EvaluationSample(
                task_id=f"t{i}",
                features={"difficulty": float(i) / 30},
                predicted_score=0.8,
                actual_score=0.5 + float(i) / 60,
            ))
        importance = me.feature_importance()
        assert "difficulty" in importance
        assert importance["difficulty"] > 0

    def test_summary(self):
        me = MetaEvaluator()
        for i in range(15):
            me.record(EvaluationSample(
                task_id=f"t{i}",
                features={"f": 0.5},
                predicted_score=0.8,
                actual_score=0.8,
            ))
        s = me.summary()
        assert "calibration_error" in s
        assert "drift_detected" in s
        assert "feature_importance" in s


__all__ = ["TestMetaEvaluator"]
