from zilli.fusion.engine import FusionStrategy, ModelOutput, ResultFusion


class TestResultFusion:
    def test_init(self):
        f = ResultFusion()
        assert f.default_strategy == FusionStrategy.CONFIDENCE_WEIGHTED

    def test_confidence_weighted_numeric(self):
        f = ResultFusion()
        outputs = [
            ModelOutput(model_name="m1", text="0.8", confidence=0.9),
            ModelOutput(model_name="m2", text="0.6", confidence=0.5),
        ]
        result = f.fuse(outputs)
        assert result.strategy == FusionStrategy.CONFIDENCE_WEIGHTED
        assert len(result.contributing_models) == 2
        assert float(result.fused_text) > 0.6

    def test_confidence_weighted_text(self):
        f = ResultFusion()
        outputs = [
            ModelOutput(model_name="m1", text="hello", confidence=0.9),
            ModelOutput(model_name="m2", text="world", confidence=0.5),
        ]
        result = f.fuse(outputs)
        assert result.fused_text == "hello"

    def test_majority_vote(self):
        f = ResultFusion()
        outputs = [
            ModelOutput(model_name="m1", text="yes", confidence=0.8),
            ModelOutput(model_name="m2", text="yes", confidence=0.7),
            ModelOutput(model_name="m3", text="no", confidence=0.6),
        ]
        result = f.fuse(outputs, strategy=FusionStrategy.MAJORITY_VOTE)
        assert result.fused_text == "yes"

    def test_average_numeric(self):
        f = ResultFusion()
        outputs = [
            ModelOutput(model_name="m1", text="1.0", confidence=0.8),
            ModelOutput(model_name="m2", text="3.0", confidence=0.7),
        ]
        result = f.fuse(outputs, strategy=FusionStrategy.AVERAGE)
        assert float(result.fused_text) == 2.0

    def test_best_confidence(self):
        f = ResultFusion()
        outputs = [
            ModelOutput(model_name="m1", text="alpha", confidence=0.3),
            ModelOutput(model_name="m2", text="beta", confidence=0.9),
        ]
        result = f.fuse(outputs, strategy=FusionStrategy.BEST_CONFIDENCE)
        assert result.fused_text == "beta"

    def test_stacking_numeric(self):
        f = ResultFusion()
        outputs = [
            ModelOutput(model_name="m1", text="2.0", confidence=0.8),
            ModelOutput(model_name="m2", text="4.0", confidence=0.6),
        ]
        result = f.fuse(outputs, strategy=FusionStrategy.STACKING)
        assert float(result.fused_text) > 2.0

    def test_single_output(self):
        f = ResultFusion(min_models=1)
        outputs = [ModelOutput(model_name="m1", text="only", confidence=0.9)]
        result = f.fuse(outputs)
        assert result.fused_text == "only"

    def test_empty_output(self):
        f = ResultFusion()
        result = f.fuse([])
        assert result.fused_text == ""

    def test_disagreement(self):
        f = ResultFusion()
        outputs = [
            ModelOutput(model_name="m1", text="yes", confidence=0.8),
            ModelOutput(model_name="m2", text="no", confidence=0.7),
        ]
        result = f.fuse(outputs)
        assert result.disagreement_score > 0

    def test_should_arbitrate(self):
        f = ResultFusion(disagreement_threshold=0.3)
        f.set_judge_model("judge")
        outputs = [
            ModelOutput(model_name="m1", text="yes", confidence=0.8),
            ModelOutput(model_name="m2", text="no", confidence=0.7),
        ]
        result = f.fuse(outputs)
        assert f.should_arbitrate(result)

    def test_summary(self):
        f = ResultFusion()
        outputs = [ModelOutput(model_name="m1", text="x", confidence=0.8)]
        f.fuse(outputs)
        s = f.summary()
        assert s["total_fusions"] == 1
        assert "avg_confidence" in s


__all__ = ["TestResultFusion"]
