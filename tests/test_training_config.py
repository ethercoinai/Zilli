import pytest
from pydantic import ValidationError
from zilli.training.config import TrainingConfig, RLTrainerConfig


class TestTrainingConfig:
    def test_defaults(self):
        cfg = TrainingConfig()
        assert cfg.algorithm == "CISPO"
        assert cfg.clip_range == 0.2
        assert cfg.kl_penalty == 0.01

    def test_rejects_invalid_algorithm(self):
        with pytest.raises(ValidationError):
            TrainingConfig(algorithm="INVALID")

    def test_rejects_negative_clip(self):
        with pytest.raises(ValidationError):
            TrainingConfig(clip_range=-0.1)

    def test_from_dict(self):
        cfg = TrainingConfig.from_dict({"clip_range": 0.3, "unknown_key": 99})
        assert cfg.clip_range == 0.3
        assert not hasattr(cfg, "unknown_key")

    def test_to_training_kwargs(self):
        cfg = TrainingConfig(algorithm="GRPO", clip_range=0.25)
        kwargs = cfg.to_training_kwargs()
        assert kwargs["clip_range"] == 0.25
        assert "algorithm" not in kwargs
        assert "batch_size" not in kwargs


class TestRLTrainerConfig:
    def test_defaults(self):
        cfg = RLTrainerConfig()
        assert cfg.training.algorithm == "CISPO"
        assert cfg.dry_run is False

    def test_rejects_extra(self):
        with pytest.raises(ValidationError):
            RLTrainerConfig(unknown_field=True)
