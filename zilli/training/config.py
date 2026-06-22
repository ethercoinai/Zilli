from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field


class TrainingConfig(BaseModel):
    algorithm: str = Field(default="CISPO", pattern=r"^(CISPO|GRPO)$")
    clip_range: float = Field(default=0.2, ge=0.0, le=1.0)
    kl_penalty: float = Field(default=0.01, ge=0.0)
    is_weight_cap: float = Field(default=5.0, ge=1.0)
    gamma: float = Field(default=0.99, ge=0.0, le=1.0)
    gae_lambda: float = Field(default=0.95, ge=0.0, le=1.0)
    entropy_coef: float = Field(default=0.01, ge=0.0)
    vf_coef: float = Field(default=0.5, ge=0.0)
    batch_size: int = Field(default=64, ge=1)
    learning_rate: float = Field(default=3e-4, gt=0.0)

    model_config = ConfigDict(extra="forbid")

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> TrainingConfig:
        return cls(**{k: v for k, v in d.items() if k in cls.model_fields})

    def to_training_kwargs(self) -> Dict[str, Any]:
        return self.model_dump(exclude={"algorithm", "batch_size", "learning_rate"})


class RLTrainerConfig(BaseModel):
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    cost_aware: bool = False
    monthly_budget: Optional[float] = None
    dry_run: bool = False

    model_config = ConfigDict(extra="forbid")


__all__ = ["TrainingConfig", "RLTrainerConfig"]
