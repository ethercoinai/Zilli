from typing import Dict, List, Optional

from zilli.training.cispo import CISPO_Trainer
from zilli.training.config import TrainingConfig
from zilli.training.grpo import GRPO_Trainer


class RLTrainer:
    def __init__(self, config: Optional[Dict] = None):
        self.training_config = TrainingConfig.from_dict(config or {})
        algorithm = self.training_config.algorithm
        if algorithm == "CISPO":
            self.impl = CISPO_Trainer(self.training_config.to_training_kwargs())
        elif algorithm == "GRPO":
            self.impl = GRPO_Trainer(self.training_config.to_training_kwargs())
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def update(self, batch: List[Dict]) -> Dict[str, float]:
        if self.training_config.algorithm == "GRPO":
            advantages = self.impl.compute_advantages(batch)
        else:
            rewards = [t.get("reward", 0.0) for t in batch]
            dones = [t.get("done", False) for t in batch]
            advantages = self.impl.compute_advantages(rewards, dones)

        return self.impl.compute_loss(batch, advantages)
