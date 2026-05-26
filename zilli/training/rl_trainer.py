from typing import Dict, List, Optional
from zilli.training.cispo import CISPO_Trainer
from zilli.training.grpo import GRPO_Trainer


class RLTrainer:
    def __init__(self, config: Dict):
        self.config = config
        algorithm = config.get("algorithm", "CISPO").upper()
        if algorithm == "CISPO":
            self.impl = CISPO_Trainer(config)
        elif algorithm == "GRPO":
            self.impl = GRPO_Trainer(config)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

    def update(self, batch: List[Dict]) -> Dict[str, float]:
        if self.config.get("algorithm", "").upper() == "GRPO":
            advantages = self.impl.compute_advantages(batch)
        else:
            rewards = [t.get("reward", 0.0) for t in batch]
            dones = [t.get("done", False) for t in batch]
            advantages = self.impl.compute_advantages(rewards, dones)

        return self.impl.compute_loss(batch, advantages)
