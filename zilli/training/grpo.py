from typing import Dict, List

import numpy as np


class GRPO_Trainer:  # noqa: N801
    def __init__(self, config: Dict):
        self.config = config
        self.clip_range = config.get("clip_range", 0.2)
        self.kl_penalty = config.get("kl_penalty", 0.01)

    def compute_advantages(self, group_trajectories: List[Dict]) -> List[float]:
        if not group_trajectories:
            return []
        rewards = np.array([t.get("reward", 0.0) for t in group_trajectories])
        baseline = rewards.mean()
        std = rewards.std() + 1e-8
        advantages = (rewards - baseline) / std
        return advantages.tolist()

    def compute_loss(self, trajectories: List[Dict], advantages: List[float]) -> Dict[str, float]:
        if not trajectories or not advantages:
            return {"loss": 0.0, "policy_loss": 0.0, "kl": 0.0}

        log_probs = np.array([t.get("log_prob", 0.0) for t in trajectories])
        old_log_probs = np.array([t.get("old_log_prob", 0.0) for t in trajectories])
        advantages = np.array(advantages)

        ratio = np.exp(np.clip(log_probs - old_log_probs, -5, 5))
        clipped = np.clip(ratio, 1 - self.clip_range, 1 + self.clip_range)

        surr1 = ratio * advantages
        surr2 = clipped * advantages
        loss = -np.minimum(surr1, surr2).mean()

        kl = (log_probs - old_log_probs).mean()
        total = loss + self.kl_penalty * kl

        return {
            "loss": float(total),
            "policy_loss": float(loss),
            "kl": float(kl),
        }
