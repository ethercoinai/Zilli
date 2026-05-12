import math
from typing import List, Dict, Any
import numpy as np


class CISPO_Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.clip_range = config.get("clip_range", 0.2)
        self.kl_penalty = config.get("kl_penalty", 0.01)
        self.is_weight_cap = config.get("is_weight_cap", 5.0)
        self.gamma = config.get("gamma", 0.99)
        self.config = config

    def compute_loss(self, trajectories: List[Dict], advantages: List[float]) -> Dict[str, float]:
        log_probs = np.array([t.get("log_prob", 0.0) for t in trajectories])
        old_log_probs = np.array([t.get("old_log_prob", 0.0) for t in trajectories])
        advantages = np.array(advantages)

        ratio = np.exp(np.clip(log_probs - old_log_probs, -10, 10))
        ratio = np.clip(ratio, 0, self.is_weight_cap)

        clipped_ratio = np.clip(ratio, 1 - self.clip_range, 1 + self.clip_range)

        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        policy_loss = -np.minimum(surr1, surr2).mean()

        kl = (log_probs - old_log_probs).mean()
        total_loss = policy_loss + self.kl_penalty * kl

        approx_kl = ((ratio - 1) - np.log(ratio)).mean()

        return {
            "loss": float(total_loss),
            "policy_loss": float(policy_loss),
            "kl": float(kl),
            "approx_kl": float(approx_kl),
            "clip_frac": float(np.mean(np.abs(ratio - 1) > self.clip_range)),
        }

    def compute_advantages(self, rewards: List[float], dones: List[bool]) -> List[float]:
        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = 0.0
            else:
                next_val = 0.0
            delta = rewards[t] + self.gamma * next_val * (1 - int(dones[t]))
            gae = delta + self.gamma * gae
            advantages.insert(0, gae)
        return advantages
