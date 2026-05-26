from typing import List, Dict, Any, Optional
import numpy as np


class CISPO_Trainer:
    def __init__(self, config: Dict[str, Any]):
        self.clip_range = config.get("clip_range", 0.2)
        self.kl_penalty = config.get("kl_penalty", 0.01)
        self.is_weight_cap = config.get("is_weight_cap", 5.0)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.vf_coef = config.get("vf_coef", 0.5)
        self.config = config

    def compute_loss(self, trajectories: List[Dict], advantages: List[float]) -> Dict[str, float]:
        log_probs = np.array([t.get("log_prob", 0.0) for t in trajectories])
        old_log_probs = np.array([t.get("old_log_prob", 0.0) for t in trajectories])
        advantages = np.array(advantages)
        advantages = self._normalize_advantages(advantages)

        ratio = np.exp(np.clip(log_probs - old_log_probs, -10, 10))
        ratio = np.clip(ratio, 0, self.is_weight_cap)

        clipped_ratio = np.clip(ratio, 1 - self.clip_range, 1 + self.clip_range)

        surr1 = ratio * advantages
        surr2 = clipped_ratio * advantages
        policy_loss = -np.minimum(surr1, surr2).mean()

        kl = (log_probs - old_log_probs).mean()
        entropy = self._compute_entropy(log_probs)
        total_loss = policy_loss + self.kl_penalty * kl - self.entropy_coef * entropy

        value_loss = 0.0
        if "value" in trajectories[0] if trajectories else False:
            values = np.array([t.get("value", 0.0) for t in trajectories])
            returns = self._compute_returns(advantages, values)
            value_loss = ((values - returns) ** 2).mean() * self.vf_coef
            total_loss = total_loss + value_loss

        clip_frac = float(np.mean(np.abs(ratio - 1) > self.clip_range))
        approx_kl = float(((ratio - 1) - np.log(ratio)).mean())

        return {
            "loss": float(total_loss),
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "kl": float(kl),
            "entropy": float(entropy),
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "mean_advantage": float(advantages.mean()),
            "std_advantage": float(advantages.std()),
        }

    def compute_advantages(self, rewards: List[float], dones: List[bool]) -> List[float]:
        returns = []
        discounted_return = 0.0
        for t in reversed(range(len(rewards))):
            discounted_return = rewards[t] + self.gamma * discounted_return * (1 - int(dones[t]))
            returns.insert(0, discounted_return)
        return returns

    def compute_gae_advantages(self, rewards: List[float], values: List[float],
                                dones: List[bool]) -> List[float]:
        advantages = []
        gae = 0.0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = 0.0
            else:
                next_val = values[t + 1]
            delta = rewards[t] + self.gamma * next_val * (1 - int(dones[t])) - values[t]
            gae = delta + self.gamma * self.gae_lambda * (1 - int(dones[t])) * gae
            advantages.insert(0, gae)
        return advantages

    def _normalize_advantages(self, advantages: np.ndarray) -> np.ndarray:
        if advantages.std() > 1e-8:
            return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def _compute_entropy(self, log_probs: np.ndarray) -> float:
        probs = np.exp(np.clip(log_probs, -10, 0))
        entropy = -(probs * log_probs).sum() / len(log_probs)
        return float(entropy)

    def _compute_returns(self, advantages: np.ndarray, values: np.ndarray) -> np.ndarray:
        return advantages + values


__all__ = ["CISPO_Trainer"]
