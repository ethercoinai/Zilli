from typing import Any, Dict, List

import numpy as np


class CISPO_Trainer:  # noqa: N801
    def __init__(self, config: Dict[str, Any]):
        self.clip_range = config.get("clip_range", 0.2)
        self.kl_penalty = config.get("kl_penalty", 0.01)
        self.is_weight_cap = config.get("is_weight_cap", 5.0)
        self.gamma = config.get("gamma", 0.99)
        self.gae_lambda = config.get("gae_lambda", 0.95)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        self.vf_coef = config.get("vf_coef", 0.5)
        self.config = config

    def compute_advantages(self, rewards: List[float], dones: List[bool]) -> List[float]:
        if not rewards:
            return []
        returns = []
        discounted_return = 0.0
        for t in reversed(range(len(rewards))):
            discounted_return = rewards[t] + self.gamma * discounted_return * (1 - int(dones[t]))
            returns.append(discounted_return)
        returns.reverse()
        return returns

    def compute_gae_advantages(self, rewards: List[float], values: List[float],
                                dones: List[bool]) -> List[float]:
        if not rewards or not values or not dones:
            return []
        if not (len(rewards) == len(values) == len(dones)):
            raise ValueError("rewards, values, dones must have equal length")
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

    def _compute_returns_target(self, advantages: np.ndarray, values: np.ndarray,
                                 trajectory_has_values: bool) -> np.ndarray:
        if trajectory_has_values:
            return values + advantages
        return advantages

    def compute_loss(self, trajectories: List[Dict], advantages: List[float]) -> Dict[str, float]:
        if not trajectories or not advantages:
            return {
                "loss": 0.0, "policy_loss": 0.0, "value_loss": 0.0,
                "kl": 0.0, "entropy": 0.0, "approx_kl": 0.0,
                "clip_frac": 0.0, "mean_advantage": 0.0, "std_advantage": 0.0,
            }

        log_probs = np.array([t.get("log_prob", 0.0) for t in trajectories])
        old_log_probs = np.array([t.get("old_log_prob", 0.0) for t in trajectories])
        advantages_arr = np.array(advantages)
        advantages_arr = self._normalize_advantages(advantages_arr)

        ratio = np.exp(np.clip(log_probs - old_log_probs, -5, 5))
        ratio = np.clip(ratio, 0, self.is_weight_cap)

        clipped_ratio = np.clip(ratio, 1 - self.clip_range, 1 + self.clip_range)

        surr1 = ratio * advantages_arr
        surr2 = clipped_ratio * advantages_arr
        policy_loss = -np.minimum(surr1, surr2).mean()

        kl = (log_probs - old_log_probs).mean()
        entropy = self._compute_entropy(log_probs)
        total_loss = policy_loss + self.kl_penalty * kl - self.entropy_coef * entropy

        trajectory_has_values = trajectories and all("value" in t for t in trajectories)
        value_loss = 0.0
        if trajectory_has_values:
            values = np.array([t.get("value", 0.0) for t in trajectories])
            returns_target = self._compute_returns_target(advantages_arr, values, True)
            value_loss = ((values - returns_target) ** 2).mean() * self.vf_coef
            total_loss = total_loss + value_loss

        clip_frac = float(np.mean(np.abs(ratio - 1) > self.clip_range))
        safe_ratio = np.clip(ratio, 1e-8, None)
        approx_kl = float(((safe_ratio - 1) - np.log(safe_ratio)).mean())

        return {
            "loss": float(total_loss),
            "policy_loss": float(policy_loss),
            "value_loss": float(value_loss),
            "kl": float(kl),
            "entropy": float(entropy),
            "approx_kl": approx_kl,
            "clip_frac": clip_frac,
            "mean_advantage": float(advantages_arr.mean()),
            "std_advantage": float(advantages_arr.std()),
        }

    def _normalize_advantages(self, advantages: np.ndarray) -> np.ndarray:
        if advantages.std() > 1e-8:
            return (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        return advantages

    def _compute_entropy(self, log_probs: np.ndarray) -> float:
        probs = np.exp(np.clip(log_probs, -10, 0))
        probs = np.clip(probs, 1e-10, None)
        entropy = -(probs * np.log(probs)).sum() / len(log_probs)
        return float(entropy)


__all__ = ["CISPO_Trainer"]
