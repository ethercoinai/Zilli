import numpy as np
from typing import Dict, Any


class DualModelDistillationLoss:
    def __init__(self, lambda_bc=1.0, lambda_rl=0.5, lambda_reg=0.1,
                 beta=0.1, gamma=0.2, delta=0.5):
        self.lambda_bc = lambda_bc
        self.lambda_rl = lambda_rl
        self.lambda_reg = lambda_reg
        self.beta = beta
        self.gamma = gamma
        self.delta = delta

    def compute_bc_loss(self, executor_action_probs: np.ndarray,
                        planner_action_probs: np.ndarray,
                        planner_action_ids: np.ndarray) -> float:
        if isinstance(planner_action_ids, np.ndarray) and planner_action_ids.ndim == 1:
            planner_action_id = int(np.argmax(planner_action_ids))
        else:
            planner_action_id = int(planner_action_ids)

        cross_entropy = -np.log(executor_action_probs[planner_action_id] + 1e-8)
        kl = np.sum(planner_action_probs * (
            np.log(planner_action_probs + 1e-8) - np.log(executor_action_probs + 1e-8)
        ))
        return float(cross_entropy + self.beta * kl)

    def compute_rl_loss(self, executor_reward: float, planner_reward: float) -> float:
        pg = -executor_reward
        shape = self.gamma * (executor_reward - planner_reward) ** 2
        return pg + shape

    def compute_regularization_loss(self, executor_action_embed: np.ndarray,
                                    planner_action_embed: np.ndarray) -> float:
        distance = float(np.linalg.norm(executor_action_embed - planner_action_embed))
        return max(0.0, distance - self.delta)

    def total_loss(self, step: Dict[str, Any]) -> float:
        bc = self.compute_bc_loss(
            step["executor_probs"],
            step["planner_probs"],
            step["planner_action_id"],
        )
        rl = self.compute_rl_loss(
            step["executor_reward"],
            step["planner_reward"],
        )
        reg = self.compute_regularization_loss(
            step["executor_embed"],
            step["planner_embed"],
        )
        return self.lambda_bc * bc + self.lambda_rl * rl + self.lambda_reg * reg


__all__ = ["DualModelDistillationLoss"]
