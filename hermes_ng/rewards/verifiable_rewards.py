from typing import List, Dict, Any
from hermes_ng.schema.actions import BaseAction


class VerifiableReward:
    def compute(self, trajectory: List[BaseAction], final_state: Dict) -> float:
        reward = 0.0

        for action in trajectory:
            if self._validate_action_schema(action):
                reward += 0.1

        if final_state.get("task_completed"):
            reward += 1.0

        if final_state.get("forbidden_action_executed"):
            reward -= 2.0

        return max(-2.0, min(2.0, reward))

    def compute_trajectory(self, trajectory: List[Dict], final_state: Dict) -> float:
        reward = 0.0

        for step in trajectory:
            obs = step.get("observation", {})
            if isinstance(obs, dict) and obs.get("success"):
                reward += 0.1

        if final_state.get("task_completed"):
            reward += 1.0

        errors = sum(
            1 for t in trajectory
            if isinstance(t.get("observation", {}), dict)
            and "error" in t["observation"]
        )
        reward -= 0.25 * errors

        return max(-2.0, min(2.0, reward))

    def _validate_action_schema(self, action: BaseAction) -> bool:
        try:
            _ = action.model_dump()
            return True
        except Exception:
            return False


__all__ = ["VerifiableReward"]
