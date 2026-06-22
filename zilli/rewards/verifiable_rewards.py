from typing import Any, Dict, List, Optional

from zilli.schema.actions import BaseAction


class VerifiableReward:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.schema_reward = self.config.get("schema_reward", 0.1)
        self.task_completion_reward = self.config.get("task_completion_reward", 1.0)
        self.forbidden_penalty = self.config.get("forbidden_penalty", 2.0)
        self.error_penalty = self.config.get("error_penalty", 0.25)
        self.template_match_reward = self.config.get("template_match_reward", 0.5)
        self.efficiency_bonus = self.config.get("efficiency_bonus", 0.3)

    def compute(self, trajectory: List[BaseAction], final_state: Dict) -> float:
        reward = 0.0

        for action in trajectory:
            if self._validate_action_schema(action):
                reward += self.schema_reward

        if final_state.get("task_completed"):
            reward += self.task_completion_reward

        if final_state.get("forbidden_action_executed"):
            reward -= self.forbidden_penalty

        return max(-self.forbidden_penalty, min(self.task_completion_reward * 2, reward))

    def compute_trajectory(self, trajectory: List[Dict], final_state: Dict) -> float:
        reward = 0.0

        for step in trajectory:
            obs = step.get("observation", {})
            if isinstance(obs, dict) and obs.get("success"):
                reward += self.schema_reward

        if final_state.get("task_completed"):
            reward += self.task_completion_reward

        errors = sum(
            1 for t in trajectory
            if isinstance(t.get("observation", {}), dict)
            and t["observation"].get("success") is not True
        )
        reward -= self.error_penalty * errors

        template_score = final_state.get("template_match_score", 0.0)
        reward += self.template_match_reward * template_score

        efficiency = final_state.get("efficiency", 0.0)
        reward += self.efficiency_bonus * efficiency

        if final_state.get("forbidden_action_executed"):
            reward -= self.forbidden_penalty

        return max(-self.forbidden_penalty, min(self.task_completion_reward * 3, reward))

    def compute_template_match(self, trajectory: List[Dict],
                                template: List[Dict]) -> float:
        if not template:
            return 1.0
        matched = 0.0
        for i, tmpl_step in enumerate(template):
            weight = tmpl_step.get("reward_weight", 1.0)
            if i < len(trajectory):
                actual_tool = trajectory[i].get("action", {}).get("tool_name", "")
                expected_tool = tmpl_step.get("tool", "")
                if actual_tool == expected_tool:
                    matched += weight
        total_weight = sum(s.get("reward_weight", 1.0) for s in template)
        return matched / total_weight if total_weight > 0 else 0.0

    def compute_efficiency(self, trajectory: List[Dict], max_steps: int) -> float:
        if max_steps <= 0:
            return 0.0
        steps_used = len(trajectory)
        ratio = steps_used / max_steps
        if ratio <= 0.3:
            return 1.0
        elif ratio <= 0.6:
            return 0.6
        elif ratio <= 0.8:
            return 0.3
        else:
            return 0.0

    def compute_diversity(self, trajectory: List[Dict]) -> float:
        tools = set()
        for step in trajectory:
            tool = step.get("action", {}).get("tool_name", "")
            if tool:
                tools.add(tool)
        if len(tools) >= 4:
            return 1.0
        elif len(tools) >= 2:
            return 0.5
        return 0.0

    def _validate_action_schema(self, action: BaseAction) -> bool:
        try:
            _ = action.model_validate(action.model_dump())
            return True
        except Exception:
            return False


__all__ = ["VerifiableReward"]
