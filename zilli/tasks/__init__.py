import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

from zilli.schema.actions import TaskConfig

logger = logging.getLogger("zilli.tasks")


def load_tasks(category: str = None) -> List[Dict[str, Any]]:
    """加载所有任务 YAML 文件，使用 TaskConfig schema 验证。"""
    base = Path(__file__).parent
    all_tasks = []

    dirs = [base / "basic", base / "benchmark"]
    for d in dirs:
        if d.is_dir():
            for f in sorted(d.glob("*.tasks.yaml")):
                if f.name.startswith(".") or f.name.endswith("~") or f.name.endswith(".swp"):
                    continue
                with open(f, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
                    if data:
                        for item in data:
                            if "id" in item:
                                try:
                                    validated = TaskConfig(**item)
                                    all_tasks.append(validated.model_dump())
                                except Exception as e:
                                    logger.warning(
                                        "Task '%s' in %s failed validation: %s",
                                        item.get("id", "?"), f, e,
                                    )

    if category:
        all_tasks = [t for t in all_tasks if t.get("category") == category]

    return all_tasks


def validate_task(task_dict: Dict[str, Any]) -> TaskConfig:
    """将任务 dict 验证为 TaskConfig Pydantic 模型。"""
    return TaskConfig(**task_dict)


def list_tasks_summary(tasks: List[Dict]) -> str:
    """生成任务摘要字符串。"""
    lines = []
    for t in tasks:
        tmpl = t.get("trajectory_template", [])
        reward_rules = t.get("reward_rules", [])
        lines.append(
            f"  [{t.get('category','?')}] {t['id']}: {t['name']} "
            f"(steps={t.get('max_steps','?')}, "
            f"tmpl_steps={len(tmpl)}, "
            f"reward_rules={len(reward_rules)})"
        )
    return "\n".join(lines)


class TaskRunner:
    def __init__(self, task: Dict[str, Any]):
        self.task = task
        self.step_count = 0
        self.max_steps = task.get("max_steps", 20)
        self.trajectory: List[Dict] = []
        self.completed = False
        self.final_reward = 0.0
        self.trajectory_template = task.get("trajectory_template", [])
        self.reward_rules = task.get("reward_rules", [])

    def record_action(self, action: Dict[str, Any], observation: Dict[str, Any]):
        self.step_count += 1
        self.trajectory.append({
            "step": self.step_count,
            "action": action,
            "observation": observation,
        })

    def should_truncate(self) -> bool:
        if self.step_count >= self.max_steps:
            return True
        if len(self.trajectory) >= 5:
            recent_tools = [t["action"].get("tool_name", "") for t in self.trajectory[-5:]]
            if len(set(recent_tools)) <= 1:
                return True
        return False

    def evaluate(self, final_state: Dict) -> float:
        criteria = self.task.get("eval_criteria", [])
        score = 0.0
        max_score = len(criteria) if criteria else 1

        for c in criteria:
            ctype = c.get("type", "")
            if ctype == "task_completed" and final_state.get("task_completed"):
                score += 1
            elif ctype == "memory_recall" and final_state.get("memory_recalled"):
                score += 1
            elif ctype == "skill_created" and final_state.get("skill_created"):
                score += 1
            elif ctype == "error_detected_in_round_1" and final_state.get("error_detected"):
                score += 1
            elif ctype == "corrected_in_round_2" and final_state.get("corrected"):
                score += 1
            elif ctype == "auto_truncate_on_loop" and final_state.get("truncated"):
                score += 1
            elif ctype == "reflection_generated" and final_state.get("reflection_done"):
                score += 1
            elif ctype == "build_success" and final_state.get("build_ok"):
                score += 1
            elif ctype == "test_passed" and final_state.get("tests_passed", 0) >= c.get("min_pass_rate", 0):
                score += 1
            elif final_state.get(f"{ctype}_ok"):
                score += 1

        return score / max_score if max_score > 0 else 0.0

    def evaluate_trajectory_template(self) -> float:
        if not self.trajectory_template:
            return 1.0
        matched = 0
        for i, tmpl_step in enumerate(self.trajectory_template):
            if i < len(self.trajectory):
                actual_tool = self.trajectory[i]["action"].get("tool_name", "")
                expected_tool = tmpl_step.get("tool", "")
                if actual_tool == expected_tool:
                    matched += tmpl_step.get("reward_weight", 1.0)
        total_weight = sum(s.get("reward_weight", 1.0) for s in self.trajectory_template)
        return matched / total_weight if total_weight > 0 else 0.0

    def evaluate_reward_rules(self, final_state: Dict) -> float:
        if not self.reward_rules:
            return 0.0
        total = 0.0
        for rule in self.reward_rules:
            rtype = rule.get("type", "")
            weight = rule.get("weight", 1.0)
            if rtype == "task_completion" and final_state.get("task_completed"):
                total += weight
            elif rtype == "format":
                valid_count = sum(
                    1 for t in self.trajectory
                    if isinstance(t.get("action"), dict) and "tool_name" in t["action"]
                )
                total += weight * (valid_count / max(len(self.trajectory), 1))
            elif rtype == "safety":
                has_forbidden = final_state.get("forbidden_action_executed", False)
                if not has_forbidden:
                    total += weight
            elif rtype == "efficiency":
                if self.step_count <= self.max_steps * 0.8:
                    total += weight
            elif rtype == "tool_accuracy":
                for t in self.trajectory:
                    obs = t.get("observation", {})
                    if isinstance(obs, dict) and not obs.get("success", True):
                        total -= weight * 0.5
        return total


__all__ = ["load_tasks", "TaskRunner", "validate_task", "list_tasks_summary"]
