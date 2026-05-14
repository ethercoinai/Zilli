import os
import yaml
from typing import List, Dict, Any
from pathlib import Path


_TASK_CACHE: Dict[str, List[Dict[str, Any]]] = {}


def load_tasks(category: str = None) -> List[Dict[str, Any]]:
    base = Path(__file__).parent
    all_tasks = []

    dirs = [base / "basic", base / "benchmark"]
    for d in dirs:
        if d.is_dir():
            for f in sorted(d.glob("*.tasks.yaml")):
                with open(f, "r", encoding="utf-8") as fh:
                    data = yaml.safe_load(fh)
                    if data:
                        all_tasks.extend(data)

    if category:
        all_tasks = [t for t in all_tasks if t.get("category") == category]

    _TASK_CACHE["all"] = all_tasks
    return all_tasks


class TaskRunner:
    def __init__(self, task: Dict[str, Any]):
        self.task = task
        self.step_count = 0
        self.max_steps = task.get("max_steps", 20)
        self.trajectory: List[Dict] = []
        self.completed = False
        self.final_reward = 0.0

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
            recent_actions = [t["action"].get("tool_name") for t in self.trajectory[-5:]]
            if len(set(recent_actions)) <= 1:
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
