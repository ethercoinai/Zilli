from typing import Dict, List

import numpy as np

from zilli.training.distillation import DistillationSample


def make_dummy_golden(
    count: int = 10,
    base_reward: float = 0.9,
    seed: int = 42,
) -> List[Dict]:
    rng = np.random.default_rng(seed)
    trajectories = []
    for i in range(count):
        traj = [
            {
                "action": {"tool_name": "memory_write", "key": "x", "value": str(i)},
                "observation": {"success": True},
                "reward": 1.0,
                "done": False,
                "log_prob": float(rng.normal(-1.0, 0.5)),
                "old_log_prob": float(rng.normal(-0.5, 0.5)),
            },
            {
                "action": {"tool_name": "memory_read", "key": "x"},
                "observation": {"success": True, "value": str(i)},
                "reward": 1.0,
                "done": True,
                "log_prob": float(rng.normal(-1.0, 0.5)),
                "old_log_prob": float(rng.normal(-0.5, 0.5)),
            },
        ]
        trajectories.append({"trajectory": traj, "reward": base_reward + i * 0.01})
    return trajectories


def make_dummy_failure(
    count: int = 5,
    seed: int = 42,
) -> List[Dict]:
    rng = np.random.default_rng(seed + 1)
    trajectories = []
    for _ in range(count):
        traj = [
            {
                "action": {"tool_name": "bash_run", "command": "bad"},
                "observation": {"error": "fail", "success": False},
                "reward": -0.5,
                "done": True,
                "log_prob": float(rng.normal(-2.0, 0.5)),
                "old_log_prob": float(rng.normal(-1.0, 0.5)),
            },
        ]
        trajectories.append({"trajectory": traj, "reward": 0.1})
    return trajectories


def make_dummy_distillation_samples(
    count: int = 100,
    seed: int = 42,
) -> List[DistillationSample]:
    rng = np.random.default_rng(seed + 2)

    samples = []
    for idx in range(count):
        samples.append(DistillationSample(
            executor_action={"tool": "write", "key": "x", "value": str(idx)},
            planner_action={"tool": "write", "key": "y", "value": str(idx)},
            executor_log_prob=float(rng.normal(-1.0, 0.5)),
            planner_log_prob=float(rng.normal(-0.5, 0.3)),
            executor_reward=float(rng.uniform(0.3, 1.0)),
            planner_reward=float(rng.uniform(0.6, 1.0)),
            executor_embedding=list(rng.standard_normal(4)),
            planner_embedding=list(rng.standard_normal(4)),
        ))
    return samples


__all__ = ["make_dummy_golden", "make_dummy_failure", "make_dummy_distillation_samples"]
