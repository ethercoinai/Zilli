import random
import math
from typing import List, Dict, Any, Optional
from collections import defaultdict

from zilli.data.trajectory_cleaner import TrajectoryCleaner


class TrajectoryStore:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.golden_trajectories: List[Dict] = []
        self.failure_trajectories: List[Dict] = []
        self.rollout_buffer: List[Dict] = []
        self._priorities: List[float] = []
        self.cleaner = TrajectoryCleaner(self.config.get("cleaner_config"))
        self.max_size = self.config.get("max_size", 10000)
        self.min_reward_for_golden = self.config.get("min_reward_for_golden", 0.8)
        self.max_reward_for_failure = self.config.get("max_reward_for_failure", 0.3)

    def add_trajectory(self, trajectory: List[Dict], final_reward: float):
        if len(trajectory) < 2:
            return

        entry = {
            "trajectory": trajectory,
            "reward": final_reward,
            "length": len(trajectory),
        }

        if final_reward >= self.min_reward_for_golden:
            entry["type"] = "golden"
            self.golden_trajectories.append(entry)
            self._priorities.append(final_reward)
        elif final_reward <= self.max_reward_for_failure:
            entry["type"] = "failure"
            entry["error_summary"] = self._summarize_error(trajectory)
            self.failure_trajectories.append(entry)
            self._priorities.append(1.0 - final_reward)
        else:
            entry["type"] = "neutral"
            self.rollout_buffer.append(entry)
            self._priorities.append(0.5)

        self._enforce_max_size()

    def sample_batch(self, batch_size: int, golden_ratio: float = 0.5,
                     use_priority: bool = False) -> List[Dict]:
        n_golden = int(batch_size * golden_ratio)
        n_failure = batch_size - n_golden

        sampled = []

        if use_priority and self._priorities:
            sampled = self._priority_sample(batch_size)
        else:
            if self.golden_trajectories and n_golden > 0:
                sampled.extend(random.choices(
                    self.golden_trajectories,
                    k=min(n_golden, len(self.golden_trajectories)),
                ))
            if self.failure_trajectories and n_failure > 0:
                sampled.extend(random.choices(
                    self.failure_trajectories,
                    k=min(n_failure, len(self.failure_trajectories)),
                ))

        if len(sampled) < batch_size:
            extra = batch_size - len(sampled)
            pool = self.golden_trajectories + self.failure_trajectories
            if pool:
                sampled.extend(random.choices(pool, k=extra))

        random.shuffle(sampled)
        return sampled[:batch_size]

    def augment_batch(self, batch: List[Dict]) -> List[Dict]:
        augmented = []
        for entry in batch:
            augmented.append(entry)
            traj = entry.get("trajectory", [])
            if len(traj) >= 4:
                noisy = self._add_noise(traj, noise_level=0.05)
                augmented.append({**entry, "trajectory": noisy, "augmented": True})
        return augmented

    def add_production_trajectories(self, trajectories: List[Dict]):
        for t in trajectories:
            self.rollout_buffer.append(t)

    def purify(self) -> int:
        count = 0

        cleaned_golden = []
        for entry in self.golden_trajectories:
            cleaned, warnings = self.cleaner.clean(entry["trajectory"])
            if not cleaned:
                count += 1
            else:
                cleaned_golden.append({**entry, "trajectory": cleaned})
        self.golden_trajectories = cleaned_golden

        cleaned_failure = []
        for entry in self.failure_trajectories:
            cleaned, warnings = self.cleaner.clean(entry["trajectory"])
            if not cleaned:
                count += 1
            else:
                cleaned_failure.append({**entry, "trajectory": cleaned})
        self.failure_trajectories = cleaned_failure

        cleaned_buffer = []
        for entry in self.rollout_buffer:
            cleaned, warnings = self.cleaner.clean(entry["trajectory"])
            if cleaned:
                cleaned_buffer.append({**entry, "trajectory": cleaned})
        self.rollout_buffer = cleaned_buffer

        return count

    def stats(self) -> Dict[str, Any]:
        return {
            "golden": len(self.golden_trajectories),
            "failure": len(self.failure_trajectories),
            "buffer": len(self.rollout_buffer),
            "total": (len(self.golden_trajectories) +
                      len(self.failure_trajectories) +
                      len(self.rollout_buffer)),
            "avg_golden_reward": self._avg_reward(self.golden_trajectories),
            "avg_failure_reward": self._avg_reward(self.failure_trajectories),
            "avg_trajectory_length": self._avg_length(),
        }

    def _summarize_error(self, trajectory: List[Dict]) -> str:
        errors = []
        for step in trajectory:
            obs = step.get("observation", {})
            if isinstance(obs, dict) and "error" in obs:
                errors.append(obs["error"])
        if errors:
            return "; ".join(errors[:3])
        return "Unknown failure"

    def _trim_priorities(self):
        total = (len(self.golden_trajectories) +
                 len(self.failure_trajectories) +
                 len(self.rollout_buffer))
        if len(self._priorities) > total:
            self._priorities = self._priorities[-total:]

    def _enforce_max_size(self):
        total = (len(self.golden_trajectories) +
                 len(self.failure_trajectories) +
                 len(self.rollout_buffer))
        if total > self.max_size:
            excess = total - self.max_size
            if self.rollout_buffer:
                self.rollout_buffer = self.rollout_buffer[excess:]
            elif self.failure_trajectories:
                self.failure_trajectories = self.failure_trajectories[excess:]
            elif self.golden_trajectories:
                self.golden_trajectories = self.golden_trajectories[excess:]
            self._trim_priorities()

    def _priority_sample(self, batch_size: int) -> List[Dict]:
        all_entries = self.golden_trajectories + self.failure_trajectories
        if not all_entries:
            return []

        priorities = self._priorities[:len(all_entries)]
        total_p = sum(priorities)
        if total_p == 0:
            return random.choices(all_entries, k=min(batch_size, len(all_entries)))

        weights = [p / total_p for p in priorities]
        return random.choices(all_entries, weights=weights, k=min(batch_size, len(all_entries)))

    @staticmethod
    def _add_noise(trajectory: List[Dict], noise_level: float = 0.05) -> List[Dict]:
        noisy = []
        for step in trajectory:
            step_copy = {
                "step": step.get("step"),
                "action": dict(step.get("action", {})),
                "observation": dict(step.get("observation", {})),
            }
            if random.random() < noise_level:
                obs = step_copy["observation"]
                if "reward" in obs:
                    obs["reward"] = obs["reward"] * (1 + random.uniform(-0.1, 0.1))
            noisy.append(step_copy)
        return noisy

    @staticmethod
    def _avg_reward(trajs: List[Dict]) -> float:
        if not trajs:
            return 0.0
        return sum(t.get("reward", 0.0) for t in trajs) / len(trajs)

    def _avg_length(self) -> float:
        all_trajs = self.golden_trajectories + self.failure_trajectories + self.rollout_buffer
        if not all_trajs:
            return 0.0
        return sum(t.get("length", 0) for t in all_trajs) / len(all_trajs)


__all__ = ["TrajectoryStore"]
