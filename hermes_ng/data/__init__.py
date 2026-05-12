import random
from typing import List, Dict, Any, Optional
from collections import defaultdict


class TrajectoryStore:
    def __init__(self):
        self.golden_trajectories: List[Dict] = []
        self.failure_trajectories: List[Dict] = []
        self.rollout_buffer: List[Dict] = []

    def add_trajectory(self, trajectory: List[Dict], final_reward: float):
        entry = {
            "trajectory": trajectory,
            "reward": final_reward,
            "length": len(trajectory),
        }
        if final_reward > 0.8:
            entry["type"] = "golden"
            self.golden_trajectories.append(entry)
        elif final_reward < 0.3:
            entry["type"] = "failure"
            entry["error_summary"] = self._summarize_error(trajectory)
            self.failure_trajectories.append(entry)
        self.rollout_buffer.append(entry)

    def sample_batch(self, batch_size: int, golden_ratio: float = 0.5) -> List[Dict]:
        n_golden = int(batch_size * golden_ratio)
        n_failure = batch_size - n_golden

        sampled = []
        if self.golden_trajectories and n_golden > 0:
            sampled.extend(random.choices(self.golden_trajectories, k=min(n_golden, len(self.golden_trajectories))))
        if self.failure_trajectories and n_failure > 0:
            sampled.extend(random.choices(self.failure_trajectories, k=min(n_failure, len(self.failure_trajectories))))

        if len(sampled) < batch_size:
            extra = batch_size - len(sampled)
            pool = self.golden_trajectories + self.failure_trajectories
            if pool:
                sampled.extend(random.choices(pool, k=extra))

        random.shuffle(sampled)
        return sampled[:batch_size]

    def add_production_trajectories(self, trajectories: List[Dict]):
        for t in trajectories:
            self.rollout_buffer.append(t)

    def _summarize_error(self, trajectory: List[Dict]) -> str:
        errors = []
        for step in trajectory:
            obs = step.get("observation", {})
            if isinstance(obs, dict) and "error" in obs:
                errors.append(obs["error"])
        if errors:
            return "; ".join(errors[:3])
        return "Unknown failure"

    def purify(self) -> int:
        count = 0
        cleaned_golden = []
        for entry in self.golden_trajectories:
            if self._is_contaminated(entry["trajectory"]):
                count += 1
            else:
                cleaned_golden.append(entry)
        self.golden_trajectories = cleaned_golden

        cleaned_failure = []
        for entry in self.failure_trajectories:
            if not self._is_contaminated(entry["trajectory"]):
                cleaned_failure.append(entry)
            else:
                count += 1
        self.failure_trajectories = cleaned_failure
        return count

    def _is_contaminated(self, trajectory: List[Dict]) -> bool:
        for step in trajectory:
            obs = step.get("observation", {})
            if isinstance(obs, dict):
                err = obs.get("error", "")
                if "contaminated" in str(err).lower() or "corrupted" in str(err).lower():
                    return True
        return False

    def stats(self) -> Dict[str, Any]:
        return {
            "golden": len(self.golden_trajectories),
            "failure": len(self.failure_trajectories),
            "buffer": len(self.rollout_buffer),
            "total": len(self.golden_trajectories) + len(self.failure_trajectories) + len(self.rollout_buffer),
        }
