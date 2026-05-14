import asyncio
import logging
import json
from pathlib import Path
from typing import List, Dict
from zilli.data import TrajectoryStore


logger = logging.getLogger("zilli.learner")


class ContinuousLearner:
    def __init__(self, store: TrajectoryStore, interval_hours: int = 24, data_dir: str = ""):
        self.store = store
        self.interval = interval_hours
        self.data_dir = Path(data_dir) if data_dir else Path.cwd() / "production_data"
        self._running = False
        self._total_production_trajs = 0

    async def run(self):
        self._running = True
        logger.info("ContinuousLearner started, interval=%dh, data_dir=%s", self.interval, self.data_dir)
        while self._running:
            await asyncio.sleep(self.interval * 3600)
            new_trajectories = await self._collect_production_trajectories()
            for traj in new_trajectories:
                self.store.add_trajectory(traj.get("trajectory", []), traj.get("reward", 0.0))
            self._total_production_trajs += len(new_trajectories)
            logger.info("Collected %d production trajectories (total: %d)", len(new_trajectories), self._total_production_trajs)

            if len(self.store.rollout_buffer) > 1000:
                self._trigger_online_sft()

    async def stop(self):
        self._running = False
        logger.info("ContinuousLearner stopped")

    async def _collect_production_trajectories(self) -> List[Dict]:
        if not self.data_dir.exists():
            return []
        trajectories = []
        for f in sorted(self.data_dir.glob("*.json")):
            try:
                with open(f, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    if isinstance(data, list):
                        trajectories.extend(data)
                    else:
                        trajectories.append(data)
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to read %s: %s", f, e)
        return trajectories

    def _trigger_online_sft(self):
        logger.info(
            "Online SFT triggered: %d trajectories in buffer, %d golden, %d failure",
            len(self.store.rollout_buffer),
            len(self.store.golden_trajectories),
            len(self.store.failure_trajectories),
        )

    def stats(self) -> Dict:
        return {
            "running": self._running,
            "interval_hours": self.interval,
            "total_production_trajs": self._total_production_trajs,
            "data_dir": str(self.data_dir),
        }


__all__ = ["ContinuousLearner"]
