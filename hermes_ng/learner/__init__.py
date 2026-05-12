import asyncio
import logging
from typing import List, Dict
from hermes_ng.data import TrajectoryStore


class ContinuousLearner:
    def __init__(self, store: TrajectoryStore, interval_hours: int = 24):
        self.store = store
        self.interval = interval_hours
        self._running = False

    async def run(self):
        self._running = True
        while self._running:
            await asyncio.sleep(self.interval * 3600)
            new_trajectories = await self._collect_production_trajectories()
            for traj in new_trajectories:
                self.store.add_trajectory(traj.get("trajectory", []), traj.get("reward", 0.0))

            if len(self.store.rollout_buffer) > 1000:
                self._trigger_online_sft()

    async def stop(self):
        self._running = False

    async def _collect_production_trajectories(self) -> List[Dict]:
        return []

    def _trigger_online_sft(self):
        pass
