import asyncio
import json
import logging
import shutil
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

from zilli.data import TrajectoryStore

logger = logging.getLogger("zilli.learner")


@dataclass
class LearningCycle:
    cycle_id: int
    start_time: float
    end_time: Optional[float] = None
    new_trajectories: int = 0
    total_trajectories: int = 0
    sft_triggered: bool = False
    sft_metrics: Optional[Dict] = None


class ContinuousLearner:
    def __init__(self, store: TrajectoryStore, interval_hours: int = 24,
                 data_dir: str = "", archive_dir: str = "",
                 sft_threshold: int = 1000,
                 sft_callback: Optional[Callable] = None):
        self.store = store
        self.interval = interval_hours
        self.data_dir = Path(data_dir) if data_dir else Path.cwd() / "production_data"
        self.archive_dir = Path(archive_dir) if archive_dir else self.data_dir / "archived"
        self.sft_threshold = sft_threshold
        self.sft_callback = sft_callback
        self._running = False
        self._total_production_trajs = 0
        self._cycle_count = 0
        self._cycles: List[LearningCycle] = []
        self._recent_errors: deque = deque(maxlen=100)

    async def run(self):
        self._running = True
        consecutive_errors = 0
        logger.info(
            "ContinuousLearner started, interval=%dh, data_dir=%s, sft_threshold=%d",
            self.interval, self.data_dir, self.sft_threshold,
        )
        while self._running:
            self._cycle_count += 1
            cycle = LearningCycle(
                cycle_id=self._cycle_count,
                start_time=time.time(),
            )

            try:
                new_trajectories, processed_files = await self._collect_production_trajectories()
                for traj in new_trajectories:
                    self.store.add_trajectory(traj.get("trajectory", []), traj.get("reward", 0.0))
                self._total_production_trajs += len(new_trajectories)
                cycle.new_trajectories = len(new_trajectories)
                cycle.total_trajectories = self._total_production_trajs

                if self._should_trigger_sft():
                    result = await self._trigger_online_sft()
                    cycle.sft_triggered = True
                    cycle.sft_metrics = result

                self._archive_processed_data(processed_files)
                consecutive_errors = 0
            except Exception as e:
                consecutive_errors += 1
                logger.error("Cycle %d failed: %s", self._cycle_count, e, exc_info=True)
                cycle.sft_metrics = {"error": str(e)}

            self._cycles.append(cycle)
            cycle.end_time = time.time()

            logger.info(
                "Cycle %d: collected %d trajectories (total: %d), sft=%s, elapsed=%.1fs",
                self._cycle_count, cycle.new_trajectories, self._total_production_trajs,
                cycle.sft_triggered, cycle.end_time - cycle.start_time,
            )

            if consecutive_errors > 0:
                await self._retry_backoff(consecutive_errors)
            else:
                await asyncio.sleep(self.interval * 3600)

    async def _retry_backoff(self, attempt: int) -> None:
        delay = min(60 * (2 ** attempt), 3600)
        await asyncio.sleep(delay)

    async def stop(self):
        self._running = False
        logger.info(
            "ContinuousLearner stopped after %d cycles, %d total trajectories",
            self._cycle_count, self._total_production_trajs,
        )

    async def _collect_production_trajectories(self) -> tuple[list[Dict], list[Path]]:
        if not self.data_dir.exists():
            self.data_dir.mkdir(parents=True, exist_ok=True)
            return [], []
        trajectories = []
        processed_files: list[Path] = []

        def _read_file(f: Path):
            with open(f, "r", encoding="utf-8") as fh:
                return json.load(fh)

        for f in sorted(self.data_dir.glob("*.json")):
            try:
                data = await asyncio.to_thread(_read_file, f)
                if isinstance(data, list):
                    trajectories.extend(data)
                else:
                    trajectories.append(data)
                processed_files.append(f)
                self._recent_errors.append(("ok", f.name))
            except (json.JSONDecodeError, IOError) as e:
                logger.warning("Failed to read %s: %s", f, e)
                self._recent_errors.append(("error", f.name))
        return trajectories, processed_files

    def _should_trigger_sft(self) -> bool:
        if not self.sft_callback:
            return False
        total = (len(self.store.golden_trajectories) +
                 len(self.store.failure_trajectories) +
                 len(self.store.rollout_buffer))
        return total >= self.sft_threshold

    async def _trigger_online_sft(self) -> Dict:
        store_stats = self.store.stats()
        metrics = {
            "timestamp": time.time(),
            "golden": store_stats["golden"],
            "failure": store_stats["failure"],
            "buffer": store_stats["buffer"],
            "total_production": self._total_production_trajs,
        }

        if self.sft_callback:
            try:
                result = self.sft_callback(store_stats)
                if hasattr(result, "__await__"):
                    result = await result
                if result:
                    metrics.update(result)
            except Exception as e:
                logger.error("SFT callback failed: %s", e)
                metrics["error"] = str(e)

        logger.info(
            "Online SFT triggered: golden=%d failure=%d buffer=%d total_prod=%d",
            store_stats["golden"], store_stats["failure"],
            store_stats["buffer"], self._total_production_trajs,
        )

        sft_log = self.data_dir / "sft_events.jsonl"
        with open(sft_log, "a") as f:
            f.write(json.dumps(metrics) + "\n")

        return metrics

    def _archive_processed_data(self, processed_files: list[Path]):
        if not processed_files:
            return
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        for f in processed_files:
            if not f.exists():
                continue
            dest = self.archive_dir / f.name
            try:
                shutil.move(str(f), str(dest))
            except OSError as e:
                logger.warning("Failed to archive %s: %s", f, e)

    def stats(self) -> Dict:
        return {
            "running": self._running,
            "interval_hours": self.interval,
            "total_production_trajs": self._total_production_trajs,
            "cycle_count": self._cycle_count,
            "data_dir": str(self.data_dir),
            "archive_dir": str(self.archive_dir),
            "sft_threshold": self.sft_threshold,
            "recent_errors": len([e for e in self._recent_errors if e[0] == "error"]),
            "recent_files": len(self._recent_errors),
        }


__all__ = ["ContinuousLearner"]
