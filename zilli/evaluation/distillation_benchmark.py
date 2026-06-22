import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from zilli.training.champion_challenger import ChampionChallenger
from zilli.training.distillation import DistillationCycle, DistillationScheduler

logger = logging.getLogger("zilli.evaluation.benchmark")


@dataclass
class BenchmarkEntry:
    timestamp: float
    model_name: str
    phase: str
    loss: float
    kl: float
    exec_reward: float
    plan_reward: float
    sample_count: int
    wall_time_sec: float
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "model_name": self.model_name,
            "phase": self.phase,
            "loss": round(self.loss, 4),
            "kl": round(self.kl, 4),
            "exec_reward": round(self.exec_reward, 4),
            "plan_reward": round(self.plan_reward, 4),
            "sample_count": self.sample_count,
            "wall_time_sec": round(self.wall_time_sec, 2),
            "metadata": self.metadata,
        }


class BenchmarkTracker:
    def __init__(self, log_dir: str = ""):
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "arena_logs"
        self._entries: List[BenchmarkEntry] = []

    def record_before(self, scheduler: DistillationScheduler, model_name: str = "executor") -> BenchmarkEntry:
        stats = scheduler.stats()
        entry = BenchmarkEntry(
            timestamp=time.time(),
            model_name=model_name,
            phase="before_distill",
            loss=stats.get("recent_avg_kl", 0.0),
            kl=stats.get("recent_avg_kl", 0.0),
            exec_reward=0.0,
            plan_reward=0.0,
            sample_count=stats["total_samples"],
            wall_time_sec=0.0,
            metadata={"buffer_size": stats["buffer_size"]},
        )
        self._entries.append(entry)
        self._log(entry)
        return entry

    def record_after(self, cycle: DistillationCycle, model_name: str = "executor") -> BenchmarkEntry:
        entry = BenchmarkEntry(
            timestamp=time.time(),
            model_name=model_name,
            phase="after_distill",
            loss=cycle.total_loss,
            kl=cycle.kl_divergence,
            exec_reward=cycle.avg_executor_reward,
            plan_reward=cycle.avg_planner_reward,
            sample_count=cycle.samples,
            wall_time_sec=(cycle.end_time or time.time()) - cycle.start_time,
            metadata={"lora_triggered": cycle.lora_triggered},
        )
        self._entries.append(entry)
        self._log(entry)
        return entry

    def record_ab_result(self, variant_name: str, loss: float, kl: float,
                         sample_count: int, wall_time_sec: float,
                         metadata: Optional[Dict] = None) -> BenchmarkEntry:
        entry = BenchmarkEntry(
            timestamp=time.time(),
            model_name=variant_name,
            phase="ab_test",
            loss=loss,
            kl=kl,
            exec_reward=0.0,
            plan_reward=0.0,
            sample_count=sample_count,
            wall_time_sec=wall_time_sec,
            metadata=metadata or {},
        )
        self._entries.append(entry)
        self._log(entry)
        return entry

    def to_arena_match(self, champion: str, challenger: str,
                       before: BenchmarkEntry, after: BenchmarkEntry) -> dict:
        return {
            "match_id": f"{champion}_vs_{challenger}_{int(time.time())}",
            "timestamp": time.time(),
            "champion": champion,
            "challenger": challenger,
            "type": "distillation_benchmark",
            "before": before.to_dict(),
            "after": after.to_dict(),
            "loss_delta": round(after.loss - before.loss, 4),
            "kl_delta": round(after.kl - before.kl, 4),
        }

    def log_arena_match(self, match: dict):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        p = self.log_dir / "distill_benchmarks.jsonl"
        with open(p, "a") as f:
            f.write(json.dumps(match) + "\n")
        logger.info("Benchmark match logged to %s", p)

    def get_recent(self, limit: int = 10) -> List[dict]:
        return [e.to_dict() for e in self._entries[-limit:]]

    def _log(self, entry: BenchmarkEntry):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        p = self.log_dir / "benchmark_entries.jsonl"
        with open(p, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")


def run_benchmarked_distillation(
    scheduler: DistillationScheduler,
    tracker: BenchmarkTracker,
    model_name: str = "executor",
    arena: Optional[ChampionChallenger] = None,
    champion_name: str = "pre_distill",
    challenger_name: str = "post_distill",
) -> DistillationCycle:
    before = tracker.record_before(scheduler, model_name)

    cycle = scheduler.run_cycle()

    if cycle:
        after = tracker.record_after(cycle, model_name)
        match = tracker.to_arena_match(champion_name, challenger_name, before, after)
        tracker.log_arena_match(match)

        if arena:
            arena.add_score(champion_name, before.loss)
            arena.add_score(challenger_name, after.loss)
            arena.run_match(
                challenger_name,
                eval_fn=lambda m: _mock_scores(m, cycle),
            )

    return cycle


def _mock_scores(model_name: str, cycle: DistillationCycle) -> List[float]:
    if "post" in model_name:
        return [cycle.avg_planner_reward]
    return [cycle.avg_executor_reward]


__all__ = [
    "BenchmarkEntry", "BenchmarkTracker", "run_benchmarked_distillation",
]
