import json
import logging
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from zilli.infra.device_utils import get_device, is_gpu_available

logger = logging.getLogger("zilli.distillation")


@dataclass
class DistillationSample:
    executor_action: Dict
    planner_action: Dict
    executor_log_prob: float
    planner_log_prob: float
    executor_reward: float
    planner_reward: float
    state_embedding: Optional[List[float]] = None
    executor_embedding: Optional[List[float]] = None
    planner_embedding: Optional[List[float]] = None


@dataclass
class DistillationCycle:
    cycle_id: int
    start_time: float
    end_time: Optional[float] = None
    samples: int = 0
    bc_loss: float = 0.0
    rl_loss: float = 0.0
    reg_loss: float = 0.0
    total_loss: float = 0.0
    kl_divergence: float = 0.0
    avg_executor_reward: float = 0.0
    avg_planner_reward: float = 0.0
    lora_triggered: bool = False
    metrics: Dict = field(default_factory=dict)


class DistillationScheduler:
    def __init__(
        self,
        lambda_bc: float = 1.0,
        lambda_rl: float = 0.5,
        lambda_reg: float = 0.1,
        kl_beta: float = 0.1,
        reward_gamma: float = 0.2,
        embedding_delta: float = 0.5,
        lora_threshold: int = 1000,
        distill_interval_hours: int = 24,
        full_sft_interval_days: int = 7,
        log_dir: str = "",
        lora_callback: Optional[Callable] = None,
        full_sft_callback: Optional[Callable] = None,
    ):
        self.lambda_bc = lambda_bc
        self.lambda_rl = lambda_rl
        self.lambda_reg = lambda_reg
        self.kl_beta = kl_beta
        self.reward_gamma = reward_gamma
        self.embedding_delta = embedding_delta
        self.lora_threshold = lora_threshold
        self.distill_interval = distill_interval_hours
        self.full_sft_interval = full_sft_interval_days * 24
        self.lora_callback = lora_callback
        self.full_sft_callback = full_sft_callback
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "distill_logs"

        self._buffer: List[DistillationSample] = []
        self._cycles: List[DistillationCycle] = []
        self._total_samples: int = 0
        self._lora_events: int = 0
        self._full_sft_events: int = 0
        self._last_lora_time: float = 0.0
        self._last_full_sft_time: float = 0.0
        self._recent_kl: deque = deque(maxlen=100)

    def add_sample(self, sample: DistillationSample):
        self._buffer.append(sample)
        self._total_samples += 1

    def add_batch(self, samples: List[DistillationSample]):
        self._buffer.extend(samples)
        self._total_samples += len(samples)

    def compute_bc_loss(self, executor_log_probs: List[float],
                        planner_log_probs: List[float]) -> float:
        bc = 0.0
        kl = 0.0
        n = len(executor_log_probs)
        if n == 0:
            return 0.0, 0.0
        for i in range(n):
            ep = math.exp(min(max(executor_log_probs[i], -10), 0))
            pp = math.exp(min(max(planner_log_probs[i], -10), 0))
            ep = max(min(ep, 1.0 - 1e-10), 1e-10)
            pp = max(min(pp, 1.0 - 1e-10), 1e-10)
            bc += -math.log(ep) * pp
            kl += pp * (math.log(pp) - math.log(ep))
        bc /= n
        kl = max(0.0, kl / n)
        return bc + self.kl_beta * kl, kl

    def compute_rl_loss(self, executor_rewards: List[float],
                        planner_rewards: List[float]) -> float:
        n = len(executor_rewards)
        if n == 0:
            return 0.0
        loss = 0.0
        for i in range(n):
            r_e = executor_rewards[i]
            r_p = planner_rewards[i]
            loss += -r_e + self.reward_gamma * (r_e - r_p) ** 2
        return loss / n

    def compute_reg_loss(self, samples: List[DistillationSample]) -> float:
        loss = 0.0
        count = 0
        for s in samples:
            if s.executor_embedding is not None and s.planner_embedding is not None:
                dist = math.sqrt(
                    sum((a - b) ** 2 for a, b in
                        zip(s.executor_embedding, s.planner_embedding))
                )
                loss += max(0.0, dist - self.embedding_delta)
                count += 1
        return (loss / count) if count > 0 else 0.0

    def run_cycle(self) -> Optional[DistillationCycle]:
        if not self._buffer:
            return None

        now = time.time()
        cycle = DistillationCycle(
            cycle_id=len(self._cycles) + 1,
            start_time=now,
            samples=len(self._buffer),
        )

        executor_probs = []
        planner_probs = []
        executor_rewards = []
        planner_rewards = []

        for s in self._buffer:
            executor_probs.append(s.executor_log_prob)
            planner_probs.append(s.planner_log_prob)
            executor_rewards.append(s.executor_reward)
            planner_rewards.append(s.planner_reward)

        gpu_result = self.compute_loss_torch(self._buffer)
        if gpu_result:
            bc_loss = gpu_result["bc_loss"]
            rl_loss = gpu_result["rl_loss"]
            reg_loss = gpu_result["reg_loss"]
            total_loss = gpu_result["total"]
            kl = gpu_result["kl"]
        else:
            bc_loss, kl = self.compute_bc_loss(executor_probs, planner_probs)
            rl_loss = self.compute_rl_loss(executor_rewards, planner_rewards)
            reg_loss = self.compute_reg_loss(self._buffer)
            total_loss = (
                self.lambda_bc * bc_loss
                + self.lambda_rl * rl_loss
                + self.lambda_reg * reg_loss
            )

        avg_exec_reward = sum(executor_rewards) / len(executor_rewards) if executor_rewards else 0.0
        avg_plan_reward = sum(planner_rewards) / len(planner_rewards) if planner_rewards else 0.0

        self._recent_kl.append(kl)

        cycle.bc_loss = bc_loss
        cycle.rl_loss = rl_loss
        cycle.reg_loss = reg_loss
        cycle.total_loss = total_loss
        cycle.kl_divergence = kl
        cycle.avg_executor_reward = avg_exec_reward
        cycle.avg_planner_reward = avg_plan_reward

        needs_lora = (
            self._total_samples >= self.lora_threshold
            and (now - self._last_lora_time) >= self.distill_interval * 3600
        )
        needs_full = (now - self._last_full_sft_time) >= self.full_sft_interval * 3600

        if needs_full and self.full_sft_callback:
            logger.info("Triggering full SFT/DPO training cycle")
            try:
                result = self.full_sft_callback(self._buffer)
                cycle.metrics["full_sft_result"] = result
            except Exception as e:
                logger.error("Full SFT callback failed: %s", e)
                cycle.metrics["full_sft_error"] = str(e)
            self._last_full_sft_time = now
            self._full_sft_events += 1

        if needs_lora and self.lora_callback:
            logger.info("Triggering LoRA distillation (samples=%d, kl=%.4f)",
                        self._total_samples, kl)
            try:
                result = self.lora_callback(self._buffer)
                cycle.metrics["lora_result"] = result
                cycle.lora_triggered = True
            except Exception as e:
                logger.error("LoRA callback failed: %s", e)
                cycle.metrics["lora_error"] = str(e)
            self._last_lora_time = now
            self._lora_events += 1
        elif needs_lora:
            logger.info("LoRA threshold reached but no callback registered")

        cycle.end_time = time.time()
        self._log_cycle(cycle)
        self._cycles.append(cycle)
        self._buffer.clear()

        logger.info(
            "Distill cycle %d: loss=%.4f bc=%.4f rl=%.4f reg=%.4f kl=%.4f "
            "exec_reward=%.3f plan_reward=%.3f lora=%s samples=%d",
            cycle.cycle_id, total_loss, bc_loss, rl_loss, reg_loss, kl,
            avg_exec_reward, avg_plan_reward, cycle.lora_triggered, cycle.samples,
        )

        return cycle

    def should_distill(self) -> bool:
        if not self._buffer:
            return False
        if len(self._buffer) >= self.lora_threshold:
            return True
        if self._cycles and self._cycles[-1].end_time is not None:
            elapsed = time.time() - self._cycles[-1].end_time
            return elapsed >= self.distill_interval * 3600
        return False

    def stats(self) -> Dict:
        return {
            "total_samples": self._total_samples,
            "buffer_size": len(self._buffer),
            "cycles_completed": len(self._cycles),
            "lora_events": self._lora_events,
            "full_sft_events": self._full_sft_events,
            "last_lora_hours_ago": (
                (time.time() - self._last_lora_time) / 3600
                if self._last_lora_time > 0 else None
            ),
            "recent_avg_kl": (
                sum(self._recent_kl) / len(self._recent_kl)
                if self._recent_kl else 0.0
            ),
            "lambda_bc": self.lambda_bc,
            "lambda_rl": self.lambda_rl,
            "lambda_reg": self.lambda_reg,
            "kl_beta": self.kl_beta,
        }

    def compute_loss_torch(self, samples: List[DistillationSample]) -> Optional[Dict[str, float]]:
        if not is_gpu_available():
            return None
        try:
            import torch
            device = get_device()
            exec_log_probs = torch.tensor(
                [s.executor_log_prob for s in samples], device=device,
            )
            plan_log_probs = torch.tensor(
                [s.planner_log_prob for s in samples], device=device,
            )
            exec_probs = torch.exp(torch.clamp(exec_log_probs, max=0))
            plan_probs = torch.exp(torch.clamp(plan_log_probs, max=0))
            exec_probs = torch.clamp(exec_probs, min=1e-10, max=1 - 1e-10)
            plan_probs = torch.clamp(plan_probs, min=1e-10, max=1 - 1e-10)
            exec_rew = torch.tensor([s.executor_reward for s in samples], device=device)
            plan_rew = torch.tensor([s.planner_reward for s in samples], device=device)

            bc = -(plan_probs * torch.log(exec_probs)).mean()
            kl = (plan_probs * (torch.log(plan_probs) - torch.log(exec_probs))).mean()
            bc_loss = bc + self.kl_beta * kl

            rl_loss = (-exec_rew + self.reward_gamma * (exec_rew - plan_rew) ** 2).mean()

            reg = torch.tensor(0.0, device=device)
            count = 0
            for s in samples:
                if s.executor_embedding is not None and s.planner_embedding is not None:
                    ee = torch.tensor(s.executor_embedding, device=device)
                    pe = torch.tensor(s.planner_embedding, device=device)
                    d = torch.norm(ee - pe)
                    reg = reg + torch.clamp(d - self.embedding_delta, min=0)
                    count += 1
            reg_loss = (reg / count) if count > 0 else torch.tensor(0.0, device=device)

            total = (
                self.lambda_bc * bc_loss
                + self.lambda_rl * rl_loss
                + self.lambda_reg * reg_loss
            )

            return {
                "bc_loss": bc_loss.item(),
                "rl_loss": rl_loss.item(),
                "reg_loss": reg_loss.item(),
                "total": total.item(),
                "kl": kl.item(),
            }
        except ImportError:
            logger.debug("torch not available for GPU compute")
            return None

    def save_checkpoint(self, path: str = "") -> str:
        if not path:
            path = str(self.log_dir / "checkpoint.json")
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "scheduler_params": {
                "lambda_bc": self.lambda_bc,
                "lambda_rl": self.lambda_rl,
                "lambda_reg": self.lambda_reg,
                "kl_beta": self.kl_beta,
                "reward_gamma": self.reward_gamma,
                "embedding_delta": self.embedding_delta,
                "lora_threshold": self.lora_threshold,
                "distill_interval_hours": self.distill_interval,
                "full_sft_interval_days": self.full_sft_interval // 24,
                "log_dir": str(self.log_dir),
            },
            "state": {
                "buffer": [
                    {
                        "executor_action": s.executor_action,
                        "planner_action": s.planner_action,
                        "executor_log_prob": s.executor_log_prob,
                        "planner_log_prob": s.planner_log_prob,
                        "executor_reward": s.executor_reward,
                        "planner_reward": s.planner_reward,
                        "state_embedding": s.state_embedding,
                        "executor_embedding": s.executor_embedding,
                        "planner_embedding": s.planner_embedding,
                    }
                    for s in self._buffer
                ],
                "cycles": [
                    {
                        "cycle_id": c.cycle_id,
                        "start_time": c.start_time,
                        "end_time": c.end_time,
                        "samples": c.samples,
                        "bc_loss": c.bc_loss,
                        "rl_loss": c.rl_loss,
                        "reg_loss": c.reg_loss,
                        "total_loss": c.total_loss,
                        "kl_divergence": c.kl_divergence,
                        "avg_executor_reward": c.avg_executor_reward,
                        "avg_planner_reward": c.avg_planner_reward,
                        "lora_triggered": c.lora_triggered,
                        "metrics": c.metrics,
                    }
                    for c in self._cycles
                ],
                "total_samples": self._total_samples,
                "lora_events": self._lora_events,
                "full_sft_events": self._full_sft_events,
                "last_lora_time": self._last_lora_time,
                "last_full_sft_time": self._last_full_sft_time,
                "recent_kl": list(self._recent_kl),
            },
        }
        with open(p, "w") as f:
            json.dump(data, f, indent=2)
        logger.info("Checkpoint saved to %s (%d samples, %d cycles)", p, self._total_samples, len(self._cycles))
        return str(p)

    @classmethod
    def load_checkpoint(cls, path: str) -> "DistillationScheduler":
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        with open(p) as f:
            data = json.load(f)
        params = data["scheduler_params"]
        scheduler = cls(**params)
        state = data["state"]
        scheduler._buffer = [
            DistillationSample(**s) for s in state["buffer"]
        ]
        scheduler._cycles = [
            DistillationCycle(**c) for c in state["cycles"]
        ]
        scheduler._total_samples = state["total_samples"]
        scheduler._lora_events = state["lora_events"]
        scheduler._full_sft_events = state["full_sft_events"]
        scheduler._last_lora_time = state["last_lora_time"]
        scheduler._last_full_sft_time = state["last_full_sft_time"]
        scheduler._recent_kl = deque(state["recent_kl"], maxlen=100)
        logger.info("Checkpoint loaded from %s (%d samples, %d cycles)", p, scheduler._total_samples, len(scheduler._cycles))
        return scheduler

    def _log_cycle(self, cycle: DistillationCycle):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / "distill_cycles.jsonl"
        entry = {
            "cycle_id": cycle.cycle_id,
            "samples": cycle.samples,
            "total_loss": cycle.total_loss,
            "bc_loss": cycle.bc_loss,
            "rl_loss": cycle.rl_loss,
            "reg_loss": cycle.reg_loss,
            "kl": cycle.kl_divergence,
            "avg_executor_reward": cycle.avg_executor_reward,
            "avg_planner_reward": cycle.avg_planner_reward,
            "lora_triggered": cycle.lora_triggered,
            "elapsed_sec": round(cycle.end_time - cycle.start_time, 1),
            "timestamp": cycle.end_time,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


__all__ = ["DistillationScheduler", "DistillationSample", "DistillationCycle"]
