import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from zilli.adaptive.sota_scheduler import DynamicSOTAScheduler
from zilli.data import TrajectoryStore
from zilli.envs import HermesSandbox
from zilli.infra import LengthElasticController
from zilli.infra.async_scheduler import AsyncRolloutScheduler
from zilli.schema.actions import FinishAction, MemoryWriteAction
from zilli.tasks import load_tasks
from zilli.training.champion_challenger import ArenaStatus, ChampionChallenger
from zilli.training.distillation import DistillationSample, DistillationScheduler
from zilli.training.rl_trainer import RLTrainer

logger = logging.getLogger("zilli.train")


class TrainingExperiment:
    def __init__(self, name: str, config: Dict[str, Any], log_dir: str = "./experiments"):
        self.name = name
        self.config = config
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics: list = []
        self.start_time = time.time()
        self.best_reward = float("-inf")

    def log_epoch(self, epoch: int, metrics: Dict[str, Any]):
        entry = {
            "epoch": epoch,
            "elapsed_sec": round(time.time() - self.start_time, 1),
            **metrics,
        }
        self.metrics.append(entry)

        log_path = self.log_dir / f"{self.name}_metrics.jsonl"
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def save_checkpoint(self, tag: str, extra: Optional[Dict] = None):
        ckpt = {
            "experiment": self.name,
            "tag": tag,
            "timestamp": time.time(),
            "config": self.config,
            "epoch": len(self.metrics),
            "best_reward": self.best_reward,
            "metrics": self.metrics[-10:] if self.metrics else [],
        }
        if extra:
            ckpt.update(extra)
        ckpt_path = self.log_dir / f"{self.name}_ckpt_{tag}.json"
        with open(ckpt_path, "w") as f:
            json.dump(ckpt, f, indent=2)
        logger.info("Checkpoint saved: %s", ckpt_path)

    def summary(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "epochs": len(self.metrics),
            "elapsed_sec": round(time.time() - self.start_time, 1),
            "best_reward": self.best_reward,
            "latest_metrics": self.metrics[-1] if self.metrics else None,
        }


def _wrap_step(action: Dict, result: Dict) -> Dict:
    return {
        "action": action,
        "observation": result.get("observation", {}),
        "reward": result.get("reward", 0.0),
        "done": result.get("done", False),
    }


async def run_rollout(sandbox: HermesSandbox, task: Dict) -> Dict:
    sandbox.reset()
    traj = []
    total_reward = 0.0
    max_steps = task.get("max_steps", 10)
    if max_steps < 1:
        max_steps = 1
    task_id = task.get("id", "unknown")
    scenario = task.get("initial_context", {})
    if scenario:
        sandbox.context["scenario"] = scenario

    steps_taken = 0
    for step_num in range(max_steps):
        steps_taken = step_num + 1
        action = MemoryWriteAction(
            action_id=f"{task_id}_{step_num}",
            reasoning=f"Step {step_num} of {task_id}",
            key=f"step_{step_num}",
            value=f"progress_{step_num}",
        )
        action_dict = action.model_dump()
        result = await sandbox.step(action)
        traj.append(_wrap_step(action_dict, result))
        total_reward += result.get("reward", 0.0)
        if result.get("done", False):
            break

    finish = FinishAction(
        action_id=f"{task_id}_finish",
        reasoning=f"Complete {task_id}",
        summary=f"Finished {task_id} in {steps_taken} steps",
    )
    final = await sandbox.step(finish)
    traj.append(_wrap_step(finish.model_dump(), final))
    total_reward += final.get("reward", 0.0)

    return {
        "trajectory": traj,
        "reward": total_reward,
        "tokens": max(steps_taken * 256, 256),
    }


async def main(config_path: str = None, experiment_name: str = "zilli_default"):
    base = Path(__file__).parent
    cfg_path = Path(config_path) if config_path else base / "configs" / "training_config.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)

    training_config = raw_config.get("training", {})
    num_epochs = training_config.get("num_epochs", 100)
    batch_size = training_config.get("batch_size", 128)
    checkpoint_interval = training_config.get("checkpoint_interval", 20)
    log_dir = training_config.get("log_dir", "./experiments")

    sandbox = HermesSandbox()
    store = TrajectoryStore(training_config.get("store_config"))
    scheduler = AsyncRolloutScheduler(
        window_sec=training_config.get("window_sec", 60),
        max_retries=training_config.get("max_retries", 2),
    )
    length_controller = LengthElasticController()
    trainer = RLTrainer(training_config)
    experiment = TrainingExperiment(experiment_name, training_config, log_dir)

    distill_config = training_config.get("distillation", {})
    distill_scheduler = DistillationScheduler(
        lambda_bc=distill_config.get("lambda_bc", 1.0),
        lambda_rl=distill_config.get("lambda_rl", 0.5),
        lambda_reg=distill_config.get("lambda_reg", 0.1),
        kl_beta=distill_config.get("kl_beta", 0.1),
        reward_gamma=distill_config.get("reward_gamma", 0.2),
        embedding_delta=distill_config.get("embedding_delta", 0.5),
        lora_threshold=distill_config.get("lora_threshold", 1000),
        distill_interval_hours=distill_config.get("distill_interval_hours", 24),
        full_sft_interval_days=distill_config.get("full_sft_interval_days", 7),
        log_dir=log_dir,
    )

    sota_scheduler = DynamicSOTAScheduler(
        monthly_budget_usd=training_config.get("sota_budget", 500.0),
    )

    arena_config = training_config.get("arena", {})
    arena = ChampionChallenger(
        min_win_gap=arena_config.get("min_win_gap", 0.05),
        significance_level=arena_config.get("significance_level", 0.05),
        min_eval_tasks=arena_config.get("min_eval_tasks", 10),
        warmup_rounds=arena_config.get("warmup_rounds", 3),
        log_dir=log_dir,
    )
    arena.register_model("executor_base", "v1", ArenaStatus.CHAMPION)

    tasks = load_tasks()
    if not tasks:
        logger.warning("No tasks loaded, using dummy rollouts")
        tasks = [{"id": "dummy", "max_steps": 5}]

    logger.info(
        "Starting training: %s, %d epochs, %d tasks, batch_size=%d",
        experiment_name, num_epochs, len(tasks), batch_size,
    )

    for epoch in range(num_epochs):
        batch_tasks = tasks[:8] if len(tasks) > 8 else tasks

        async def sota_aware_rollout(task: Dict) -> Dict:
            task_type = task.get("type", "default")
            difficulty = task.get("difficulty", 0.5)
            use_sota = sota_scheduler.should_call_sota(
                task_type, {"max_prob": difficulty},
            )
            result = await run_rollout(sandbox, task)
            task_success = result.get("reward", 0.0) > 0
            if use_sota:
                sota_scheduler.record_call("default", task_type, task_success)
            else:
                sota_scheduler.record_without_sota(task_type, task_success)
            return result

        rollout_results = await scheduler.schedule(
            sota_aware_rollout,
            batch_tasks,
            timeout_per_task=training_config.get("timeout_per_task", 300),
        )

        effective_lengths = [r.tokens for r in rollout_results if r.completed and r.tokens > 0]
        if effective_lengths:
            length_controller.adapt(effective_lengths)

        for result in rollout_results:
            if result.completed:
                store.add_trajectory(result.trajectory, result.reward)

        batch = store.sample_batch(batch_size=batch_size)
        if batch:
            metrics = trainer.update(batch)
        else:
            metrics = {"loss": 0.0, "policy_loss": 0.0, "kl": 0.0}

        store_stats = store.stats()
        lc_stats = length_controller.get_stats()
        sched_stats = scheduler.get_stats()
        sota_stats = sota_scheduler.stats()

        epoch_metrics = {
            "loss": metrics.get("loss", 0.0),
            "golden": store_stats["golden"],
            "failure": store_stats["failure"],
            "buffer": store_stats["buffer"],
            "cap": lc_stats["current_cap"],
            "mode": lc_stats["parallel_mode"],
            "completed": sched_stats["total_completed"],
            "errors": sched_stats["total_errors"],
            "sota_calls": sota_stats["total_calls"],
            "sota_budget_left": round(sota_stats["remaining_budget"], 2),
        }

        for result in rollout_results:
            if result.completed and result.trajectory:
                for step in result.trajectory:
                    sample = DistillationSample(
                        executor_action=step.get("action", {}),
                        planner_action=step.get("action", {}),
                        executor_log_prob=0.7,
                        planner_log_prob=0.85,
                        executor_reward=step.get("reward", 0.0),
                        planner_reward=min(step.get("reward", 0.0) + 0.2, 1.0),
                    )
                    distill_scheduler.add_sample(sample)

        if distill_scheduler.should_distill():
            cycle = distill_scheduler.run_cycle()
            if cycle:
                epoch_metrics["distill_loss"] = round(cycle.total_loss, 4)
                epoch_metrics["distill_kl"] = round(cycle.kl_divergence, 4)
                epoch_metrics["lora_triggered"] = cycle.lora_triggered

        if epoch > 0 and epoch % 20 == 0:
            def eval_fn(model_name: str) -> list:
                return [store_stats.get("avg_golden_reward", 0.0)]
            match = arena.run_match("executor_base", eval_fn)
            if match:
                epoch_metrics["arena_champion"] = match.champion_score
                epoch_metrics["arena_challenger"] = match.challenger_score
                epoch_metrics["arena_winner"] = match.winner or ""

        sota_scheduler.reset_hourly_counter()

        avg_reward = store_stats.get("avg_golden_reward", 0.0)
        if avg_reward > experiment.best_reward:
            experiment.best_reward = avg_reward

        experiment.log_epoch(epoch, epoch_metrics)

        if epoch % 10 == 0:
            purified = store.purify()
            logger.info(
                "Epoch %3d | loss=%.4f | golden=%d failure=%d buff=%d | "
                "cap=%d mode=%s | purified=%d | best=%.3f",
                epoch, metrics.get("loss", 0.0),
                store_stats["golden"], store_stats["failure"], store_stats["buffer"],
                lc_stats["current_cap"], lc_stats["parallel_mode"],
                purified, experiment.best_reward,
            )

        if epoch > 0 and epoch % checkpoint_interval == 0:
            experiment.save_checkpoint(f"epoch_{epoch}", {
                "store_stats": store_stats,
                "lc_config": length_controller.get_config(),
            })

    experiment.save_checkpoint("final", {
        "distill_stats": distill_scheduler.stats(),
        "arena_stats": arena.stats(),
        "arena_leaderboard": arena.leaderboard(),
    })
    total_summary = experiment.summary()
    total_summary["distill"] = distill_scheduler.stats()
    total_summary["arena"] = arena.stats()
    logger.info("Training complete. Summary: %s", json.dumps(total_summary))
    return experiment


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import sys
    config_path = sys.argv[1] if len(sys.argv) > 1 else None
    asyncio.run(main(config_path))
