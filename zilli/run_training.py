import asyncio
import yaml
from pathlib import Path

from zilli.envs import HermesSandbox
from zilli.data import TrajectoryStore
from zilli.infra import LengthElasticController
from zilli.infra.async_scheduler import AsyncRolloutScheduler
from zilli.training.rl_trainer import RLTrainer
from zilli.tasks import load_tasks


async def main():
    config_path = Path(__file__).parent / "configs" / "training_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    sandbox = HermesSandbox()
    store = TrajectoryStore()
    scheduler = AsyncRolloutScheduler(window_sec=60)
    length_controller = LengthElasticController()
    trainer = RLTrainer(config.get("training", {}))

    tasks = load_tasks()

    for epoch in range(100):
        rollout_results = await scheduler.schedule(
            sandbox.step,
            tasks[:4] if len(tasks) > 4 else tasks,
        )

        effective_lengths = [r.tokens for r in rollout_results]
        length_controller.adapt(effective_lengths)

        for result in rollout_results:
            store.add_trajectory(result.trajectory, result.reward)

        batch = store.sample_batch(batch_size=128)
        if batch:
            metrics = trainer.update(batch)

        if epoch % 10 == 0:
            purified = store.purify()
            stats = store.stats()
            print(
                f"Epoch {epoch:3d} | "
                f"golden={stats['golden']} failure={stats['failure']} "
                f"purified={purified} "
                f"cap={length_controller.current_cap} "
                f"mode={length_controller.parallel_mode}"
            )


if __name__ == "__main__":
    asyncio.run(main())
