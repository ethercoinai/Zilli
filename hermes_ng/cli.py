import argparse
import sys
import yaml
from pathlib import Path

from hermes_ng.tasks import load_tasks, TaskRunner
from hermes_ng.envs import HermesSandbox
from hermes_ng.data import TrajectoryStore
from hermes_ng.training.rl_trainer import RLTrainer
from hermes_ng.rewards import VerifiableReward
from hermes_ng.version import version


def main():
    parser = argparse.ArgumentParser(description="Hermes-NG: 面向AI自主开发的Agent工具")
    parser.add_argument("--version", action="version", version=f"hermes-ng v{version}")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("list-tasks", help="列出所有可用任务")
    sub.add_parser("list-basic", help="列出基础验证任务")
    sub.add_parser("list-benchmark", help="列出基准评估任务")
    sub.add_parser("sandbox-test", help="测试沙箱环境")

    eval_parser = sub.add_parser("evaluate", help="在沙箱中评估任务")
    eval_parser.add_argument("task_id", type=str, nargs="?", help="任务ID")

    train_parser = sub.add_parser("train", help="运行训练循环")
    train_parser.add_argument("--config", type=str, default=None, help="训练配置路径")

    args = parser.parse_args()

    if args.command == "list-tasks":
        tasks = load_tasks()
        for t in tasks:
            print(f"  [{t.get('category','?')}] {t['id']}: {t['name']}")

    elif args.command == "list-basic":
        _list_category("basic")

    elif args.command == "list-benchmark":
        _list_category("benchmark")

    elif args.command == "sandbox-test":
        _run_sandbox_test()

    elif args.command == "evaluate":
        _run_evaluation(args.task_id)

    elif args.command == "train":
        _run_train(args.config)

    else:
        parser.print_help()


def _list_category(cat: str):
    tasks = load_tasks(category=cat)
    for t in tasks:
        steps = t.get("max_steps", "?")
        print(f"  {t['id']}")
        print(f"    Steps: {steps}")
        print(f"    {t.get('description', '')[:100]}")
        print()


def _run_sandbox_test():
    import asyncio
    from hermes_ng.schema.actions import MemoryWriteAction, MemoryReadAction, FinishAction

    async def test():
        sandbox = HermesSandbox()
        print("Sandbox created.")

        r1 = await sandbox.step(MemoryWriteAction(
            action_id="1", reasoning="Store name",
            key="name", value="Hermes"
        ))
        print(f"Write result: {r1}")

        r2 = await sandbox.step(MemoryReadAction(
            action_id="2", reasoning="Recall name",
            key="name"
        ))
        print(f"Read result: {r2}")

        r3 = await sandbox.step(FinishAction(
            action_id="3", reasoning="Done",
            summary="Test complete"
        ))
        print(f"Finish result: {r3}")

        print(f"\nTrajectory: {len(sandbox.get_trajectory())} steps")
        print("Sandbox test PASSED")

    asyncio.run(test())


def _run_evaluation(task_id: str = None):
    import asyncio

    tasks = load_tasks()
    if task_id:
        tasks = [t for t in tasks if t["id"] == task_id]
        if not tasks:
            print(f"Task not found: {task_id}")
            return

    async def evaluate():
        for task in tasks:
            runner = TaskRunner(task)
            reward_fn = VerifiableReward()

            print(f"Evaluating: {task['id']} ({task['name']})")

            for step_num in range(task.get("max_steps", 10)):
                if runner.should_truncate():
                    print(f"  Truncated at step {step_num}")
                    break
                runner.record_action(
                    {"tool_name": "auto", "step": step_num},
                    {"success": True, "info": f"step {step_num}"},
                )

            final_state = {
                "task_completed": True,
                "steps": runner.step_count,
                "truncated": runner.should_truncate(),
            }

            score = runner.evaluate(final_state)
            reward = reward_fn.compute_trajectory(runner.trajectory, final_state)
            print(f"  score={score:.2f} reward={reward:.2f} steps={runner.step_count}")

    asyncio.run(evaluate())


def _run_train(config_path: str = None):
    config = {
        "algorithm": "CISPO",
        "clip_range": 0.2,
        "kl_penalty": 0.01,
        "is_weight_cap": 5.0,
        "gamma": 0.99,
    }
    if config_path:
        p = Path(config_path)
        if p.exists():
            with open(p) as f:
                config.update(yaml.safe_load(f).get("training", {}))

    trainer = RLTrainer(config)
    store = TrajectoryStore()
    reward_fn = VerifiableReward()
    sandbox = HermesSandbox()

    for epoch in range(3):
        sandbox.reset()
        batch = store.sample_batch(batch_size=4) if store.golden_trajectories else []
        if batch:
            metrics = trainer.update(batch)
            print(f"Epoch {epoch}: loss={metrics['loss']:.4f}")
        else:
            print(f"Epoch {epoch}: no data yet")

    print("Training test complete.")


if __name__ == "__main__":
    main()
