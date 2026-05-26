import argparse
import sys
import yaml
from pathlib import Path

from zilli.tasks import load_tasks, TaskRunner
from zilli.envs import HermesSandbox
from zilli.data import TrajectoryStore
from zilli.training.rl_trainer import RLTrainer
from zilli.rewards import VerifiableReward
from zilli.version import version


def main():
    parser = argparse.ArgumentParser(description="Zilli: 面向AI自主开发的Agent工具")
    parser.add_argument("--version", action="version", version=f"zilli v{version}")
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
    from zilli.schema.actions import MemoryWriteAction, MemoryReadAction, FinishAction

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
    from zilli.envs import HermesSandbox
    from zilli.schema.actions import MemoryWriteAction

    tasks = load_tasks()
    if task_id:
        tasks = [t for t in tasks if t["id"] == task_id]
        if not tasks:
            print(f"Task not found: {task_id}")
            return

    async def evaluate():
        for task in tasks:
            sandbox = HermesSandbox(scenario=task.get("initial_context"))
            runner = TaskRunner(task)
            reward_fn = VerifiableReward()

            print(f"Evaluating: {task['id']} ({task['name']})")

            for step_num in range(task.get("max_steps", 10)):
                if runner.should_truncate():
                    print(f"  Truncated at step {step_num}")
                    break
                action = MemoryWriteAction(
                    action_id=f"eval_{step_num}",
                    reasoning=f"Step {step_num}",
                    key=f"k_{step_num}", value=f"v_{step_num}",
                )
                result = await sandbox.step(action)
                runner.record_action(action.model_dump(), result.get("observation", {}))

            final_state = {
                "task_completed": sandbox.context.get("finished", False),
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

    import numpy as np
    dummy_golden = [
        {"action": {"tool_name": "memory_write", "key": "x", "value": "1"},
         "observation": {"success": True}, "reward": 1.0, "done": False,
         "log_prob": np.random.normal(-1.0, 0.5), "old_log_prob": np.random.normal(-0.5, 0.5)},
        {"action": {"tool_name": "memory_read", "key": "x"},
         "observation": {"success": True, "value": "1"}, "reward": 1.0, "done": True,
         "log_prob": np.random.normal(-1.0, 0.5), "old_log_prob": np.random.normal(-0.5, 0.5)},
    ]
    dummy_failure = [
        {"action": {"tool_name": "bash_run", "command": "bad"},
         "observation": {"error": "fail", "success": False}, "reward": -0.5, "done": True,
         "log_prob": np.random.normal(-2.0, 0.5), "old_log_prob": np.random.normal(-1.0, 0.5)},
    ]
    for i in range(10):
        store.add_trajectory(dummy_golden, 0.9 + i * 0.01)
    for i in range(5):
        store.add_trajectory(dummy_failure, 0.1)

    for epoch in range(3):
        sandbox.reset()
        batch = store.sample_batch(batch_size=4, golden_ratio=0.5)
        if batch:
            metrics = trainer.update(batch)
            print(f"Epoch {epoch}: loss={metrics['loss']:.4f} policy_loss={metrics['policy_loss']:.4f}")
        else:
            print(f"Epoch {epoch}: no data yet")

    print("Training test complete.")


if __name__ == "__main__":
    main()
