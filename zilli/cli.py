import argparse
import time
from pathlib import Path

import yaml

from zilli.data import TrajectoryStore
from zilli.envs import HermesSandbox
from zilli.rewards import VerifiableReward
from zilli.tasks import TaskRunner, load_tasks
from zilli.training.rl_trainer import RLTrainer
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
    eval_parser.add_argument("--cost-aware", action="store_true", help="启用成本控制调度")
    eval_parser.add_argument("--budget", type=float, default=None, help="月度预算上限（美元）")

    train_parser = sub.add_parser("train", help="运行训练循环")
    train_parser.add_argument("--config", type=str, default=None, help="训练配置路径")
    train_parser.add_argument("--cost-aware", action="store_true", help="启用成本控制调度")
    train_parser.add_argument("--budget", type=float, default=None, help="月度预算上限（美元）")

    distill_parser = sub.add_parser("distill", help="运行蒸馏循环")
    distill_parser.add_argument("--config", type=str, default=None, help="蒸馏配置YAML路径")
    distill_parser.add_argument("--samples", type=int, default=100, help="蒸馏样本数")
    distill_parser.add_argument("--checkpoint", type=str, default=None, help="检查点路径（存/加载）")
    distill_parser.add_argument("--ab-test", type=str, default=None, help="AB实验配置YAML路径")
    distill_parser.add_argument("--log-dir", type=str, default="", help="日志目录")
    distill_parser.add_argument("--device", type=str, default="auto", help="推理设备 (cpu/cuda/auto)")

    cost_parser = sub.add_parser("cost", help="预算控制")
    cost_sub = cost_parser.add_subparsers(dest="cost_command")
    cost_sub.add_parser("status", help="显示预算状态")
    cost_sub.add_parser("reset-month", help="重置月度预算")

    args = parser.parse_args()

    if args.command == "cost":
        _run_cost(args.cost_command)
    elif args.command == "list-tasks":
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
        _run_evaluation(task_id=args.task_id, cost_aware=args.cost_aware, budget=args.budget)

    elif args.command == "train":
        _run_train(args.config, cost_aware=args.cost_aware, budget=args.budget)

    elif args.command == "distill":
        _run_distill(
            config_path=args.config,
            num_samples=args.samples,
            checkpoint_path=args.checkpoint,
            ab_test_path=args.ab_test,
            log_dir=args.log_dir or "",
        )

    else:
        parser.print_help()


def _run_cost(cmd: str = None):
    from zilli.envs.cost_controller import CostController

    cc = CostController()
    if cmd == "status":
        snap = cc.snapshot()
        sep = "=" * 40
        print(sep)
        print("  Budget Status")
        print(sep)
        print(f"  Remaining:    ${snap.remaining_budget:.2f}")
        print(f"  Total calls:  {snap.total_calls}")
        print(f"  Hourly calls: {snap.calls_this_hour} / {snap.hourly_quota:.0f}")
        emergency = "YES" if snap.emergency_mode else "no"
        print(f"  Emergency:    {emergency}")
        print(f"  Updated:      {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(snap.timestamp))}")
        print(sep)
        stats = cc.scheduler.stats()
        if stats["task_types"]:
            print("\n  Per-task stats:")
            for ttype, tstats in stats["task_types"].items():
                print(f"    {ttype:20s}  fail_rate={tstats['failure_rate']:.2f}  "
                      f"gap={tstats['success_with_sota'] - tstats['success_without_sota']:.2f}")
    elif cmd == "reset-month":
        cc.reset_monthly()
        print("Monthly budget reset.")
    else:
        print("Usage: zilli cost {status|reset-month}")


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

    from zilli.schema.actions import FinishAction, MemoryReadAction, MemoryWriteAction

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


def _run_evaluation(task_id: str = None, cost_aware: bool = False, budget: float = None):
    import asyncio

    from zilli.envs import CostController, HermesSandbox
    from zilli.schema.actions import MemoryWriteAction

    tasks = load_tasks()
    if task_id:
        tasks = [t for t in tasks if t["id"] == task_id]
        if not tasks:
            print(f"Task not found: {task_id}")
            return

    cc = CostController(monthly_budget=budget or 500.0) if cost_aware else None
    if cc:
        print(f"Cost-aware mode: ${cc.scheduler.monthly_budget:.0f} monthly budget")

    async def evaluate():
        planner_calls = 0
        executor_calls = 0

        for task in tasks:
            sandbox = HermesSandbox(scenario=task.get("initial_context"))
            runner = TaskRunner(task)
            reward_fn = VerifiableReward()

            task_type = task.get("category", "general")
            print(f"Evaluating: {task['id']} ({task['name']}) type={task_type}")

            for step_num in range(task.get("max_steps", 10)):
                if runner.should_truncate():
                    print(f"  Truncated at step {step_num}")
                    break

                if cc:
                    state = {"max_prob": max(0.3, 1.0 - step_num * 0.05)}
                    use_planner = cc.should_use_planner(task_type, state)
                else:
                    use_planner = False

                if use_planner:
                    planner_calls += 1
                    action = MemoryWriteAction(
                        action_id=f"eval_{step_num}",
                        reasoning=f"Planner step {step_num}",
                        key=f"k_{step_num}", value=f"v_{step_num}",
                    )
                else:
                    executor_calls += 1
                    action = MemoryWriteAction(
                        action_id=f"eval_{step_num}",
                        reasoning=f"Step {step_num}",
                        key=f"k_{step_num}", value=f"v_{step_num}",
                    )

                result = await sandbox.step(action)
                success = result.get("observation", {}).get("success", False) and result.get("reward", -1) > 0
                runner.record_action(action.model_dump(), result.get("observation", {}))

                if cc:
                    if use_planner:
                        cc.record_planner_call(task_type, success)
                    else:
                        cc.record_executor_call(task_type, success)

            final_state = {
                "task_completed": sandbox.context.get("finished", False),
                "steps": runner.step_count,
                "truncated": runner.should_truncate(),
            }

            score = runner.evaluate(final_state)
            reward = reward_fn.compute_trajectory(runner.trajectory, final_state)
            print(f"  score={score:.2f} reward={reward:.2f} steps={runner.step_count}")

        if cc:
            snap = cc.snapshot()
            print("\n  Cost summary:")
            print(f"    Planner calls: {planner_calls}")
            print(f"    Executor calls: {executor_calls}")
            print(f"    Remaining budget: ${snap.remaining_budget:.2f}")
            print(f"    Emergency mode: {'YES' if snap.emergency_mode else 'no'}")

    asyncio.run(evaluate())


def _run_train(config_path: str = None, cost_aware: bool = False, budget: float = None):
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

    cc = None
    if cost_aware:
        from zilli.envs import CostController
        cc = CostController(monthly_budget=budget or 500.0)
        print(f"Cost-aware training: ${cc.scheduler.monthly_budget:.0f} monthly budget")

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

    planner_calls = 0
    executor_calls = 0
    for epoch in range(3):
        if cc:
            state = {"max_prob": max(0.3, 0.9 - epoch * 0.2)}
            if cc.should_use_planner("training", state):
                planner_calls += 1
                print(f"  Epoch {epoch}: using Planner")
                cc.record_planner_call("training", True)
            else:
                executor_calls += 1
                print(f"  Epoch {epoch}: using Executor")
                cc.record_executor_call("training", True)

        batch = store.sample_batch(batch_size=4, golden_ratio=0.5)
        if batch:
            metrics = trainer.update(batch)
            print(f"Epoch {epoch}: loss={metrics['loss']:.4f} policy_loss={metrics['policy_loss']:.4f}")
        else:
            print(f"Epoch {epoch}: no data yet")

    if cc:
        snap = cc.snapshot()
        print("\n  Cost summary:")
        print(f"    Planner calls: {planner_calls}")
        print(f"    Executor calls: {executor_calls}")
        print(f"    Remaining budget: ${snap.remaining_budget:.2f}")

    print("Training test complete.")


def _run_distill(config_path: str = None, num_samples: int = 100,
                  checkpoint_path: str = None, ab_test_path: str = None,
                  log_dir: str = ""):
    from zilli.infra.device_utils import set_device
    from zilli.training.distillation import DistillationSample, DistillationScheduler

    if config_path:
        p = Path(config_path)
        if p.exists():
            with open(p) as f:
                config = yaml.safe_load(f).get("distillation", {})
        else:
            print(f"Config not found: {config_path}")
            return
    else:
        config = {}

    kw = {
        "lambda_bc": config.get("lambda_bc", 1.0),
        "lambda_rl": config.get("lambda_rl", 0.5),
        "lambda_reg": config.get("lambda_reg", 0.1),
        "kl_beta": config.get("kl_beta", 0.1),
        "reward_gamma": config.get("reward_gamma", 0.2),
        "embedding_delta": config.get("embedding_delta", 0.5),
        "lora_threshold": config.get("lora_threshold", 1000),
        "distill_interval_hours": config.get("distill_interval_hours", 24),
        "full_sft_interval_days": config.get("full_sft_interval_days", 7),
        "log_dir": log_dir,
    }

    device = config.get("device", "auto")
    set_device(device)
    print(f"Device: {device}")

    if checkpoint_path and Path(checkpoint_path).exists():
        scheduler = DistillationScheduler.load_checkpoint(checkpoint_path)
        print(f"Resumed from checkpoint: {checkpoint_path}")
    else:
        scheduler = DistillationScheduler(**kw)

    import numpy as np
    samples = []
    for idx in range(num_samples):
        samples.append(DistillationSample(
            executor_action={"tool": "write", "key": "x", "value": str(idx)},
            planner_action={"tool": "write", "key": "y", "value": str(idx)},
            executor_log_prob=float(np.random.normal(-1.0, 0.5)),
            planner_log_prob=float(np.random.normal(-0.5, 0.3)),
            executor_reward=float(np.random.uniform(0.3, 1.0)),
            planner_reward=float(np.random.uniform(0.6, 1.0)),
            executor_embedding=list(np.random.randn(4)),
            planner_embedding=list(np.random.randn(4)),
        ))
    scheduler.add_batch(samples)
    print(f"Added {num_samples} distillation samples")

    if ab_test_path:
        _run_ab_test_cli(scheduler, samples, ab_test_path, log_dir)
        return

    cycle = scheduler.run_cycle()
    if cycle:
        print(f"Cycle {cycle.cycle_id}: loss={cycle.total_loss:.4f} "
              f"bc={cycle.bc_loss:.4f} rl={cycle.rl_loss:.4f} "
              f"kl={cycle.kl_divergence:.4f}")

    if checkpoint_path:
        scheduler.save_checkpoint(checkpoint_path)
        print(f"Checkpoint saved: {checkpoint_path}")

    print("Distillation complete.")


def _run_ab_test_cli(scheduler, samples, ab_test_path: str, log_dir: str):
    from zilli.distillation.dsl import (
        ExperimentLineage,
        ExperimentParams,
        lineage_report,
        run_multi_round,
    )

    p = Path(ab_test_path)
    if not p.exists():
        print(f"AB test config not found: {ab_test_path}")
        return

    with open(p) as f:
        config = yaml.safe_load(f)

    lineage_config = config.get("lineage", {})
    lineage = ExperimentLineage(
        name=lineage_config.get("name", "cli_ab_test"),
        auto_baseline=lineage_config.get("auto_baseline", True),
    )

    for rd in lineage_config.get("rounds", []):
        variants = []
        for v in rd.get("variants", []):
            variant_params = {k: v for k, v in v.items() if k != "name"}
            variants.append(ExperimentParams(name=v.get("name", "variant"), **variant_params))
        lineage.add_round(rd.get("name", "round"), variants)

    if not lineage.rounds:
        print("No rounds defined in AB test config")
        return

    lineage.best_params = ExperimentParams(name="baseline")
    result = run_multi_round(lineage, samples, log_dir=log_dir)
    print(lineage_report(result))


if __name__ == "__main__":
    main()
