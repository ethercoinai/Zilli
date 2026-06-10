import asyncio
import numpy as np
from zilli.distillation import DualModelDistillationLoss
from zilli.adaptive import DynamicSOTAScheduler


async def main():
    scheduler = DynamicSOTAScheduler(monthly_budget_usd=100)
    loss_func = DualModelDistillationLoss()

    print("=== Distillation step ===")
    step = {
        "executor_probs": np.array([0.2, 0.6, 0.2]),
        "planner_probs": np.array([0.1, 0.8, 0.1]),
        "planner_action_id": 1,
        "executor_reward": 0.7,
        "planner_reward": 0.95,
        "executor_embed": np.array([0.3, 0.4, 0.3]),
        "planner_embed": np.array([0.2, 0.45, 0.35]),
    }
    loss = loss_func.total_loss(step)
    print(f"Total distillation loss: {loss:.4f}")

    print("\n=== SOTA Scheduler ===")
    decision = scheduler.should_call_sota("code_generation", {"max_prob": 0.68})
    print(f"Should call SOTA? {decision}")
    if decision:
        scheduler.record_call("claude-4-opus", "code_generation", actual_success=True)
    else:
        scheduler.record_without_sota("code_generation", actual_success=False)

    print(f"Scheduler stats: {scheduler.stats()}")
    print("\nAll components integrated successfully.")


if __name__ == "__main__":
    asyncio.run(main())
