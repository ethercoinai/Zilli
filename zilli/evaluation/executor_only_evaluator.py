import asyncio
import time
import json
import numpy as np
from typing import List, Dict, Any, Callable, Optional
from dataclasses import dataclass, asdict


@dataclass
class EvalResult:
    task_name: str
    success: bool
    cost_usd: float
    inference_tokens: int
    latency_sec: float


class ExecutorOnlyEvaluator:
    def __init__(self, executor_model, task_suite: List[Dict],
                 sandbox, cost_per_1k_tokens: float = 0.001):
        self.model = executor_model
        self.tasks = task_suite
        self.sandbox = sandbox
        self.cost_per_1k_tokens = cost_per_1k_tokens

    async def run_single_task(self, task: Dict, repeat: int = 1) -> List[EvalResult]:
        results = []
        for _ in range(repeat):
            start = time.time()
            await self.sandbox.reset()
            try:
                response, token_count, _ = self.model.generate(task["prompt"])
                final_state = await self.sandbox.execute(response)
                success = task["verification_fn"](final_state)
                results.append(EvalResult(
                    task_name=task["name"],
                    success=success,
                    cost_usd=(token_count / 1000) * self.cost_per_1k_tokens,
                    inference_tokens=token_count,
                    latency_sec=time.time() - start,
                ))
            except Exception:
                results.append(EvalResult(
                    task_name=task["name"],
                    success=False, cost_usd=0.0,
                    inference_tokens=0, latency_sec=time.time() - start,
                ))
        return results

    async def evaluate(self, duration_hours: float = 8,
                       repeat_per_task: int = 10) -> Dict[str, Any]:
        all_results = []
        end_time = time.time() + duration_hours * 3600

        while time.time() < end_time:
            for task in self.tasks:
                res = await self.run_single_task(task, repeat=repeat_per_task)
                all_results.extend(res)

        total = len(all_results)
        success_count = sum(1 for r in all_results if r.success)
        total_cost = sum(r.cost_usd for r in all_results)
        avg_latency = sum(r.latency_sec for r in all_results) / total if total else 0.0
        avg_tokens = sum(r.inference_tokens for r in all_results) / total if total else 0.0

        report = {
            "success_rate": success_count / total if total else 0.0,
            "avg_cost_usd": total_cost / total if total else 0.0,
            "avg_latency_sec": avg_latency,
            "avg_tokens": avg_tokens,
            "total_cost_usd": total_cost,
            "total_runs": total,
        }
        with open("executor_only_report.json", "w") as f:
            json.dump(report, f, indent=2)
        return report


__all__ = ["ExecutorOnlyEvaluator", "EvalResult"]
