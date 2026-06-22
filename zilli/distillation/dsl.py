import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

from zilli.training.distillation import (
    DistillationCycle,
    DistillationSample,
    DistillationScheduler,
)

logger = logging.getLogger("zilli.distillation.dsl")


@dataclass
class ExperimentParams:
    name: str
    lambda_bc: float = 1.0
    lambda_rl: float = 0.5
    lambda_reg: float = 0.1
    kl_beta: float = 0.1
    reward_gamma: float = 0.2
    embedding_delta: float = 0.5
    lora_threshold: int = 1000
    distill_interval_hours: int = 24
    full_sft_interval_days: int = 7
    tags: Dict[str, str] = field(default_factory=dict)

    def to_scheduler_kwargs(self) -> dict:
        return {
            "lambda_bc": self.lambda_bc,
            "lambda_rl": self.lambda_rl,
            "lambda_reg": self.lambda_reg,
            "kl_beta": self.kl_beta,
            "reward_gamma": self.reward_gamma,
            "embedding_delta": self.embedding_delta,
            "lora_threshold": self.lora_threshold,
            "distill_interval_hours": self.distill_interval_hours,
            "full_sft_interval_days": self.full_sft_interval_days,
        }

    def label(self) -> str:
        parts = [self.name]
        if self.tags:
            parts.extend(f"{k}={v}" for k, v in self.tags.items())
        return "_".join(parts)


@dataclass
class ExperimentResult:
    params: ExperimentParams
    cycles: List[DistillationCycle]
    total_samples: int
    avg_loss: float
    avg_kl: float
    avg_exec_reward: float
    avg_plan_reward: float
    wall_time_sec: float
    lora_count: int
    metadata: Dict = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "experiment": self.params.label(),
            "total_samples": self.total_samples,
            "cycles": len(self.cycles),
            "avg_loss": round(self.avg_loss, 4),
            "avg_kl": round(self.avg_kl, 4),
            "avg_exec_reward": round(self.avg_exec_reward, 4),
            "avg_plan_reward": round(self.avg_plan_reward, 4),
            "lora_events": self.lora_count,
            "lora_rate": round(self.lora_count / max(len(self.cycles), 1), 3),
            "wall_time_sec": round(self.wall_time_sec, 1),
        }


@dataclass
class ABTestGroup:
    name: str
    variants: List[ExperimentParams] = field(default_factory=list)

    def add(self, params: ExperimentParams) -> "ABTestGroup":
        self.variants.append(params)
        return self


@dataclass
class ABIteration:
    group: ABTestGroup
    results: List[ExperimentResult] = field(default_factory=list)
    best: Optional[ExperimentResult] = None

    def summary(self) -> dict:
        return {
            "group": self.group.name,
            "variants": [r.summary() for r in self.results],
            "best": self.best.summary() if self.best else None,
        }


def run_experiment(
    params: ExperimentParams,
    samples: List[DistillationSample],
    lora_callback: Optional[Callable] = None,
    full_sft_callback: Optional[Callable] = None,
    log_dir: str = "",
) -> ExperimentResult:
    kw = params.to_scheduler_kwargs()
    kw["lora_callback"] = lora_callback
    kw["full_sft_callback"] = full_sft_callback
    kw["log_dir"] = log_dir

    scheduler = DistillationScheduler(**kw)
    scheduler.add_batch(samples)

    start = time.time()
    cycle = scheduler.run_cycle()
    wall_time = time.time() - start

    cycles = [cycle] if cycle else []

    avg_loss = cycles[0].total_loss if cycles else 0.0
    avg_kl = cycles[0].kl_divergence if cycles else 0.0
    avg_exec_r = cycles[0].avg_executor_reward if cycles else 0.0
    avg_plan_r = cycles[0].avg_planner_reward if cycles else 0.0
    lora_count = sum(1 for c in cycles if c.lora_triggered)

    return ExperimentResult(
        params=params,
        cycles=cycles,
        total_samples=scheduler.stats()["total_samples"],
        avg_loss=avg_loss,
        avg_kl=avg_kl,
        avg_exec_reward=avg_exec_r,
        avg_plan_reward=avg_plan_r,
        wall_time_sec=wall_time,
        lora_count=lora_count,
    )


def run_ab_test(
    group: ABTestGroup,
    samples: List[DistillationSample],
    lora_callback: Optional[Callable] = None,
    full_sft_callback: Optional[Callable] = None,
    log_dir: str = "",
) -> ABIteration:
    results: List[ExperimentResult] = []

    for variant in group.variants:
        logger.info("Running AB variant: %s", variant.label())
        result = run_experiment(
            params=variant,
            samples=samples,
            lora_callback=lora_callback,
            full_sft_callback=full_sft_callback,
            log_dir=log_dir,
        )
        results.append(result)

    best = _pick_best(results)
    return ABIteration(group=group, results=results, best=best)


def _pick_best(results: List[ExperimentResult]) -> Optional[ExperimentResult]:
    if not results:
        return None
    return min(results, key=lambda r: r.avg_loss)


def compare(results: List[ExperimentResult]) -> dict:
    if len(results) < 2:
        return {"error": "Need at least 2 results to compare"}

    baseline = results[0]
    rows = []
    for r in results[1:]:
        rows.append({
            "variant": r.params.label(),
            "vs_baseline": baseline.params.label(),
            "loss_delta": round(r.avg_loss - baseline.avg_loss, 4),
            "kl_delta": round(r.avg_kl - baseline.avg_kl, 4),
            "exec_reward_delta": round(r.avg_exec_reward - baseline.avg_exec_reward, 4),
            "plan_reward_delta": round(r.avg_plan_reward - baseline.avg_plan_reward, 4),
            "lora_rate_delta": round(
                r.lora_count / max(len(r.cycles), 1)
                - baseline.lora_count / max(len(baseline.cycles), 1),
                3,
            ),
        })
    return {
        "baseline": baseline.params.label(),
        "comparisons": rows,
        "best": min(results, key=lambda x: x.avg_loss).params.label(),
    }


def export_results(iteration: ABIteration, path: str):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "group": iteration.group.name,
        "variants": [v.summary() for v in iteration.results],
        "best": iteration.best.summary() if iteration.best else None,
    }
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("AB results exported to %s", path)


@dataclass
class RoundDef:
    name: str
    variants: List[ExperimentParams] = field(default_factory=list)

    def add(self, params: ExperimentParams) -> "RoundDef":
        self.variants.append(params)
        return self


@dataclass
class RoundResult:
    round_name: str
    iteration: ABIteration
    round_index: int

    def summary(self) -> dict:
        return {
            "round": self.round_name,
            "index": self.round_index,
            "best": self.iteration.best.summary() if self.iteration.best else None,
            "variants": [r.summary() for r in self.iteration.results],
        }


@dataclass
class ExperimentLineage:
    name: str
    rounds: List[RoundDef] = field(default_factory=list)
    results: List[RoundResult] = field(default_factory=list)
    best_params: Optional[ExperimentParams] = None
    auto_baseline: bool = True

    def add_round(self, name: str, variants: List[ExperimentParams]) -> "ExperimentLineage":
        rd = RoundDef(name=name)
        for v in variants:
            rd.add(v)
        self.rounds.append(rd)
        return self

    def summary(self) -> dict:
        return {
            "lineage": self.name,
            "rounds": len(self.rounds),
            "auto_baseline": self.auto_baseline,
            "history": [r.summary() for r in self.results],
            "best_overall": self.best_params.label() if self.best_params else None,
        }


def run_multi_round(
    lineage: ExperimentLineage,
    samples: List[DistillationSample],
    lora_callback: Optional[Callable] = None,
    full_sft_callback: Optional[Callable] = None,
    log_dir: str = "",
) -> ExperimentLineage:
    best_from_prev: Optional[ExperimentParams] = None

    for i, rd in enumerate(lineage.rounds):
        logger.info("Starting round %d: %s (%d variants)", i, rd.name, len(rd.variants))
        variants = list(rd.variants)

        if best_from_prev and lineage.auto_baseline:
            baseline_in_round = any(v.name == best_from_prev.name for v in variants)
            if not baseline_in_round:
                logger.info("Injecting best from round %d as baseline: %s", i, best_from_prev.label())
                variants.insert(0, best_from_prev)

        group = ABTestGroup(name=f"{lineage.name}_{rd.name}")
        for v in variants:
            group.add(v)

        iteration = run_ab_test(
            group=group, samples=samples,
            lora_callback=lora_callback,
            full_sft_callback=full_sft_callback,
            log_dir=log_dir,
        )

        round_result = RoundResult(
            round_name=rd.name,
            iteration=iteration,
            round_index=i,
        )
        lineage.results.append(round_result)

        if iteration.best:
            best_from_prev = iteration.best.params

    if lineage.results and lineage.results[-1].iteration.best:
        all_results = [r.iteration.best for r in lineage.results if r.iteration.best]
        lineage.best_params = _pick_best(all_results).params if all_results else None

    return lineage


def lineage_report(lineage: ExperimentLineage) -> str:
    lines = [f"Lineage: {lineage.name}", "=" * 40]
    for r in lineage.results:
        lines.append(f"\nRound [{r.round_index}] {r.round_name}")
        lines.append("-" * 30)
        for v in r.iteration.results:
            marker = " <<< BEST" if r.iteration.best and v is r.iteration.best else ""
            lines.append(
                f"  {v.params.label():20s} loss={v.avg_loss:.4f}  kl={v.avg_kl:.4f}  "
                f"exec_r={v.avg_exec_reward:.3f}  plan_r={v.avg_plan_reward:.3f}  "
                f"t={v.wall_time_sec:.1f}s{marker}"
            )
    if lineage.best_params:
        lines.append(f"\nBest overall: {lineage.best_params.label()}")
    return "\n".join(lines)


__all__ = [
    "ExperimentParams", "ExperimentResult", "ABTestGroup", "ABIteration",
    "run_experiment", "run_ab_test", "compare", "export_results",
    "RoundDef", "RoundResult", "ExperimentLineage",
    "run_multi_round", "lineage_report",
]
