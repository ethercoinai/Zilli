# Explanation: Zilli Architecture

## What problem does Zilli solve?

Zilli is a self-evolving AI agent framework. The core idea: use a powerful (expensive)
"Planner" model to teach a cheaper "Executor" model how to write tools, then
continuously improve the Executor through distillation and reinforcement learning.

## Why distillation?

Direct RL training of agent models is sample-inefficient. Distillation gives
the Executor a **behavior cloning signal** (what the Planner would do) alongside
the **RL signal** (what actually works). This hybrid loss converges faster and
is more stable than either alone.

```
Loss = λ_bc * BC(samples) + λ_rl * RL(samples) + λ_reg * Reg(samples)
```

- **BC** forces Executor to mimic Planner's action distribution
- **RL** rewards outcomes (test passes, task completion)
- **Reg** prevents Executor embeddings from drifting too far

## Why a DSL?

The DSL (`zilli.distillation.dsl`) provides a **declarative** interface for
experiments — no need to wire up schedulers manually for common patterns. Three
levels of abstraction:

1. `run_experiment()` — single config, single run
2. `run_ab_test()` — compare N configs on the same data
3. `run_multi_round()` — sequential rounds with auto-baseline injection

Each level delegates to `DistillationScheduler` under the hood.

## Device strategy

CUDA detection follows a **fail-soft** design:

- `detect_device("cuda")` returns `"cpu"` if no GPU found (never raises)
- `get_device()` caches the result so hot-path checks are O(1)
- `compute_loss_torch()` auto-routes to GPU/CPU based on the cached device

This means the entire pipeline works on a laptop with zero config.

## Checkpoint design

Checkpoints serialize the training buffer (accumulated samples) and scheduler
state (cycle count, parameters). This enables:

- **Resume after crash**: just re-load and continue
- **Incremental training**: run N samples, save, run M more, save
- **Portability**: checkpoints are JSON — readable, diffable, debuggable

## Benchmark bridge

The `BenchmarkTracker` decouples distillation from evaluation. It writes to
append-only JSONL files (not a DB), making it easy to:

- Compare before/after metrics across runs
- Feed data into external dashboards
- Audit historically

## CLI philosophy

The CLI (`zilli distill`) is a thin wrapper over the Python API. Design
choices:

- No `--serve` or `--pipe` modes (keep it simple)
- Config files are optional (YAML or kwargs)
- Checkpoint path doubles as "resume if exists" flag

## Code health

The project uses `ruff` for linting and `pytest` for testing. Key rules:

- `F` (Pyflakes): no undefined names, no unused imports
- `N` (Naming): PEP 8 convention, suppress with `# noqa: N801` for legacy acronyms
- `I` (isort): standard library → third party → local

Test state isolation is a known concern: shared `_global_device` in
`device_utils.py` can leak across test classes. Fixtures or per-test cleanup
are preferred over class-level teardown.
