# Tutorial: Getting Started with Zilli

Zilli is a self-evolving AI agent framework. This tutorial walks through running
your first distillation experiment in under 5 minutes.

## Prerequisites

```bash
pip install -e .
pip install -e ".[train,dev]"
```

## 1. Run a distillation cycle

The simplest way to test the system:

```python
from zilli.training.distillation import DistillationScheduler, DistillationSample
from zilli.infra.device_utils import set_device

set_device("cpu")  # or "cuda" if you have a GPU

scheduler = DistillationScheduler(log_dir="./my_first_run")
samples = [
    DistillationSample(
        executor_action={"tool": "write", "key": "x", "value": "1"},
        planner_action={"tool": "write", "key": "x", "value": "1"},
        executor_log_prob=-1.0,
        planner_log_prob=-0.8,
        executor_reward=0.5,
        planner_reward=0.9,
    )
    for _ in range(50)
]

scheduler.add_batch(samples)
cycle = scheduler.run_cycle()
print(f"Loss: {cycle.total_loss:.4f}, KL: {cycle.kl_divergence:.4f}")
```

## 2. Run an AB test

Compare two hyperparameter configurations:

```python
from zilli.distillation.dsl import (
    ExperimentParams, ABTestGroup, run_ab_test, compare,
)

group = ABTestGroup(name="lr_comparison")
group.add(ExperimentParams(name="baseline", lambda_bc=1.0, lambda_rl=0.5))
group.add(ExperimentParams(name="rl_heavy", lambda_bc=0.5, lambda_rl=1.0))

iteration = run_ab_test(group, samples, log_dir="./ab_test")
print(compare(iteration.results))
```

## 3. Multi-round experiment

Run sequential rounds with automatic best-pick injection:

```python
from zilli.distillation.dsl import (
    ExperimentParams, ExperimentLineage, run_multi_round, lineage_report,
)

lineage = ExperimentLineage(name="hyperparam_search")
lineage.add_round("round_1", [
    ExperimentParams(name="A", lambda_bc=1.0, lambda_rl=0.5),
    ExperimentParams(name="B", lambda_bc=0.5, lambda_rl=1.0),
])
lineage.add_round("round_2", [
    ExperimentParams(name="C", lambda_reg=0.2),
    ExperimentParams(name="D", lambda_reg=0.05),
])

result = run_multi_round(lineage, samples, log_dir="./multi_round")
print(lineage_report(result))
```

## 4. CLI one-liner

```bash
zilli distill --samples 100 --log-dir ./cli_run
zilli distill --ab-test configs/ab_test.yaml --samples 200
```

## 5. Checkpoint and resume

```bash
zilli distill --samples 100 --checkpoint ./ckpt.json
zilli distill --samples 50 --checkpoint ./ckpt.json  # resumes
```

## 6. Run a SWE fix loop

Zilli can autonomously reproduce, diagnose, fix, and verify bugs in a codebase — a SWE-bench-style agent loop.

### Python API

```python
from zilli.models import ModelRegistry
from zilli.swe import SWEAgent, SWEConfig

# Use any registered model for diagnosis + fix generation
registry = ModelRegistry()
model = registry.get_model("executor")  # or "planner" for stronger reasoning

cfg = SWEConfig(
    max_iterations=3,
    test_command="python -m pytest tests/ -x -q",
    verbose=True,
)

agent = SWEAgent(cfg, model_backend=model)
result = await agent.run("Fix the failing test in test_parser.py", "./my_repo")

print(f"Success: {result.success}")
print(f"Patch:\n{result.patch.to_diff()}")
```

### CLI one-liner

```bash
# Basic: fix a bug from description
zilli swe --issue "修复 test_parser.py 中的解析错误" --repo ./my_repo

# Load issue from file
zilli swe --issue ./bug_report.txt --repo ./my_repo --iterations 5

# Use planner model for stronger diagnosis + Docker sandbox isolation
zilli swe --issue "API rate limiting not working" --model planner --sandbox --verbose
```

### How it works

```
Issue → [Reproduce] → [Explore] → [Diagnose (LLM)] → [Generate Fix] → [Verify] → Patch
         ↓ failure                        ↑
         └──── narrow scope, retry ───────┘
```

- **Reproduce**: Runs `pytest` (or your test command) to confirm failure
- **Explore**: `grep` + `glob` to find relevant source files
- **Diagnose**: Uses Zilli model backend to analyze root cause
- **Fix**: Generates a minimal patch via LLM
- **Verify**: Re-runs tests; on failure, narrows scope and retries (up to `max_iterations`)
- **Sandbox**: Optional Docker isolation via `zilli.swe.sandbox.Sandbox`

## Next steps

- Read `docs/reference-distillation.md` for the distillation API reference
- See `docs/howto-common-tasks.md` for CLI usage patterns, including SWE
- See `README.md` for the architecture overview
