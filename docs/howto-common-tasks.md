# How-to Guide: Common Tasks

## Run a custom distillation cycle

```python
from zilli.training.distillation import DistillationScheduler, DistillationSample

scheduler = DistillationScheduler(
    lambda_bc=0.8, lambda_rl=0.6, log_dir="./custom_run"
)

samples = [DistillationSample(
    executor_action={"tool": "write", "key": str(i)},
    planner_action={"tool": "write", "key": str(i)},
    executor_log_prob=-0.5,
    planner_log_prob=-0.3,
    executor_reward=1.0,
    planner_reward=1.0,
) for i in range(100)]

scheduler.add_batch(samples)
result = scheduler.run_cycle()
print(f"Loss: {result.total_loss:.4f}")
```

## Use GPU acceleration

The scheduler auto-detects CUDA. To verify:

```python
from zilli.infra.device_utils import get_device, is_gpu_available
from zilli.training.distillation import DistillationScheduler

print(f"Device: {get_device()}")
print(f"GPU: {is_gpu_available()}")

s = DistillationScheduler()
import torch
samples = [DistillationSample(
    executor_action={"a": 1}, planner_action={"a": 1},
    executor_log_prob=-0.5, planner_log_prob=-0.3,
    executor_reward=1.0, planner_reward=1.0,
)]
loss = s.compute_loss_torch(samples)  # runs on GPU if available
```

## Save and resume with checkpoint

```python
# First session
s = DistillationScheduler(log_dir="./ckpt_demo")
s.add_batch(samples_100)
s.run_cycle()
s.save_checkpoint("./ckpt.json")

# Later session
s2 = DistillationScheduler(log_dir="./ckpt_demo")
restored = s2.load_checkpoint("./ckpt.json")
assert restored
s2.add_batch(samples_50)
s2.run_cycle()  # continues from previous state
```

## Compare AB test results

```python
from zilli.distillation.dsl import ABTestGroup, run_ab_test, compare

g = ABTestGroup(name="batch_size_test")
g.add(ExperimentParams("small", lambda_bc=0.5))
g.add(ExperimentParams("large", lambda_bc=1.5))

iter = run_ab_test(g, samples)
print(compare(iter.results))
```

## Interpret output

- `total_loss` lower = better fit
- `kl_divergence` > 0.5 indicates student diverging from planner
- `bc_loss` should dominate early training, `rl_loss` later
- Check `distill_logs/` for per-cycle JSON dumps

## Run the SWE bug fix loop

The SWE agent autonomously reproduces, diagnoses, and fixes test failures.

### CLI: basic usage

```bash
# Fix by issue description
zilli swe --issue "ModuleNotFoundError when importing utils" --repo ./my_project

# Fix from file (GitHub issue body, bug report, etc.)
zilli swe --issue ./bug.md --repo ./my_project

# Increase retries for hard bugs
zilli swe --issue "随机内存越界" --repo ./cpp_project --iterations 5
```

### CLI: with model selection

```bash
# Use a specific model for diagnosis
zilli swe --issue "Login test fails" --repo ./app --model planner

# Verbose output (shows patch + context)
zilli swe --issue "API timeout" --repo ./api --model executor --verbose
```

### CLI: with sandbox

```bash
# Run in Docker sandbox (requires Docker)
zilli swe --issue "Fix CI test" --repo ./app --sandbox

# Custom test command
zilli swe --issue "Fix lint error" --repo ./lib --test-cmd "ruff check ."
```

### Python API: basic

```python
from zilli.swe import SWEAgent, SWEConfig
from zilli.models import ModelRegistry

registry = ModelRegistry()
model = registry.get_model("executor")

cfg = SWEConfig(max_iterations=3, verbose=True)
agent = SWEAgent(cfg, model_backend=model)
result = await agent.run("Fix the failing test", "./repo")

if result.success:
    print("✅ Fix verified by tests")
    print(result.patch.to_diff())
else:
    print(f"❌ Failed after {result.iterations} iterations")
    print(result.context.summarize())
```

### Python API: custom test command

```python
cfg = SWEConfig(
    test_command="npm test",
    test_timeout=300.0,
    max_iterations=5,
    target_files=["src/"],
)
agent = SWEAgent(cfg)
```

### Python API: iterate on results

```python
result = await agent.run("Bug description", "./repo")
for cycle in (result.loop_result.cycles or []):
    status = "✅" if cycle.verification and cycle.verification.passed else "❌"
    print(f"  Cycle {cycle.id}: {status} ({cycle.duration_ms:.0f}ms)")
    if cycle.error:
        print(f"    Error: {cycle.error}")
```

### Interpret SWE output

- `success`: tests passed after fix → patch is verified
- `patch.to_diff()`: unified diff of changes made
- `context.summarize()`: what files were explored, error analysis, fix proposal
- `iterations`: number of fix attempts (1 = first try)
- `loop_result.cycles`: detailed history of each attempt with verification results
