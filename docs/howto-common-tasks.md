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
