# Reference: Distillation Pipeline API

## DistillationScheduler

Primary class for distillation cycles.

```
zilli.training.distillation.DistillationScheduler
```

### Constructor

```python
DistillationScheduler(
    lambda_bc: float = 1.0,       # behavior cloning weight
    lambda_rl: float = 0.5,       # RL loss weight
    lambda_reg: float = 0.1,      # regularization weight
    kl_beta: float = 0.1,         # KL penalty coefficient
    gamma: float = 0.2,           # reward-shaping coefficient
    log_dir: str = "./distill_logs",
    device: str | None = None,    # auto-detect if None
)
```

### Key methods

| Method | Returns | Description |
|--------|---------|-------------|
| `add_batch(samples)` | None | Queue samples for next cycle |
| `run_cycle()` | DistillationCycleResult | Execute one distillation step |
| `compute_loss(samples)` | dict[str, float] | Compute loss without updating |
| `compute_loss_torch(samples)` | dict[str, float] | GPU-accelerated loss (fallback CPU) |
| `save_checkpoint(path)` | None | Serialize buffer + state |
| `load_checkpoint(path)` | bool | Restore from checkpoint |

### DistillationCycleResult

```python
@dataclass
class DistillationCycleResult:
    total_loss: float
    bc_loss: float
    rl_loss: float
    reg_loss: float
    kl_divergence: float
    buffer_size: int
    cycle_id: int
```

## Distillation DSL

Declarative experiment framework.

```
zilli.distillation.dsl
```

### ExperimentParams

```python
@dataclass
ExperimentParams(
    name: str,
    lambda_bc: float = 1.0,
    lambda_rl: float = 0.5,
    lambda_reg: float = 0.1,
    kl_beta: float = 0.1,
    gamma: float = 0.2,
)
```

### Single run

```python
run_experiment(
    params: ExperimentParams,
    samples: list[DistillationSample],
    log_dir: str = "./distill_logs",
) -> ExperimentResult
```

### AB test

```python
ABTestGroup(name: str)
    .add(params: ExperimentParams) -> Self

run_ab_test(
    group: ABTestGroup,
    samples: list[DistillationSample],
    log_dir: str = "./distill_logs",
) -> ABIteration

compare(results: list[ExperimentResult]) -> str
```

### Multi-round

```python
ExperimentLineage(name: str)
    .add_round(
        name: str,
        params: list[ExperimentParams],
        auto_baseline: bool = True,
    ) -> Self

run_multi_round(
    lineage: ExperimentLineage,
    samples: list[DistillationSample],
    log_dir: str = "./distill_logs",
) -> ExperimentLineageResult

lineage_report(result: ExperimentLineageResult) -> str
```

## Device utilities

```
zilli.infra.device_utils
```

| Function | Returns | Description |
|----------|---------|-------------|
| `detect_device(prefer="auto")` | str | Detect available device |
| `get_device(prefer=None)` | str | Get cached device (lazy init) |
| `set_device(device)` | None | Explicitly set global device |
| `is_gpu_available()` | bool | Check GPU availability |
| `validate_device(device)` | str | Validate and normalize device name |

Device strings: `"cpu"`, `"cuda"`, `"mps"`.

## Benchmark

```
zilli.evaluation.distillation_benchmark
```

```python
BenchmarkTracker(log_dir: str = "./arena_logs")
    .record(
        name: str,
        before_metrics: dict,
        after_metrics: dict,
        metadata: dict | None = None,
    ) -> None
```

Writes to `arena_logs/benchmark_entries.jsonl` and `arena_logs/distill_benchmarks.jsonl`.

## CLI

```
zilli distill [OPTIONS]
```

| Flag | Default | Description |
|------|---------|-------------|
| `--samples` | `50` | Number of samples to generate |
| `--log-dir` | `"./distill_logs"` | Output directory |
| `--checkpoint` | None | Resume from checkpoint file |
| `--config` | None | YAML config file |
| `--ab-test` | None | AB test config YAML |
| `--device` | None | Device override |
