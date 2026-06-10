# Zilli — 开发记录

## 项目概述

Zilli（原 Hermes-NG）是一个面向 AI 自主开发的下一代 Agent 工具工程方案。核心理念："AI 写 AI"、"评估即开发"、"从环境中来"、"从 Agent 到 RL"。

## 2026-05-14 开发记录

### 1. 代码审查与修复

对照 `Zilli工程文件.md` 完成代码审查，发现并修复了 8 个问题：

#### 结构性问题（高优先级）
- **实现在 `__init__.py`**：将 7 个包的实现代码从 `__init__.py` 移至对应模块文件（`mock_env.py`、`experience_replay.py`、`cispo.py`、`length_controller.py`、`verifiable_rewards.py`、`skill_evolution.py`、`continuous_learner.py`），`__init__.py` 仅做 re-export
- **CISPO_Trainer 两处定义**：统一在 `training/cispo.py`，删除 `training/__init__.py` 中的重复实现

#### 功能性问题（中优先级）
- **GAE 实现不完整**：修复 `compute_advantages()`，使用正确的蒙特卡洛回报计算；新增 `compute_gae_advantages()` 方法
- **进化引擎桩代码**：`_wrap_as_dspy_module()` 改为实际读取文件内容并提取函数名
- **持续学习空壳**：`_collect_production_trajectories()` 改为从 `production_data/*.json` 读取真实数据

#### 低优先级
- **CLI evaluate 不执行**：重写为使用 `runner.record_action()` + `runner.trajectory`
- **Scheduler task_id 冲突**：使用 `uuid.uuid4()` 替代硬编码字符串
- **GPU 配置过高**：`train_gpus` 从 256 降为 8，`inference_gpus` 从 512 降为 16

### 2. 品牌更名：Hermes-NG → Zilli

- Python 包：`hermes_ng/` → `zilli/`
- CLI 命令：`hermes-ng` → `zilli`，`hermes-evolve` → `zilli-evolve`
- 测试文件：`test_hermes_ng.py` → `test_zilli.py`
- 远程仓库：`github.com/iclawos/Hermes-NG.git` → `github.com/iclawos/Zilli.git`

### 3. 文档产出

- `NG对比.md` → `Zilli-Hermes对比.md`：Zilli 与 hermes-web-ui 对比分析
- `检查.md`：代码审查报告
- `检查报告.md`：初步审查问题清单（已全部修复）

## 项目结构

```
zilli/                   # Python 包根目录
├── __init__.py          # 顶层导出（BaseAction 等）
├── cli.py               # CLI 入口（zilli）
├── version.py           # 版本号
├── run_training.py      # 训练主入口
├── schema/
│   ├── __init__.py
│   └── actions.py       # Pydantic 动作基类（BaseAction）
├── tasks/
│   ├── __init__.py      # load_tasks() + TaskRunner
│   ├── basic/.tasks.yaml
│   └── benchmark/.tasks.yaml
├── envs/
│   ├── __init__.py
│   └── mock_env.py      # HermesSandbox + TOOL_REGISTRY
├── data/
│   ├── __init__.py
│   └── experience_replay.py  # TrajectoryStore
├── training/
│   ├── __init__.py
│   ├── cispo.py         # CISPO_Trainer
│   ├── grpo.py          # GRPO_Trainer
│   └── rl_trainer.py    # RLTrainer 工厂
├── rewards/
│   ├── __init__.py
│   └── verifiable_rewards.py  # VerifiableReward
├── infra/
│   ├── __init__.py
│   ├── length_controller.py   # LengthElasticController
│   └── async_scheduler.py     # AsyncRolloutScheduler
├── evolution/
│   ├── __init__.py
│   ├── skill_evolution.py     # SkillEvolutionEngine
│   └── cli.py                 # zilli-evolve CLI
├── learner/
│   ├── __init__.py
│   └── continuous_learner.py  # ContinuousLearner
├── configs/training_config.yaml
└── scripts/run_evolution.sh
```

## 构建与测试

```bash
# 安装
pip install -e .

# 运行测试
python3 -m pytest tests/ -v

# CLI
python3 -m zilli.cli --version
python3 -m zilli.cli list-tasks
python3 -m zilli.cli evaluate
python3 -m zilli.cli sandbox-test
```

## 架构

Phase 1: 任务定义 → Phase 2: 轨迹数据 → Phase 3: RL 基础设施 → Phase 4: RL 训练 → Phase 5: 自动进化

## 设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| Schema 严格模式 | `extra="forbid"` | 确保 tool calling 类型安全 |
| 采样策略 | `golden_ratio=0.5` | 平衡正向和负向样本 |
| 长度自适应 | Earl 三重机制 | 防止上下文爆炸 |
| RL 算法 | CISPO + GRPO | 多轮 Agent 优化，MoE 稳定 |
| 优势估计 | 蒙特卡洛 + GAE | 支持无 Value Network 场景 |

## Health Stack

- lint: ruff check .
- test: pytest
