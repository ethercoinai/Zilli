# Hermes-NG 开发日志

## 概述

- **项目名称**: zilli
- **工程依据**: `hermes-NG工程文件.md`
- **设计哲学**: "AI 写 AI"、"评估即开发"、"从环境中来"、"从 Agent 到 RL"
- **架构**: 5 Phase 自进化闭环

---

## 开发记录

### 2026-05-13 Phase 1: 项目初始化与核心骨架

- 创建项目目录结构，配置 `pyproject.toml`
- 实现 `BaseAction` Pydantic Schema（严格类型安全，`extra="forbid"`）
- 实现基础开发验证套件 `.tasks.yaml`（F5记忆注入、Skill边界定位、轨迹强制结束、自我修正）
- 实现长上下文基准评估套件 `.tasks.yaml`（代码仓库、长流程金融分析、多Agent协作）
- 实现 `TaskRunner` 任务加载与验证引擎

**产出文件**:
- `zilli/schema/actions.py`
- `zilli/tasks/basic/.tasks.yaml`
- `zilli/tasks/benchmark/.tasks.yaml`
- `zilli/tasks/__init__.py`

### 2026-05-13 Phase 2: 环境模拟器与轨迹数据

- 实现 `HermesSandbox` 安全沙箱环境，支持 `step()` 执行动作并返回 `observation/reward/done`
- 实现模拟工具执行系统（memory_read/write, skill_create/update, bash_run, file_read/write）
- 实现 `TrajectoryStore` 分层经验回放池（黄金轨迹 + 失败反思）
- 实现 `TrajectoryStore.sample_batch` 混合采样逻辑
- 实现轨迹净化机制（错误标记、污染检测、优先级降级）
- 实现轨迹自动摘要功能

**产出文件**:
- `zilli/envs/mock_env.py`
- `zilli/data/experience_replay.py`

### 2026-05-13 Phase 3: RL训练基础设施

- 实现 `LengthElasticController` 三重长度自适应机制（cap调整、并行模式切换、布局感知分发）
- 实现 `AsyncRolloutScheduler` Windowed FIFO 异步调度器
- 实现 `LayoutAwareDispatcher` 避免集中式数据交换瓶颈
- 实现训练配置 YAML（slime+Miles, SGLang+Megatron-LM, CISPO算法, 256+512 GPU）

**产出文件**:
- `zilli/infra/length_controller.py`
- `zilli/infra/async_scheduler.py`
- `zilli/configs/training_config.yaml`

### 2026-05-13 Phase 4: RL算法与Reward系统

- 实现 `CISPO_Trainer`（带Clipping的Importance Sampling策略优化）
- 实现 `GRPO_Trainer`（Group Relative Policy Optimization，无Value Network）
- 实现 `VerifiableReward`（RLVR: 格式校验 + 任务完成 + 负向惩罚）
- 实现统一 `RLTrainer` 抽象基类与工厂方法

**产出文件**:
- `zilli/training/cispo.py`
- `zilli/training/grpo.py`
- `zilli/training/rl_trainer.py`
- `zilli/rewards/verifiable_rewards.py`

### 2026-05-13 Phase 5: 进化引擎与自动化

- 实现 `SkillEvolutionEngine`（DSPy模块封装 → 轨迹反思 → GEPA进化搜索 → PR生成）
- 实现 `ContinuousLearner` 在线持续学习循环
- 实现 `run_evolution.sh` 自动化任务调度脚本
- 实现训练主入口 `run_training.py`（端到端执行控制）
- 实现 CLI 命令行接口
- 编写单元测试

**产出文件**:
- `zilli/evolution/skill_evolution.py`
- `zilli/evolution/cli.py`
- `zilli/learner/continuous_learner.py`
- `zilli/scripts/run_evolution.sh`
- `zilli/run_training.py`
- `zilli/cli.py`
- `tests/` 测试套件

---

## 设计决策记录

| 决策 | 选项 | 选择 | 理由 |
|------|------|------|------|
| Schema严格模式 | extra="forbid" vs "allow" | forbid | 确保tool calling类型安全 |
| 采样策略 | golden_ratio=0.5 | 50%黄金+50%失败 | 平衡正向和负向样本 |
| 长度自适应 | 动态cap v.s. 固定窗口 | Earl三重机制 | 防止上下文爆炸 |
| RL算法 | CISPO v.s. PPO | CISPO | 专为多轮Agent优化，MoE稳定 |
| 优势估计 | GRPO v.s. Critic | 两者均支持 | GRPO适合长度剧烈变化场景 |
