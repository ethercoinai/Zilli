# Zilli 项目文件结构

```
zilli/
├── __init__.py                    # 顶层导出（BaseAction 等）
├── cli.py                        # CLI 入口（zilli）
├── version.py                    # 版本号
├── run_training.py               # 训练主入口（集成蒸馏 + arena）
│
├── schema/                       # Phase 1: 任务定义与类型安全
│   ├── __init__.py
│   └── actions.py                # Pydantic 动作基类（BaseAction）
│
├── tasks/                        # Phase 1: 可验证评估集
│   ├── __init__.py               # load_tasks() + TaskRunner
│   ├── basic/__init__.py         # 基础开发验证套件
│   └── benchmark/__init__.py     # 长上下文基准评估套件
│
├── envs/                         # Phase 2: 环境模拟器
│   ├── __init__.py
│   └── mock_env.py               # HermesSandbox + TOOL_REGISTRY
│
├── data/                         # Phase 2: 轨迹数据
│   ├── __init__.py
│   ├── experience_replay.py      # TrajectoryStore 分层经验回放
│   └── trajectory_cleaner.py     # TrajectoryCleaner 轨迹净化
│
├── infra/                        # Phase 3: RL 训练基础设施
│   ├── __init__.py
│   ├── length_controller.py      # LengthElasticController 三重自适应
│   └── async_scheduler.py        # AsyncRolloutScheduler 异步调度
│
├── training/                     # Phase 4: 模型训练与RL策略
│   ├── __init__.py
│   ├── rl_trainer.py             # RLTrainer 工厂（CISPO/GRPO）
│   ├── cispo.py                  # CISPO_Trainer 算法
│   ├── grpo.py                   # GRPO_Trainer 算法
│   ├── distillation.py           # DistillationScheduler 蒸馏调度
│   └── champion_challenger.py    # ChampionChallenger A/B 擂台
│
├── rewards/                      # Phase 4: 奖励塑形
│   ├── __init__.py
│   └── verifiable_rewards.py     # VerifiableReward RLVR
│
├── distillation/                 # Phase 4: 蒸馏损失函数
│   ├── __init__.py
│   └── losses.py                 # DualModelDistillationLoss
│
├── adaptive/                     # Phase 4: 自适应SOTA调度
│   ├── __init__.py
│   └── sota_scheduler.py         # DynamicSOTAScheduler
│
├── evaluation/                   # Phase 4: 独立评估
│   ├── __init__.py
│   └── executor_only_evaluator.py # ExecutorOnlyEvaluator
│
├── evolution/                    # Phase 5: 自动化进化
│   ├── __init__.py
│   ├── skill_evolution.py        # SkillEvolutionEngine GEPA
│   └── cli.py                    # zilli-evolve CLI
│
├── learner/                      # Phase 5: 持续学习
│   ├── __init__.py
│   └── continuous_learner.py     # ContinuousLearner 在线学习
│
├── configs/
│   └── training_config.yaml      # 训练配置（含蒸馏+arena）
│
└── scripts/
    └── run_evolution.sh          # 自动化进化流程脚本
```
