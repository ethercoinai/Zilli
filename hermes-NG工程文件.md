# Hermes-NG：面向AI自主开发的下一代Agent工具工程方案

## 一、核心设计理念

Hermes-NG的设计哲学和开发方法，根植于对Agentic RL Scaling的研究认知：

- **“AI 写 AI”** ：Hermes-NG 直接引导大语言模型编写自己的核心工程代码。

- **“评估即开发”** ：唯一的开发标准是一套可自动验证的 “操作测试集”。系统的开发过程，即不断优化自身直至通过评估。

- **“从环境中来”** ：训练数据源于 Hermes-NG 在模拟测试环境中，自主探索、犯错和修正所生成的轨迹。

- **“从 Agent 到 RL”** ：Hermes-NG 从静态的agent框架进化为一个由强化学习驱动的动态、自适应的智能体系统。

Hermes-NG 将系统设计、数据生成、模型训练和应用部署融为一体，形成一个自我进化的闭环。整个工程方案的编排将围绕以下执行蓝图进行，所有开发任务将按这个蓝图顺序推进：

```
┌─────────────────────────────────────────────────────────────────────────────┐  
│                    Hermes-NG 自主开发执行蓝图                                │  
├─────────────────────────────────────────────────────────────────────────────┤  
│                                                                            │  
│  Phase 1        Phase 2        Phase 3        Phase 4        Phase 5      │  
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │  
│  │Definition│ → │  Data    │ → │ Infra    │ → │  Model   │ → │  Auto    │ │  
│  │  Phase   │   │  Phase   │   │  Phase   │   │  Phase   │   │  Evolve  │ │  
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘ │  
│       ↓              ↓              ↓              ↓              ↓        │  
│  评估体系规范   轨迹数据仓库   训练-推理一致性   持续RL训练    自优化闭环    │  
│  +API契约      +经验回放池   +动态长度弹性   +CISPO+GRPO    +AB测试发布   │  
│                                                                            │  
└─────────────────────────────────────────────────────────────────────────────┘
```

## 二、Phase 1：任务定义与评估体系构建

### 2.1 核心输出： 可验证评估集

Hermes-NG 的第一项任务是定义自己的目标。一切开发以**可自动验证的任务**为度量标准。所有任务统一遵循 `trajectory` + `reward` = `verifiable` 的设计范式。

**Action 1.1：基础开发验证套件（`hermes\_ng/tasks/basic/ .tasks.yaml`）**

AI 需要开发一个 YAML 配置，包含以下任务：

- **F5 记忆注入测试**：验证在多轮对话后，Hermes-NG 依然能够准确应答 3 轮前提到的关键信息（如笔名），验证 `MEMORY.md` 的正确更新。

- **Skill 边界定位**：Agent 需自主识别代码中语法错误的边界点并生成修复 `Skill`，验证 `skill\_manage` 的正确创建。

- **轨迹强制结束处理**：当 `rollout` 因长度或逻辑陷入死循环时，系统应能自动截断并指示 Agent 回顾反思。

- **自我修正**：Agent 在生成一个错误的 API 调用之后，在第二轮中应能基于系统报错信息，自主纠正并正确完成任务。

**Action 1.2：Agentic RL 长上下文评估套件（`hermes\_ng/tasks/benchmark/ .tasks.yaml`）**

- **代码仓库级任务**：Bash 环境：`git clone` + 理解 README 指引 + 执行镜像构建 + 单元测试通过。

- **长流程金融分析**：抓取 CSV 财报数据 → 数据清洗 → 多维分析 → 生成可视化图表。

- **多 Agent 协作任务**：一个 Agent 作为 “研究员” 获取信息，另一个作为 “writer” 生成最终报告。

**Action 1.3：Agent API 操作契约设计（`hermes\_ng/schema/actions.py`）**

强制 Hermes-NG 生成严格遵守类型安全的 `pydantic` 基类，用来约束自身 tool calling 的输入输出：

```
class BaseAction(BaseModel):  
    """所有的Agent动作必须继承此基类"""  
    action\_id: str = Field(..., description="唯一动作标识符")  
    reasoning: str = Field(..., description="执行此动作的思考路径")  
    tool\_name: str = Field(..., description="调用的工具名称")  
  
    class Config:  
        extra = "forbid" \# 严格模式，禁止额外字段
```

## 三、Phase 2：轨迹数据构建与环境模拟

### 3.1 核心输出： 经验轨迹仓库

人类开发者无需编写业务逻辑代码，只需制定 Hermes-NG 所处的**模拟环境**边界和自我批判的**评分标准**。

**Action 2.1：基础环境模拟器（`hermes\_ng/envs/mock\_env.py`）**

通过 `slime` 框架提供的可定制化 `rollout` 接口，AI 将指导 Hermes-NG 编码一个 Python 环境模拟器，让 Agent 在其中试错和收集轨迹：

```
import asyncio  
from typing import List, Dict, Any  
  
class HermesSandbox:  
    """Agent安全沙箱环境——所有开发、迭代、失败都在这里产生数据"""  
    def \_\_init\_\_(self):  
        self.memory\_store = \{\} \# 模拟的长期记忆  
        self.skill\_library = \[\] \# 模拟的技能库  
        self.current\_trajectory: List\[Dict\] = \[\]  
  
    async def step(self, action: BaseAction) -\> Dict\[str, Any\]:  
        """执行Agent调用的动作，记录轨迹并给出模拟reward"""  
        \# 记录动作  
        self.current\_trajectory.append(action.dict())  
        try:  
            \# 根据动作类型进行模拟逻辑处理...  
            result = await self.\_execute\_mock\_tool(action)  
            reward = 1.0 if result.get("success") else -0.5  
            return \{"observation": result, "reward": reward, "done": False\}  
        except Exception as e:  
            \# 失败轨迹也要记录！  
            return \{"observation": \{"error": str(e)\}, "reward": -1.0, "done": True\}
```

**Action 2.2：轨迹数据持久化与采样（`hermes\_ng/data/experience\_replay.py`）**

Hermes-NG 需要构建一个数据收集管道，将 `sandbox.step()` 产生的轨迹存入分层的 `experience replay buffer`。系统使用“黄金轨迹（高 Reward）”和“失败反思（低 Reward）”两类数据：

```
class TrajectoryStore:  
    """基于分层的经验回放池，优化采样效率"""  
    def \_\_init\_\_(self):  
        self.golden\_trajectories = \[\]  \# 成功率 \> 0.8  
        self.failure\_trajectories = \[\]  \# 成功率 \< 0.3，含错误信息  
        self.rollout\_buffer = \[\]        \# 临时缓存最新rollout数据  
  
    def add\_trajectory(self, trajectory: List\[Dict\], final\_reward: float):  
        if final\_reward \> 0.8:  
            self.golden\_trajectories.append(trajectory)  
        elif final\_reward \< 0.3:  
            self.failure\_trajectories.append(\{  
                "trajectory": trajectory,  
                "error\_summary": self.\_summarize\_error(trajectory)  
            \})  
  
    def sample\_batch(self, batch\_size: int, golden\_ratio: float = 0.5):  
        """混合采样：50%黄金轨迹 + 50%失败反思轨迹"""  
        n\_golden = int(batch\_size \* golden\_ratio)  
        n\_failure = batch\_size - n\_golden  
        \# 实现采样逻辑...
```

**Action 2.3：轨迹净化与数据质量控制**

在数据进入训练管道之前，Hermes-NG 需要实现**轨迹净化**机制。借鉴 CLEANER 的设计理念，利用模型内在的自校正能力，在数据收集阶段直接消除被错误污染的上下文。被污染的轨迹（例如 Agent 在错误的信息基础上进行推理导致失败）应被标记并剔除，或在采样时降低优先级。实验表明，通过轨迹净化机制，可以用**三分之一**的训练步数达到同等性能水平。

## 四、Phase 3：高效 RL 训练基础设施

### 4.1 核心输出： 训练-推理一致性管道

为实现 Agentic RL 的大规模训练，Hermes-NG 需搭建一套兼容高性能推理（SGLang）与稳定训练（Megatron-LM）的算力体系。

**Action 3.1：核心训练组件配置（`hermes\_ng/configs/training\_config.yaml`）**

```
training:  
  framework: "slime + Miles"  
  inference\_engine: "SGLang"      \# 确保与训练框架深度融合  
  training\_engine: "Megatron-LM"  \# 稳定支持MoE模型训练  
  algorithm: "CISPO"              \# 专为多轮Agent任务优化的RL算法，确保MoE大规模训练稳定  
  cluster:  
    train\_gpus: 256               \# 基于SGLang与Megatron解耦部署  
    inference\_gpus: 512           \# 支持异步rollout  
    colocated: false              \# 异步模式，最大化吞吐量  
    
  optimization:  
    on\_policy\_strictness: "bitwise"   \# True on-policy，零KL散度，基于Miles最新基础设施实现  
    context\_window\_limit: 32768       \# 动态上下文上限，超过则触发Earl机制  
    off\_policy\_ratio\_limit: 0.15      \# 允许15%的off-policy数据
```

**Action 3.2：长度自适应训练模块集成（`hermes\_ng/infra/length\_controller.py`）**

实现 Earl 论文中提出的三重机制，解决动态上下文长度爆炸问题：**长度自适应控制器**动态调整 rollout cap，**并行策略动态选择器**在长度增长时自动切换并行配置，**布局感知数据分发器**消除集中式数据交换瓶颈：

```
class LengthElasticController:  
    """实现Earl的三重机制，动态适应上下文长度"""  
    def \_\_init\_\_(self):  
        self.current\_cap = 8192  
        self.parallel\_mode = "dp"  \# data parallel / model parallel  
        self.dispatcher = LayoutAwareDispatcher()  
  
    def adapt(self, observed\_lengths: List\[int\]):  
        \# 机制1：控制器——根据观测长度调整cap  
        p95\_len = np.percentile(observed\_lengths, 95)  
        if p95\_len \> self.current\_cap \* 0.9:  
            self.current\_cap = min(p95\_len \* 1.2, 131072)  
        \# 机制2：选择器——长度超过阈值时切换并行配置  
        if p95\_len \> 32768 and self.parallel\_mode == "dp":  
            self.\_switch\_to\_mp()  \# 切换到Model Parallel避免OOM
```

**Action 3.3：异步 Rollout 调度与数据流管理**

借鉴 Forge 框架解决三难困境的工程经验，Hermes-NG 采用**Windowed FIFO**调度器：

```
class AsyncRolloutScheduler:  
    """  
    解决Stragler Effect（落后者效应）与数据分布偏移。  
    使用Windowed FIFO：多个worker并行rollout，以固定时间窗口聚合batch，  
    优先收集完成的任务，超时未完成的标记并降级。  
    """  
    def \_\_init\_\_(self, window\_sec: int = 60):  
        self.window = window\_sec  
        self.pending\_rollouts = \{\}  
      
    async def schedule(self, rollout\_fn, tasks: List\[Any\]) -\> List\[RolloutResult\]:  
        \# 实现Windowed FIFO异步调度逻辑...  
        \# 利用树形结构样本合并实现约40倍训练加速
```

**Action 3.4：训练的 End-to-End 执行控制（`hermes\_ng/run\_training.py`）**

`run\_training.py` 将串联所有模块，是整个训练流程的执行入口：

```
\#!/usr/bin/env python3  
"""Hermes-NG 模型训练主入口——完全由AI编码和自动化执行"""  
  
import asyncio  
from hermes\_ng.envs import HermesSandbox  
from hermes\_ng.data import TrajectoryStore  
from hermes\_ng.infra import AsyncRolloutScheduler, LengthElasticController  
from hermes\_ng.training.rl\_trainer import CISPO\_Trainer  
  
async def main():  
    \# Step 1: 初始化组件  
    sandbox = HermesSandbox()  
    store = TrajectoryStore()  
    scheduler = AsyncRolloutScheduler(window\_sec=60)  
    length\_controller = LengthElasticController()  
      
    \# Step 2: 主训练循环  
    for epoch in range(100):  
        \# 2.1 调用slime启动Rollout，收集轨迹  
        rollout\_results = await scheduler.schedule(sandbox.step, tasks)  \# 任务列表  
          
        \# 2.2 调用Earl机制，动态调整上下文  
        effective\_lengths = \[len(t.tokens) for t in rollout\_results\]  
        length\_controller.adapt(effective\_lengths)  
          
        \# 2.3 调用Forge清洗"失败"轨迹存入经验池  
        for result in rollout\_results:  
            store.add\_trajectory(result.trajectory, result.reward)  
          
        \# 2.4 调用Megatron Run RL policy update (CISPO)  
        trainer = CISPO\_Trainer(config\_path="hermes\_ng/configs/training\_config.yaml")  
        trainer.update(store.sample\_batch(batch\_size=128))  
          
        \# 2.5 验证评估  
        if epoch % 10 == 0:  
            eval\_score = run\_evaluation(sandbox, test\_tasks)  
            print(f"Epoch \{epoch\}: Eval Score = \{eval\_score:.4f\}")  
  
if \_\_name\_\_ == "\_\_main\_\_":  
    asyncio.run(main())
```

## 五、Phase 4：模型训练与强化学习策略

### 5.1 核心算法选型

**Action 4.1：CISPO 算法实现（`hermes\_ng/training/cispo.py`）**

CISPO (Clipping Importance Sampling Policy Optimization) 是专为 multi-turn Agent 训练的稳定 RL 算法，解决长上下文信用分配难题与 MoE 稳定性问题：

```
class CISPO\_Trainer:  
    """  
    Clipping Importance Sampling Policy Optimization  
    专为real-world Agentic RL设计，确保在大规模MoE训练中的稳定性。  
    """  
    def \_\_init\_\_(self, config):  
        self.clip\_range = 0.2       \# PPO风格的clipping  
        self.kl\_penalty = 0.01      \# 防止policy drift  
        self.is\_weight\_cap = 5.0    \# Importance Sampling cap  
      
    def compute\_loss(self, trajectories, advantages):  
        \# 核心公式：带clip的重要性采样策略优化  
        ratio = exp(trajectories.log\_probs - trajectories.old\_log\_probs)  
        clipped\_ratio = clamp(ratio, 1 - self.clip\_range, 1 + self.clip\_range)  
        surr1 = ratio \* advantages  
        surr2 = clipped\_ratio \* advantages  
        policy\_loss = -min(surr1, surr2).mean()  
        \# 加入KL惩罚项，确保训练稳定性  
        kl = (trajectories.log\_probs - trajectories.old\_log\_probs).mean()  
        return policy\_loss + self.kl\_penalty \* kl
```

**Action 4.2：Reward 塑形与 RLVR**

基于可验证环境的 reward 设计（`hermes\_ng/rewards/verifiable\_rewards.py`）：

```
class VerifiableReward:  
    """RLVR：通过规则验证生成奖励信号，零人类标注成本"""  
    def compute(self, trajectory: List\[BaseAction\], final\_state: Dict) -\> float:  
        reward = 0.0  
        \# 格式校验奖励  
        for action in trajectory:  
            if self.\_validate\_action\_schema(action):  
                reward += 0.1  
        \# 最终任务完成奖励  
        if final\_state.get("task\_completed"):  
            reward += 1.0  
        \# 负向惩罚（禁止行为）  
        if final\_state.get("forbidden\_action\_executed"):  
            reward -= 2.0  
        return max(-2.0, min(2.0, reward))  \# clamp到\[-2, 2\]
```

**Action 4.3：GRPO/DAPO 基线支持**

在训练配置中支持 GRPO（Group Relative Policy Optimization），在不依赖 Critic Network 的情况下进行优势估计，特别适合 Agent 轨迹长度和结构剧烈变化的场景：

```
class GRPO\_Trainer:  
    """Group Relative Policy Optimization——无需value network"""  
    def compute\_advantages(self, group\_trajectories):  
        rewards = \[t.reward for t in group\_trajectories\]  
        baseline = np.mean(rewards)  
        return \[r - baseline for r in rewards\]
```

## 六、Phase 5：自动化进化

### 6.1 核心输出： 版发布与自优化

开发流程的终点是 Agent 自发进化。

**Action 5.1：离线进化引擎**

AI 需要调用 `hermes-agent-self-evolution` 子模块，对训练好的模型和生成的 `Skills` 进行深度优化。进化分为五个阶段：数据准备 → 轨迹分析 → DSPy模块封装 → GEPA进化搜索 → PR生成与人类审核，在每阶段前设置安全门控机制：

```
\# 基于DSPy + GEPA的skill进化流程  
class SkillEvolutionEngine:  
    def evolve(self, skill\_file: str, trajectory\_data: List\[Dict\]) -\> str:  
        \# 1. 封装为DSPy Module  
        module = dspy.Predict(signature=skill\_signature)  
          
        \# 2. GEPA收集执行轨迹进行反思  
        reflections = self.\_reflect\_on\_trajectories(trajectory\_data)  
          
        \# 3. 遗传算法优化prompt/skill文本  
        optimized = ge\_pareto\_optimize(module, reflections)  
          
        \# 4. 生成本地进化code diff → 请求Human Review合并  
        return generate\_pr(optimized)
```

**Action 5.2：优化闭环自动化（`hermes\_ng/scripts/run\_evolution.sh`）**

自动化任务调度实现完整闭环：

```
\#!/bin/bash  
\# 自动化进化流程：Every week schedule  
\# 1. 收集过去一周所有Agent轨迹  
hermes-ng export data --start $(date -d "7 days ago" +%Y-%m-%d) --output ./rollouts/  
  
\# 2. 运行GEPA进化任务  
hermes-evolve run \\  
    --input ./rollouts \\  
    --target-skills ~/.hermes/skills/ \\  
    --reflection-model claude-opus-4.6 \\  
    --max-iterations 10  
  
\# 3. 自动创建PR等待审核 (Human-in-the-loop关键控制点)  
gh pr create --title "Auto-Evolution: Skills Optimization $(date +%Y%m%d)" \\  
             --body "Reflective prompt evolution via DSPy+GEPA" \\  
             --base main --head evolution/$(date +%Y%m%d)
```

**Action 5.3：持续学习与模型自迭代**

Hermes-NG 在生产环境中持续吸收用户交互数据，定期融合到经验回放池：

```
class ContinuousLearner:  
    """在线吸收生产环境经验，定期触发weight update"""  
    def \_\_init\_\_(self, store: TrajectoryStore, interval\_hours: int = 24):  
        self.store = store  
        self.interval = interval\_hours  
      
    async def run(self):  
        while True:  
            \# 收集新轨迹  
            new\_trajectories = await self.\_collect\_production\_trajectories()  
            for traj in new\_trajectories:  
                self.store.add\_trajectory(traj.trajectory, traj.reward)  
            \# 触发轻量级online update（低rank adaptation）  
            if len(self.store.rollout\_buffer) \> 1000:  
                self.\_trigger\_online\_sft()  
            await asyncio.sleep(self.interval \* 3600)
```

## 七、总结：这不仅仅是一个框架，而是一个生态系统

Hermes-NG 是一个完全能够“自己构建自己”的工程系统：

1. **开发由 AI 执行**：人类角色转变为 **任务与环境的定义者**，而非代码的实现者。它不需要人类在源代码层面进行冗长的调试和干预，AI 开发者在仿真环境和 Reward 框架的约束下，自动化地编写、迭代、优化自身的代码和内部 Skill。

2. **评估即开发**：所有开发目标被量化为可验证的任务集（Verifiable Tasks）。系统的“好”与“坏”完全由其在这些任务上的表现决定。这使得开发流程闭环，结果客观，完全脱离了主观代码 Review。

3. **持续进化**：在训练阶段，Hermes-NG 基于 `slime`、`Miles`、`Forge` 等当前最优的 Agentic RL 训练基础设施，实现从大规模 Post-Training（CISPO RL）到轻量化离线自优化（DSPy + GEPA）的无缝切换。在生产阶段，它不断捕捉环境交互轨迹，反哺训练数据池，实现“一天比一天智能”的自我进化。

通过以上工程化设计，Hermes-NG 不仅仅是下一代 Agent 工具，它本身就是 Agentic RL Scaling 理念在软件工程领域的实现。它展示了如何构建一个真正能够自我进化、自主开发的智能体系统。

