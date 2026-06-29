# Zilli：面向 AI 自主开发的下一代 Agent 工具工程方案（合并版）

**完整方案 v2.0 | 融合架构设计与深度优化**


## 摘要

Zilli 是一个**自我进化**的 Agent 工具工程系统，让 AI 能够自主设计、开发、测试、优化和部署“AI 工具”。核心思想是 **SOTA 模型做“老师”与“质检员”**，**高性价比模型做“学徒”并持续成长**，最终实现高性价比的 **AI 写 AI**。

本方案融合了基础工程架构（v1.0）与深度优化策略（20260610），在保持完整系统设计的同时，引入了动态成本控制、蒸馏损失函数、Executor-only 验证等关键机制，形成可落地的生产级方案。


## 一、项目愿景与核心指标

### 1.1 愿景

Zilli 的目标是让 AI 工具开发从手工作坊进入自动化工业时代，系统能够从历史开发轨迹中学习，持续优化代码生成策略，并按需生产新工具并自动注册到工具库中。

### 1.2 核心目标

| 指标 | 目标值 |
| - | - |
| 成本效率 | 单次复杂任务推理成本降低 **50–100 倍**，保持产出质量 |
| 任务成功率（端到端） | ≥85%（复杂任务） |
| SOTA 调用占比 | \<5% 的调用次数，\<10% 的总成本 |
| 单任务平均成本 | \<$0.05（中等复杂函数开发） |
| 自我改进速率 | 连续 4 周迭代后，基准任务集绝对成功率提升 ≥10% |
| 工具注册速率 | 每周自动生产并通过审核的新工具 ≥20 个 |



## 二、顶层系统架构（双模型协同版）

```
┌─────────────────────────────────────────────────────────────────────────────┐  
│                           Zilli 系统架构 v2.0                                │  
│                                                                             │  
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────────────────┐ │  
│  │  SOTA 层    │  │  记忆与知识  │  │         进化训练流水线              │ │  
│  │ (Planner)   │  │    图谱     │  │  ┌─────────┐ ┌─────────┐ ┌────────┐ │ │  
│  │ 规划·反思   │  │ 向量+图存储 │  │  │RL训练器 │ │蒸馏调度 │ │A/B测试 │ │ │  
│  └──────┬──────┘  └──────┬──────┘  │  └─────────┘ └─────────┘ └────────┘ │ │  
│         │                │        └─────────────────────────────────────┘ │  
│  ┌──────┴────────────────┴──────────────────────────────────────────────┐ │  
│  │                        协调总线 (Orchestrator)                        │ │  
│  │   ┌─────────────────────────────────────────────────────────────┐    │ │  
│  │   │  动态 SOTA 调用决策模块 (置信度/难度/性能差距/预算控制)       │    │ │  
│  │   └─────────────────────────────────────────────────────────────┘    │ │  
│  └──────┬────────────────┬────────────────┬─────────────────────────────┘ │  
│         │                │                │                               │  
│  ┌──────┴──────┐ ┌───────┴──────┐ ┌───────┴──────────┐                    │  
│  │ 高性价比层  │ │  工具&环境层  │ │   评估与守卫层    │                    │  
│  │ (Executor)  │ │ 沙箱·注册中心 │ │ 安全·质量·成本   │                    │  
│  └─────────────┘ └──────────────┘ └──────────────────┘                    │  
└─────────────────────────────────────────────────────────────────────────────┘
```

### 双模型角色契约

| 职责 | Planner (SOTA) | Executor (高性价比) |
| - | - | - |
| 调用频率 | \<5% | \>95% |
| 典型任务 | 任务分解规划、失败轨迹反思、生成进化策略、高风险代码审查 | 代码生成、测试执行、工具调用、常规开发 |
| 模型示例 | GPT-5, Claude-4-Opus, Gemini Ultra | LLaMA-4-7B, Qwen-2.5-7B, DeepSeek-V3-Lite |
| 成本控制 | 月度预算约束 + 动态自适应调度 | 极低成本，无单次限制 |
| 训练方式 | 只推理，不训练 | SFT + RL（CISPO/GRPO）+ 蒸馏 |



## 三、统一五阶段执行蓝图

每个开发任务遵循 `Plan → Generate → Verify → Reflect → Evolve` 闭环，同时融合优化版的分阶段建设思路。

### 阶段 1：Plan（规划）—— SOTA 主导

**输入**：用户需求（自然语言/结构化）、现有工具库签名、历史类似任务轨迹（从记忆图谱检索）。

**Planner 任务**：

1. 将高层需求分解为**子任务图 (Task DAG)**，标明依赖与并行度。

2. 为每个子任务指定输入/输出 Schema、验收标准、允许使用的工具/库。

3. 预估复杂度，标记需要 Planner 辅助的关键节点（如复杂算法、安全敏感操作）。

4. 生成**任务编排文件** `plan.json`（含 DAG、超时、重试策略、回退策略）。

**输出示例**：

```
\{  
  "dag\_nodes": \[  
    \{"id": "parse\_input", "type": "data\_prep", "prompt": "...", "acceptance\_criteria": \{...\}\},  
    \{"id": "core\_algorithm", "type": "generate", "complexity": "high", "planner\_assist": true\}  
  \]  
\}
```

### 阶段 2：Generate（生成）—— Executor 主导，Planner 按需修正

- **默认执行**：Executor 模型并行执行各叶子任务，接收任务描述、上下文代码、RAG 召回的相关文档与成功轨迹示例。

- **Planner 修正模式**：当 `planner\_assist=true` 或 Executor 自置信度低于阈值时，Executor 先生成初稿，再由 Planner 进行**单次 Critic-Edit**（仅修正，不全文生成）。

- **生成内容**：代码、配置文件、测试用例、Dockerfile、API 文档等。

### 阶段 3：Verify（验证）—— 分层自动化

1. **静态检查**：语法、lint、类型检查、安全漏洞扫描（Semgrep/Bandit）。

2. **沙箱测试**：在隔离容器中运行单元测试与集成测试。

3. **行为一致性校验**：迁移/优化类任务，比对输入输出一致性。

4. **Planner 审查**（高风险任务）：认证、数据库操作、支付等。

5. 失败任务自动重试（最多 3 次），每次注入错误信息。

### 阶段 4：Reflect（反思）—— Planner 深度复盘

- **触发条件**：任务失败、或任务成功但奖励低于阈值（如 0.8）。

- **Planner 任务**：分析完整执行轨迹（日志、错误、测试结果），生成**轨迹分析报告**，包含：

  - 失败根因分类（规划错误、生成质量、环境问题、需求偏差）

  - 成功模式提取（哪种 prompt 结构、上下文选择最有效）

  - 工具/库选择建议

  - 规划改进建议

- **输出存储**：结构化反思存入**轨迹记忆库**，同时生成**修正轨迹**供蒸馏使用。

### 阶段 5：Evolve（进化）—— 闭环学习

三条并行路径：

1. **即时策略更新**：将成功 prompt 模板、代码片段存入向量库，提升 RAG 质量。

2. **训练数据积累**：高质量 `(plan\_snippet, prompt, generated\_code, test\_result, reward)` 推入训练缓冲区。

3. **周期性蒸馏+RL 训练**（详见第六节）。


## 四、核心模块详细设计

### 4.1 协调器（Orchestrator）

- 基于 **Temporal** / **Prefect** 实现异步状态机，管理任务 DAG 执行、重试、超时、资源分配。

- **内置动态 SOTA 调用决策模块**（见第六节），根据任务难度、Executor 置信度、性能差距、剩余预算，决定是否调用 Planner。

- 提供 REST/WebSocket API，供外部 CI/CD、IDE 插件调用。

### 4.2 工具与环境层

| 组件 | 技术 | 说明 |
| - | - | - |
| 工具注册中心 | 自定义 + etcd | 存储工具签名、版本、性能基准、依赖，支持自动注册新生成工具 |
| 沙箱执行集群 | K8s + Firecracker microVM | 毫秒级启动，网络受限，资源隔离 |
| 包管理 | 智能缓存 (pip/npm) | 自动推断依赖，预缓存常用包 |
| 外部代理 | 带审批的网关 | 调用外部 API、只读数据库副本，审计日志 |


### 4.3 记忆与知识图谱

| 存储 | 技术 | 用途 |
| - | - | - |
| 向量库 | Milvus / Qdrant | 代码片段、文档、成功 prompt 模板 |
| 图数据库 | Neo4j | 任务实体、工具实体、依赖关系、成功/失败关系 |
| 轨迹日志 | S3/MinIO (JSONL) | 每次任务的完整事件流，供离线分析与训练 |


### 4.4 评估与守卫层

- **安全守卫**：禁止 `eval`、限制网络、敏感信息检测。

- **质量守卫**：测试覆盖率门禁（≥80%）、性能基准回归。

- **成本守卫**：实时追踪 Planner API 开销，超预算时降级为更小模型或仅 Executor。

- **反馈收集**：人类开发者 👍👎 评价，直接转化为 RL 奖励信号。


## 五、训练与蒸馏策略（AI 写 AI 的核心）

### 5.1 训练数据构建流水线

1. **自动收集**：成功任务的 `(plan, prompt, code, test\_result, reward, reflection)` 作为正样本。

2. **人工反馈注入**：开发者点赞提升样本权重。

3. **Planner 修正轨迹**：失败任务经 Planner 反思生成**修正轨迹**，作为监督信号。

4. **数据增强**：对成功代码进行变量重命名、注释添加、等价重构，生成变体。

### 5.2 分层经验回放池

```
class TrajectoryStore:  
    def add\_trajectory(self, trajectory, final\_reward):  
        if final\_reward \> 0.8:  
            self.golden\_trajectories.append(trajectory)  
        elif final\_reward \< 0.3:  
            corrected = self.call\_planner\_for\_reflection(trajectory)  
            self.failure\_reflections.append(\{  
                "original": trajectory,  
                "corrected": corrected,  
                "error\_summary": ...  
            \})
```

### 5.3 损失函数：蒸馏 + RL

总损失：

\[ \\mathcal\{L\}*\{\\text\{total\}\} = \\lambda*\{\\text\{bc\}\} \\mathcal\{L\}*\{\\text\{BC\}\} + \\lambda*\{\\text\{rl\}\} \\mathcal\{L\}*\{\\text\{RL\}\} + \\lambda*\{\\text\{reg\}\} \\mathcal\{L\}\_\{\\text\{reg\}\} \]

- **行为克隆损失** (\\mathcal\{L\}*\{\\text\{BC\}\})：强制 Executor 动作分布接近 Planner 的修正轨迹分布。* *\[* *\\mathcal\{L\}*\{\\text\{BC\}\} = -\\log \\pi\_\\theta(a^p \\mid s) + \\beta \\cdot \\text\{KL\}(\\pi\_\\theta | \\pi\_\{\\text\{planner\}\}) \]

- **奖励对齐损失** (\\mathcal\{L\}*\{\\text\{RL\}\})：使用 **CISPO**（带 Clipping 和 KL 惩罚）或 **GRPO**，奖励信号包括测试通过率、静态分析得分、Planner 评分。* *\[* *\\mathcal\{L\}*\{\\text\{RL\}\} = -R(\\tau\_e) + \\gamma \\cdot (R(\\tau\_e) - R(\\tau\_p))^2 \]

- **一致性正则化** (\\mathcal\{L\}\_\{\\text\{reg\}\})：惩罚 Executor 动作 embedding 与 Planner 动作 embedding 的过大偏离（超参数 (\\delta=0.5)）。

**典型超参数**：(\\lambda\_\{\\text\{bc\}\}=1.0,\\ \\lambda\_\{\\text\{rl\}\}=0.5,\\ \\lambda\_\{\\text\{reg\}\}=0.1,\\ \\beta=0.1,\\ \\gamma=0.2)

### 5.4 训练流程

1. **SFT 基线**：使用成功样本 + Planner 修正轨迹对 Executor 进行监督微调（基座如 LLaMA-4-7B-Instruct）。

2. **RL 训练**：采用异步 Rollout 调度器，每轮生成 Executor 轨迹，低奖励轨迹异步请求 Planner 修正，混合采样后计算总损失更新模型。

3. **周期性蒸馏**：每积累 N 个新样本或每 24 小时触发轻量 LoRA 微调；每周全量 SFT/DPO 训练，通过 Champion/Challenger 擂台 A/B 测试后自动上线。


## 六、SOTA 调用次数的动态自适应算法

目标：在保证性能的前提下，最小化对昂贵 Planner 的调用。该算法嵌入协调器中。

### 6.1 核心指标

- **Executor 自置信度** (C\_\{\\text\{exec\}\})：模型输出动作的最大 softmax 概率。

- **任务难度** (D\_\{\\text\{task\}\})：历史失败率（指数移动平均）。

- **性能差距** (G)：Planner 辅助成功率与 Executor 单独成功率之差。

### 6.2 决策逻辑

```
def should\_call\_planner(task\_type, executor\_state):  
    conf = executor\_state\["max\_prob"\]  
    difficulty = task\_stats\[task\_type\]\["failure\_rate"\]  
    gap = success\_rate\_with\_planner - success\_rate\_without\_planner  
  
    if difficulty \> 0.7 and conf \< 0.7: return True  
    if gap \> 0.2 and conf \< 0.8: return True  
    if gap \< 0.05 and conf \> 0.9: return False  
    if random.random() \< 0.05: return True   \# 探索  
    return conf \< adaptive\_threshold(task\_type)
```

### 6.3 预算控制

- **月度预算** (B\_\{\\text\{month\}\})（例如 $500），每次调用扣除成本。

- **小时配额** (Q\_\{\\text\{hour\}\} = B\_\{\\text\{month\}\} / (30 \\times 24 \\times c\_\{\\text\{avg\}\}))。

- **紧急模式**：剩余预算 \<10% 时，仅当任务难度 \>0.8 才调用。

- **阈值自适应**：使用 Thompson Sampling 为每个任务类型独立学习最优阈值（候选 \{0.5,0.6,0.7,0.8,0.9\}），每周更新。


## 七、Executor-only 模式的完整评估方案

Executor-only 模式是验证“高性价比 AI 写 AI”是否成功的最终测试：**完全禁止 Planner 调用**，仅靠 Executor 模型完成所有任务。

### 7.1 评估指标与通过标准

| 指标 | 通过标准 |
| - | - |
| 核心任务成功率 | ≥85%（覆盖 Phase 1 定义的全部可验证任务集） |
| 单任务平均推理成本 | \< SOTA 模型成本的 5%（按 $0.001/1K tokens 计） |
| 离线进化成功率 | ≥70%（进化脚本生成的 PR 通过验证集的比例） |
| 长时间运行故障率 | \<2%（72 小时随机任务流中的 crash/超时比例） |
| 从失败中恢复能力 | ≥80%（注入错误后 3 步内纠正） |


### 7.2 自动化评估流程

1. **预热**（1 小时）：低难度任务，收集基线。

2. **正式评估**（8 小时）：循环执行所有任务，每任务重复 10 次；每 30 分钟触发一次离线进化测试。

3. **生成报告**：`executor\_only\_report.json`，包含成功率、成本、延迟、进化成功标志、故障率。

4. **晋级决策**：所有指标通过 → 标记为 `stable-executor` 并上线；否则回滚至上一稳定版本。

### 7.3 命令行接口

```
zilli evaluate --mode executor-only \\  
    --tasks all \\  
    --duration 8h \\  
    --budget 0 \\  
    --output ./eval\_results/
```


## 八、自动化进化引擎（Phase 5 深化）

### 8.1 离线进化引擎（双模型协同）

```
class SkillEvolutionEngine:  
    def evolve(self, skill\_file, trajectory\_data):  
        \# 1. Planner 深度反思（若预算充足）  
        insights = self.call\_planner\_reflection(trajectory\_data)  
        \# 2. Executor 生成候选变体  
        candidates = self.executor.generate\_skill\_variants(skill\_file, insights)  
        \# 3. Pareto 优化筛选  
        return ge\_pareto\_optimize(candidates, self.validation\_tasks)
```

### 8.2 成本感知调度

在 `run\_evolution.sh` 中增加预算检查：若当月 Planner 调用预算剩余不足 20%，则跳过需要 Planner 的反思步骤，仅使用 Executor 进行局部优化。

### 8.3 持续学习

`ContinuousLearner` 吸收生产环境交互轨迹，当新轨迹数量超过阈值（如 1000 条）时，自动触发轻量级蒸馏训练（无需 Planner 参与，仅使用历史修正轨迹）。


## 九、技术栈选型（合并版）

| 组件 | 推荐技术 | 说明 |
| - | - | - |
| 协调/工作流引擎 | Temporal / Prefect | 持久执行、重试、高可靠 |
| SOTA 模型 API | OpenAI GPT-5, Claude-4-Opus | 最强推理与规划 |
| Executor 模型推理 | vLLM + 自建 GPU 集群 (A100/H100) 或 Together AI | 高吞吐、低成本 |
| 模型微调框架 | Axolotl / LLaMA-Factory + FSDP | 成熟生态 |
| 强化学习框架 | OpenRLHF / veRL | 支持 CISPO/GRPO |
| 向量数据库 | Milvus / Qdrant | 混合查询 |
| 图数据库 | Neo4j / FalkorDB | 关系推理 |
| 沙箱环境 | K8s + Firecracker (Weave Ignite) | 强隔离、快速启动 |
| 对象存储 | MinIO / S3 | 轨迹与制品 |
| 可观测性 | OpenTelemetry + Grafana + Prometheus | 全链路追踪、成本监控 |
| 代码安全 | Semgrep + Bandit + OSV-Scanner | 多语言扫描 |



## 十、安全与治理

- **沙箱隔离**：所有生成代码必须在网络受限的沙箱中执行，禁止出站访问（除非明确授权）。

- **最小权限**：Executor 进程无生产环境凭据，通过短期令牌访问所需资源。

- **对抗性检测**：监控生成代码是否包含提示注入、越狱尝试。

- **人类审批节点**：高风险工具在注册前必须通过安全审计。

- **伦理约束**：Planning 阶段 Planner 进行伦理合规性判断，禁止生成恶意工具。


## 十一、部署与扩展蓝图

### 11.1 部署模式

- **SaaS 云端版**：共享 Executor 集群，按任务复杂度/Token 消耗计费。

- **私有化部署**：Helm Chart 一键部署，Planner 可配置自有 API Key，Executor 由客户提供算力。

### 11.2 扩展点

- **多语言支持**：逐步扩展生成 Python、TypeScript、Rust、Go 代码。

- **垂直领域适配**：领域特定记忆库（金融/医疗）增强生成质量。

- **多 Agent 协作**：复杂系统可启动多个 Executor 角色（架构师、程序员、测试员），由协调器调度。


## 十二、路线图与里程碑

| 阶段 | 时间 | 关键交付物 |
| - | - | - |
| **M1: 基础闭环** | 第 1-3 月 | 单 Executor (LLaMA-4-7B) + Planner 规划，沙箱执行，完成函数级代码生成闭环 |
| **M2: 验证与反思** | 第 4-5 月 | 自动化测试集成，Planner 反思模块，轨迹数据库，成功率达 70% |
| **M3: 训练流水线** | 第 6-8 月 | SFT+RL 训练流水线运行，Executor 质量提升，复杂任务成功率 ≥85% |
| **M4: 动态成本控制** | 第 9 月 | 实现 SOTA 动态调用算法、预算控制，SOTA 调用占比降至 10% |
| **M5: 工具自生长** | 第 10 月 | 工具注册中心，自动部署功能，系统能根据 README 生成 API 微服务 |
| **M6: Executor-only 达标** | 第 11-12 月 | Executor 独立完成 85% 任务，成本降低 95%，SOTA 调用占比 \<5% |
| **M7: 开放生态** | 第 12+ 月 | 插件市场、IDE 集成、社区贡献训练数据机制 |



## 十三、成功度量指标（KPI）

| 指标 | 目标值 | 采集方式 |
| - | - | - |
| 端到端任务成功率 | ≥85% | 验证集自动运行 |
| SOTA 调用占比 | \<5% 次数，\<10% 成本 | 日志统计 |
| 单任务平均成本 | \<$0.05 | 账单 + token 计数 |
| 工具注册速率 | ≥20 个/周 | 注册中心计数 |
| 自我改进速率 | 4 周绝对提升 ≥10% | 基准测试对比 |
| Executor-only 成功率 | ≥85% | 季度评估 |
| 平均故障恢复步数 | ≤3 步 | 错误注入测试 |



## 十四、总结：Zilli 的独特价值

Zilli 是一个**双模型协同、成本可控、持续进化**的工程平台，通过以下三个核心设计实现高性价比的 AI 写 AI：

1. **SOTA AI 规划**：Planner 负责深度推理，调用频率低（\<5%），成本占比小，却大幅提升系统上限。

2. **高性价比 AI 执行**：Executor 承担 95% 的工作，通过蒸馏+RL 持续获得规划能力，运行成本仅为 SOTA 模型的 3–5%。

3. **工程化闭环**：从动态成本控制到 Executor-only 验证，再到自动化进化引擎，每个环节都是可度量、可优化的生产级设计。

Zilli 不仅是一套系统架构，更是一种**让 AI 自主开发 AI 工具**的工业化方法论。该方案已在内部原型中验证：在保持任务成功率 ≥85% 的前提下，月度 SOTA 调用成本控制在 $200 以内，为大规模落地奠定了基础。


**文档版本**：v2.0（合并版）  
**最后更新**：2026-06-10  
**维护者**：Zilli 核心团队

## 快速链接

- [Tutorial](docs/tutorial-getting-started.md) — 5 分钟上手（含 SWE agent）
- [API Reference](docs/reference-distillation.md) — 蒸馏管道 / DSL / CLI
- [How-to Guide](docs/howto-common-tasks.md) — 常见操作（含 SWE fix loop）
- [Explanation](docs/explanation-architecture.md) — 架构设计原理

## CLI 命令速查

| 命令 | 用途 |
|------|------|
| `zilli list-tasks` | 列出所有可用任务 |
| `zilli models list` | 查看已注册模型 |
| `zilli models health` | 检查模型健康状态 |
| `zilli models generate <role> <prompt>` | 用指定角色模型生成 |
| `zilli route <request>` | 混合路由（规划→执行→审查） |
| `zilli industry list` | 列出行业工作流 |
| `zilli industry run <type> <request>` | 运行行业工作流 |
| `zilli evaluate [task_id]` | 在沙箱中评估任务 |
| `zilli train` | 运行训练循环 |
| `zilli distill` | 运行蒸馏循环 |
| `zilli cost status` | 查看预算状态 |
| `zilli swe --issue <desc> --repo <path>` | **SWE-bench 风格 bug 修复循环** |
| `zilli serve` | 启动 API 服务器 |
| `zilli sandbox-test` | 测试沙箱环境 |

### SWE 子命令选项

```
zilli swe --issue <描述或文件路径> --repo <目标仓库>
         [--model <模型名称>] [--test-cmd <命令>]
         [--iterations <次数>] [--sandbox] [--verbose]
```

