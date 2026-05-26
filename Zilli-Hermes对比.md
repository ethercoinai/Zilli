# zilli vs hermes-web-ui 对比分析

| 维度 | **hermes-web-ui** (现有) | **zilli** (本项目) |
|------|------------------------|----------------------|
| **定位** | Hermes Agent 的 Web 管理面板 | 下一代 Agent 自主开发/RL训练框架 |
| **语言** | TypeScript 71% + Vue 26% + Python 2% | Python 100% |
| **架构** | 浏览器 → Koa BFF → Hermes Gateway → CLI | 5 Phase 闭环：Schema → Sandbox → Infra → RL → Evolution |
| **核心功能** | 聊天会话管理、平台渠道配置、用量分析、定时任务、文件浏览 | 可验证任务集、模拟沙箱、经验回放池、CISPO/GRPO 训练、Skill 进化 |
| **依赖** | node-pty, SQLite, Koa, Vue 3, Naive UI | pydantic, numpy, dspy-ai |
| **运行时** | Node.js (BFF) + 浏览器 (前端) | Python CLI + 可选的 RL 训练栈 |
| **数据存储** | `~/.hermes/config.yaml` + SQLite | `./checkpoints/` + 内存 TrajectoryStore |
| **AI 角色** | **消费 AI** — 调用已有 Hermes Agent 做对话管理 | **生产 AI** — 训练和进化 Agent 自身能力 |

## 关键差异

1. **技术栈完全不同** — TS/JS 前端项目 vs Python 训练框架，无直接代码继承关系
2. **抽象层级不同** — `hermes-web-ui` 是 Agent 的用户界面；`zilli` 是 Agent 的训练工厂
3. **数据流方向不同** — web-ui 消费 Agent 输出；zilli 生产 Agent（通过 RL 训练）
4. **可能的衔接点**：zilli 训练的模型可部署到 Hermes Agent，通过 web-ui 展示；`ContinuousLearner` 的设计意图是从生产环境（含 web-ui 产生的交互数据）吸收轨迹反哺训练

## 总结

两者属于 **同一生态的不同层级**：

```
用户交互层  ──  hermes-web-ui (Vue/Koa 面板)
                      │ 调用
                   Hermes Agent
                      ▲ 部署
模型训练层  ──  zilli (Python RL 框架)
```

web-ui 是 "用户与 Agent 交互的界面层"，zilli 是 "构建 Agent 的训练基础设施层"。
