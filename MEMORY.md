# MEMORY.md — 会话状态记录

## 本次完成

### P0（全部完成）
- **zilli-rs 决策**：确定"项目内 Rust helper 库"路线，文档化在 `docs/zilli-rs-decision.md`
- **Celery+Redis 工作流引擎**：创建 `zilli/workflow/` 包 — `celery_app.py`、`celery_executor.py`、`tasks.py`、`workflow_dag.py`；支持 DAG 持久化执行、任务重试、结果回调
- **ChromaDB 向量存储**：`zilli/envs/vector_store.py` — 替代内存 TrajectoryStore；支持集合管理、语义检索、元数据过滤

### P1（全部完成）
- **Planner 调用频率控制器**：
  - `zilli/envs/planner_budget.py` — `PlannerBudget` 类，滑动窗口 (deque) 硬性限制 planner 比例 ≤ 5%
  - `zilli/routing/frequency_controller.py` — `PlannerFrequencyController` 类，持久化文件 + 时间窗口比例控制
  - 集成到 `LocalHybridRouter` — router 初始化时接收 `planner_budget`，超限时跳过 plan() 直接走 executor
- **Streamlit 管理台**：`zilli/dashboard_app.py` — 审计日志浏览/导出、成本监控、系统状态、DAG 运行记录
- **模型能力画像系统**：`zilli/models/profiler.py` — `ModelProfiler` 类，ELO 评分、六维能力雷达图、任务结果追踪

### P2（完成）
- **install.sh 一键部署脚本**：自动检测 python/uv/pip，创建 venv，安装依赖，可选 Docker 中间件检查

### 测试
- 新增 15 个测试（`test_planner_budget.py`、`test_frequency_controller.py`、`test_profiler.py`）
- 全量测试 608 passed（原有 593 + 新增 15）
- 新模块覆盖率约 95%

## 待办
- 无阻塞项，所有 P0/P1/P2 均已实现
- 建议后续接入 CI（GitHub Actions）自动跑 608 个测试
- Streamlit dashboard 需搭配 `streamlit run zilli/dashboard_app.py` 使用
- Celery worker 需启动 `celery -A zilli.workflow.celery_app worker --loglevel=info`
