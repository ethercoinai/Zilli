from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class BaseAction(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action_id: str = Field(..., description="唯一动作标识符")
    reasoning: str = Field(..., description="执行此动作的思考路径")
    tool_name: str = Field(..., description="调用的工具名称")


class MemoryWriteAction(BaseAction):
    tool_name: str = "memory_write"
    key: str = Field(..., description="记忆键名")
    value: str = Field(..., description="记忆内容")


class MemoryReadAction(BaseAction):
    tool_name: str = "memory_read"
    key: str = Field(..., description="要读取的记忆键名")


class SkillCreateAction(BaseAction):
    tool_name: str = "skill_create"
    name: str = Field(..., description="技能名称")
    code: str = Field(..., description="技能代码内容")
    boundary: str = Field(default="", description="边界定位信息")


class SkillUpdateAction(BaseAction):
    tool_name: str = "skill_update"
    name: str = Field(..., description="技能名称")
    code: str = Field(..., description="更新后的技能代码")


class BashRunAction(BaseAction):
    tool_name: str = "bash_run"
    command: str = Field(..., description="要执行的bash命令")
    workdir: Optional[str] = Field(None, description="工作目录")


class FileReadAction(BaseAction):
    tool_name: str = "file_read"
    path: str = Field(..., description="文件路径")


class FileWriteAction(BaseAction):
    tool_name: str = "file_write"
    path: str = Field(..., description="文件路径")
    content: str = Field(..., description="文件内容")


class FinishAction(BaseAction):
    tool_name: str = "finish"
    summary: str = Field(..., description="任务完成总结")


class TrajectoryTemplateStep(BaseModel):
    tool: str = Field(..., description="预期调用的工具名称")
    description: str = Field("", description="此步骤的说明")
    expected_keywords: Optional[List[str]] = Field(None, description="动作中应包含的关键词")
    reward_weight: float = Field(1.0, description="此步骤的奖励权重")


class RewardRule(BaseModel):
    type: Literal["format", "task_completion", "safety", "efficiency", "tool_accuracy"] = Field(...,
        description="奖励规则类型")
    weight: float = Field(1.0, description="规则权重")
    params: Dict[str, Any] = Field(default_factory=dict, description="规则参数")


class TaskConfig(BaseModel):
    id: str = Field(..., description="任务唯一标识")
    name: str = Field(..., description="任务名称")
    description: str = Field("", description="任务描述")
    category: Literal["basic", "benchmark"] = Field(..., description="任务类别")
    max_steps: int = Field(20, description="最大步数", ge=1, le=100)
    trajectory_template: List[TrajectoryTemplateStep] = Field(
        default_factory=list, description="期望的轨迹模板"
    )
    eval_criteria: List[Dict[str, Any]] = Field(
        default_factory=list, description="评估标准列表"
    )
    reward_rules: List[RewardRule] = Field(
        default_factory=list, description="奖励规则列表"
    )
    reward_spec: Dict[str, float] = Field(
        default_factory=lambda: {"success": 1.0, "partial": 0.5, "failure": -1.0},
        description="奖励数值规格",
    )
    agents: Optional[List[Dict[str, Any]]] = Field(None, description="多Agent任务的角色定义")
    initial_context: Optional[Dict[str, Any]] = Field(None, description="初始上下文")


__all__ = [
    "BaseAction",
    "MemoryWriteAction",
    "MemoryReadAction",
    "SkillCreateAction",
    "SkillUpdateAction",
    "BashRunAction",
    "FileReadAction",
    "FileWriteAction",
    "FinishAction",
    "TrajectoryTemplateStep",
    "RewardRule",
    "TaskConfig",
]
