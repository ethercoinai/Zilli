from pydantic import BaseModel, Field
from typing import Optional


class BaseAction(BaseModel):
    action_id: str = Field(..., description="唯一动作标识符")
    reasoning: str = Field(..., description="执行此动作的思考路径")
    tool_name: str = Field(..., description="调用的工具名称")

    class Config:
        extra = "forbid"


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
