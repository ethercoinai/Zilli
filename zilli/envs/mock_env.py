from typing import List, Dict, Any, Optional, Callable
from zilli.schema.actions import BaseAction


TOOL_REGISTRY: Dict[str, Callable] = {}


def register_tool(name: str):
    def wrapper(fn):
        TOOL_REGISTRY[name] = fn
        return fn
    return wrapper


@register_tool("memory_write")
def _mock_memory_write(ctx: Dict, key: str, value: str) -> Dict:
    ctx.setdefault("memory", {})[key] = value
    return {"success": True, "memory_keys": list(ctx["memory"].keys())}


@register_tool("memory_read")
def _mock_memory_read(ctx: Dict, key: str) -> Dict:
    val = ctx.get("memory", {}).get(key)
    if val is not None:
        return {"success": True, "value": val}
    return {"success": False, "error": f"Key '{key}' not found"}


@register_tool("skill_create")
def _mock_skill_create(ctx: Dict, name: str, code: str, boundary: str = "") -> Dict:
    ctx.setdefault("skills", {})[name] = {"code": code, "boundary": boundary}
    return {"success": True, "skill_name": name}


@register_tool("skill_update")
def _mock_skill_update(ctx: Dict, name: str, code: str) -> Dict:
    if name in ctx.get("skills", {}):
        ctx["skills"][name]["code"] = code
        return {"success": True, "skill_name": name}
    return {"success": False, "error": f"Skill '{name}' not found"}


@register_tool("bash_run")
def _mock_bash_run(ctx: Dict, command: str, workdir: Optional[str] = None) -> Dict:
    ctx.setdefault("bash_history", []).append(command)
    if "error" in command.lower():
        return {"success": False, "error": f"Command failed: {command}", "exit_code": 1}
    return {"success": True, "stdout": f"[mock] Executed: {command}", "exit_code": 0}


@register_tool("file_read")
def _mock_file_read(ctx: Dict, path: str) -> Dict:
    content = ctx.get("files", {}).get(path)
    if content is not None:
        return {"success": True, "content": content}
    return {"success": False, "error": f"File not found: {path}"}


@register_tool("file_write")
def _mock_file_write(ctx: Dict, path: str, content: str) -> Dict:
    ctx.setdefault("files", {})[path] = content
    return {"success": True, "path": path, "bytes": len(content)}


@register_tool("finish")
def _mock_finish(ctx: Dict, summary: str) -> Dict:
    ctx["finished"] = True
    return {"success": True, "summary": summary}


class HermesSandbox:
    def __init__(self):
        self.memory_store: Dict[str, Any] = {}
        self.skill_library: List[Dict] = []
        self.current_trajectory: List[Dict] = []
        self.context: Dict = {
            "memory": {},
            "skills": {},
            "files": {},
            "bash_history": [],
            "finished": False,
        }

    async def step(self, action: BaseAction) -> Dict[str, Any]:
        action_dict = action.model_dump()
        self.current_trajectory.append(action_dict)

        tool_name = action.tool_name
        tool_fn = TOOL_REGISTRY.get(tool_name)

        if tool_fn is None:
            return {"observation": {"error": f"Unknown tool: {tool_name}"}, "reward": -1.0, "done": True}

        try:
            kwargs = {k: v for k, v in action_dict.items()
                      if k not in ("action_id", "reasoning", "tool_name") and v is not None}
            result = tool_fn(self.context, **kwargs)
            reward = 1.0 if result.get("success") else -0.5
            done = self.context.get("finished", False)
            return {"observation": result, "reward": reward, "done": done}
        except Exception as e:
            return {"observation": {"error": str(e)}, "reward": -1.0, "done": True}

    def reset(self):
        self.memory_store = {}
        self.skill_library = []
        self.current_trajectory = []
        self.context = {
            "memory": {}, "skills": {}, "files": {},
            "bash_history": [], "finished": False,
        }

    def get_trajectory(self) -> List[Dict]:
        return self.current_trajectory


__all__ = ["HermesSandbox", "TOOL_REGISTRY", "register_tool"]
