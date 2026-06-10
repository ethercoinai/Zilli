from typing import Any, Callable, Dict, List, Optional, Union

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


@register_tool("web_search")
def _mock_web_search(ctx: Dict, query: str) -> Dict:
    ctx.setdefault("search_history", []).append(query)
    results = ctx.get("scenario", {}).get("search_results", {}).get(query)
    if results is not None:
        return {"success": True, "results": results, "query": query}
    return {
        "success": True,
        "results": [{"title": f"Mock result for: {query}", "snippet": f"This is a simulated search result for '{query}'"}],
        "query": query,
    }


@register_tool("code_interpreter")
def _mock_code_interpreter(ctx: Dict, code: str, language: str = "python") -> Dict:
    ctx.setdefault("code_history", []).append({"code": code, "language": language})
    error_patterns = ctx.get("scenario", {}).get("code_errors", [])
    for pattern in error_patterns:
        if pattern in code:
            return {"success": False, "error": f"Execution error: {pattern}", "exit_code": 1}
    return {"success": True, "stdout": f"[mock] {language} code executed successfully", "exit_code": 0}


class HermesSandbox:
    def __init__(self, scenario: Optional[Dict[str, Any]] = None):
        self.memory_store: Dict[str, Any] = {}
        self.skill_library: List[Dict] = []
        self.current_trajectory: List[Dict] = []
        self.conversation_turns: int = 0
        self.context: Dict = {
            "memory": {},
            "skills": {},
            "files": {},
            "bash_history": [],
            "search_history": [],
            "code_history": [],
            "finished": False,
        }
        if scenario:
            self.context["scenario"] = scenario
            if "initial_files" in scenario:
                self.context["files"].update(scenario["initial_files"])
            if "initial_memory" in scenario:
                self.context["memory"].update(scenario["initial_memory"])

    async def step(self, action: Union[BaseAction, Dict[str, Any]]) -> Dict[str, Any]:
        action_dict = action if isinstance(action, dict) else action.model_dump()
        self.current_trajectory.append(action_dict)
        self.conversation_turns += 1

        tool_name = action_dict.get("tool_name", "") if isinstance(action, dict) else action.tool_name
        tool_fn = TOOL_REGISTRY.get(tool_name)

        if tool_fn is None:
            return {"observation": {"error": f"Unknown tool: {tool_name}"}, "reward": -1.0, "done": True}

        try:
            kwargs = {k: v for k, v in action_dict.items()
                      if k not in ("action_id", "reasoning", "tool_name") and v is not None}

            error_prob = self.context.get("scenario", {}).get("error_probability", 0.0)
            if error_prob > 0.0:
                import random
                if random.random() < error_prob:
                    return {
                        "observation": {"error": f"Simulated environment error on tool '{tool_name}'", "success": False},
                        "reward": -0.5,
                        "done": False,
                    }

            result = tool_fn(self.context, **kwargs)
            reward = 1.0 if result.get("success") else -0.5

            max_turns = self.context.get("scenario", {}).get("max_turns")
            done_by_turns = max_turns is not None and self.conversation_turns >= max_turns
            done_by_finish = self.context.get("finished", False)

            return {"observation": result, "reward": reward, "done": done_by_turns or done_by_finish}
        except Exception as e:
            return {"observation": {"error": str(e)}, "reward": -1.0, "done": True}

    def reset(self, scenario: Optional[Dict[str, Any]] = None):
        self.memory_store = {}
        self.skill_library = []
        self.current_trajectory = []
        self.conversation_turns = 0
        self.context = {
            "memory": {}, "skills": {}, "files": {},
            "bash_history": [], "search_history": [], "code_history": [],
            "finished": False,
        }
        if scenario:
            self.context["scenario"] = scenario
            if "initial_files" in scenario:
                self.context["files"].update(scenario["initial_files"])
            if "initial_memory" in scenario:
                self.context["memory"].update(scenario["initial_memory"])

    def get_trajectory(self) -> List[Dict]:
        return self.current_trajectory

    def get_stats(self) -> Dict[str, Any]:
        return {
            "turns": self.conversation_turns,
            "trajectory_length": len(self.current_trajectory),
            "memory_keys": list(self.context.get("memory", {}).keys()),
            "skills_created": list(self.context.get("skills", {}).keys()),
            "bash_commands": len(self.context.get("bash_history", [])),
            "search_queries": len(self.context.get("search_history", [])),
            "code_executions": len(self.context.get("code_history", [])),
        }


__all__ = ["HermesSandbox", "TOOL_REGISTRY", "register_tool"]
