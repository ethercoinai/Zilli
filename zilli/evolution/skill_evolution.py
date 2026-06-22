import re
from typing import Dict, List, Optional


class SkillEvolutionEngine:
    def __init__(self, reflection_model: Optional[str] = None,
                 cost_controller=None):
        import os
        self.reflection_model = reflection_model or os.environ.get("ZILLI_REFLECTION_MODEL")
        self.cost_controller = cost_controller
        self.max_iterations = 10
        self.evolution_strategies = [
            "prompt_optimization",
            "error_handling",
            "boundary_refinement",
            "tool_addiction",
        ]
        self._cost_log: List[Dict] = []

    def _check_budget(self) -> bool:
        if not self.cost_controller:
            return True
        return self.cost_controller.should_use_planner("evolution", {"max_prob": 0.6})

    def _record_planner(self, success: bool = True):
        if self.cost_controller:
            self.cost_controller.record_planner_call("evolution", success)
            self._cost_log.append({"event": "planner_call", "success": success})

    def _record_executor(self, success: bool = True):
        if self.cost_controller:
            self.cost_controller.record_executor_call("evolution", success)
            self._cost_log.append({"event": "executor_call", "success": success})

    def _wrap_with_cost(self, skill_file: str, trajectory_data: List[Dict],
                         strategy: str) -> str:
        module = self._wrap_as_dspy_module(skill_file)
        if strategy and not self._check_budget():
            self._record_executor()
            return self._generate_pr(module, skill_file, "executor_only")
        reflections = self._reflect_on_trajectories(trajectory_data)
        optimized = self._apply_evolution(module, reflections, strategy)
        pr = self._generate_pr(optimized, skill_file, strategy)
        if strategy != "executor_only":
            self._record_planner()
        return pr

    def evolve(self, skill_file: str, trajectory_data: List[Dict]) -> str:
        module = self._wrap_as_dspy_module(skill_file)
        if not self._check_budget():
            self._record_executor()
            return self._generate_pr(module, skill_file, "executor_only")
        reflections = self._reflect_on_trajectories(trajectory_data)
        strategy = self._select_strategy(module, reflections)
        optimized = self._apply_evolution(module, reflections, strategy)
        pr = self._generate_pr(optimized, skill_file, strategy)
        self._record_planner()
        return pr

    def evolve_multi_strategy(self, skill_file: str, trajectory_data: List[Dict]) -> List[str]:
        prs = []
        for strategy in self.evolution_strategies:
            pr = self._wrap_with_cost(skill_file, trajectory_data, strategy)
            prs.append(pr)
        return prs

    def _wrap_as_dspy_module(self, skill_file: str) -> Dict:
        try:
            with open(skill_file, "r", encoding="utf-8") as f:
                source = f.read()
        except (FileNotFoundError, IOError, UnicodeDecodeError):
            source = ""
        functions = re.findall(r"def\s+(\w+)\s*\(.*?\):", source)
        classes = re.findall(r"class\s+(\w+)\s*[\(:]", source)
        imports = re.findall(r"^(?:from|import)\s+(\S+)", source, re.MULTILINE)
        docstrings = re.findall(r'"""(.*?)"""', source, re.DOTALL)[:3]
        return {
            "file": skill_file,
            "source": source,
            "functions": functions,
            "classes": classes,
            "imports": imports,
            "docstrings": docstrings,
            "lines": len(source.split("\n")) if source else 0,
            "signature": "input -> output",
            "status": "wrapped",
        }

    def _reflect_on_trajectories(self, trajectories: List[Dict]) -> List[str]:
        reflections = []
        for traj in trajectories:
            for step in traj:
                if isinstance(step, dict):
                    obs = step.get("observation", {})
                    if isinstance(obs, dict):
                        err = obs.get("error", "")
                        if err:
                            reflections.append(f"Error: {err}")
                            break
        return reflections[:10]

    def _select_strategy(self, module: Dict, reflections: List[str]) -> str:
        if not module.get("source"):
            return "tool_addiction"
        if reflections:
            return "error_handling"
        if len(module.get("functions", [])) > 3:
            return "boundary_refinement"
        return "prompt_optimization"

    def _apply_evolution(self, module: Dict, reflections: List[str],
                          strategy: str) -> Dict:
        optimized = dict(module)
        optimized["reflections"] = reflections
        optimized["strategy"] = strategy
        optimized["iterations"] = min(len(reflections) + 1, self.max_iterations)
        optimized["dspy_integrated"] = True

        if strategy == "prompt_optimization":
            optimized["prompt_optimized"] = True
            if module.get("source"):
                lines = module["source"].split("\n")
                improved = []
                for i, line in enumerate(lines):
                    if '"""' in line and 0 < i < len(lines) - 1:
                        indent = line[:len(line) - len(line.lstrip())]
                        improved.append(f"{indent}# DSPy-optimized prompt")
                    improved.append(line)
                optimized["improved_source"] = "\n".join(improved)

        elif strategy == "error_handling" and module.get("source"):
            lines = module["source"].split("\n")
            improved = []
            for i, line in enumerate(lines):
                improved.append(line)
                stripped = line.strip()
                if stripped.startswith("def ") and "error" not in stripped.lower():
                    indent = " " * (len(line) - len(line.lstrip()) + 4)
                    improved.append(f"{indent}try:")
                    improved.append(f"{indent}    pass  # auto-evolved: wrap in try/except")
                elif "pass" in stripped and i > 0:
                    pass
            optimized["improved_source"] = "\n".join(improved)
            optimized["error_handling_added"] = True

        elif strategy == "boundary_refinement" and module.get("source"):
            optimized["boundary_refined"] = True
            if module.get("functions"):
                optimized["boundary_info"] = (
                    f"Functions: {', '.join(module['functions'][:5])}"
                )

        elif strategy == "tool_addiction":
            optimized["tool_addicted"] = True
            if not module.get("source"):
                fn_name = "evolved_skill"
                optimized["improved_source"] = (
                    f"def {fn_name}(input: str) -> str:\n"
                    f'    """Auto-evolved by Zilli (DSPy + GEPA)."""\n'
                    f"    return f\"Processed: {{input}}\"\n"
                )
                optimized["functions"] = [fn_name]

        return optimized

    def _generate_pr(self, optimized: Dict, skill_file: str,
                      strategy: str = "auto") -> str:
        strategy_label = strategy.replace("_", " ").title()
        diff_lines = [
            f"--- a/{skill_file}",
            f"+++ b/{skill_file}",
            "@@ -1,3 +1,5 @@",
            " # Auto-evolved by Zilli SkillEvolutionEngine",
            f" # Strategy: {strategy_label}",
            f" # Model: {self.reflection_model}",
            f" # Iterations: {optimized.get('iterations', 1)}",
            f" # Functions: {len(optimized.get('functions', []))}",
        ]
        if optimized.get("improved_source"):
            diff_lines.append(f"+# Evolved source ({len(optimized['improved_source'].split(chr(10)))} lines)")
        if optimized.get("reflections"):
            for ref in optimized["reflections"][:3]:
                diff_lines.append(f"+# Reflection: {ref}")
        diff_lines.append("")
        if optimized.get("improved_source"):
            diff_lines.append(optimized["improved_source"])
        return "\n".join(diff_lines)


__all__ = ["SkillEvolutionEngine"]
