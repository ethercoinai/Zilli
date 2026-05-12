from typing import List, Dict, Any, Optional


class SkillEvolutionEngine:
    def __init__(self, reflection_model: Optional[str] = None):
        self.reflection_model = reflection_model or "claude-opus-4.6"
        self.max_iterations = 10

    def evolve(self, skill_file: str, trajectory_data: List[Dict]) -> str:
        module = self._wrap_as_dspy_module(skill_file)
        reflections = self._reflect_on_trajectories(trajectory_data)
        optimized = self._ge_pareto_optimize(module, reflections)
        pr = self._generate_pr(optimized, skill_file)
        return pr

    def _wrap_as_dspy_module(self, skill_file: str) -> Dict:
        return {
            "file": skill_file,
            "signature": "input -> output",
            "status": "wrapped",
        }

    def _reflect_on_trajectories(self, trajectories: List[Dict]) -> List[str]:
        reflections = []
        for traj in trajectories:
            for step in traj:
                if isinstance(step, dict) and "error" in str(step.get("observation", {})):
                    reflections.append(f"Error encountered: {step['observation']}")
                    break
        return reflections[:5]

    def _ge_pareto_optimize(self, module: Dict, reflections: List[str]) -> Dict:
        optimized = dict(module)
        optimized["reflections"] = reflections
        optimized["prompt_optimized"] = True
        optimized["iterations"] = min(len(reflections) + 1, self.max_iterations)
        return optimized

    def _generate_pr(self, optimized: Dict, skill_file: str) -> str:
        diff_lines = [
            f"--- a/{skill_file}",
            f"+++ b/{skill_file}",
            "@@ -1,3 +1,5 @@",
            " # Auto-evolved by Hermes-NG SkillEvolutionEngine",
            f" # Reflection model: {self.reflection_model}",
            f" # Iterations: {optimized.get('iterations', 1)}",
        ]
        return "\n".join(diff_lines)
