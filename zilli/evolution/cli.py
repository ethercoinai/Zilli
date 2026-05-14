import argparse
import sys
from pathlib import Path
from zilli.evolution import SkillEvolutionEngine


def main():
    parser = argparse.ArgumentParser(description="Zilli-Evolve: Skill进化引擎")
    parser.add_argument("--input", type=str, required=True, help="轨迹数据目录")
    parser.add_argument("--target-skills", type=str, required=True, help="目标Skill目录")
    parser.add_argument("--reflection-model", type=str, default="claude-opus-4.6")
    parser.add_argument("--max-iterations", type=int, default=10)
    args = parser.parse_args()

    engine = SkillEvolutionEngine(
        reflection_model=args.reflection_model,
    )
    engine.max_iterations = args.max_iterations

    skills_dir = Path(args.target_skills)
    if not skills_dir.exists():
        print(f"Error: skills dir not found: {args.target_skills}", file=sys.stderr)
        sys.exit(1)

    for skill_file in skills_dir.glob("*.py"):
        print(f"Evolving: {skill_file}")
        pr = engine.evolve(str(skill_file), trajectory_data=[])
        print(pr)
        print("---")

    print("Evolution complete.")
