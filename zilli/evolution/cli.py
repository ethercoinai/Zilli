import argparse
import json
import sys
from pathlib import Path

from zilli.evolution import SkillEvolutionEngine


def _load_trajectories(input_dir: Path) -> list:
    trajectories = []
    for f in sorted(input_dir.glob("*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                data = json.load(fh)
                if isinstance(data, list):
                    trajectories.extend(data)
                else:
                    trajectories.append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"  Warning: skipping {f}: {e}", file=sys.stderr)
    return trajectories


def main():
    parser = argparse.ArgumentParser(description="Zilli-Evolve: Skill进化引擎")
    parser.add_argument("--input", type=str, required=True, help="轨迹数据目录（JSON文件）")
    parser.add_argument("--target-skills", type=str, required=True, help="目标Skill目录")
    parser.add_argument("--reflection-model", type=str, default="claude-opus-4.6")
    parser.add_argument("--max-iterations", type=int, default=10)
    args = parser.parse_args()

    engine = SkillEvolutionEngine(
        reflection_model=args.reflection_model,
    )
    engine.max_iterations = args.max_iterations

    input_dir = Path(args.input)
    if not input_dir.exists():
        print(f"Error: input dir not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    trajectory_data = _load_trajectories(input_dir)
    print(f"Loaded {len(trajectory_data)} trajectory records from {args.input}")

    skills_dir = Path(args.target_skills)
    if not skills_dir.exists():
        print(f"Error: skills dir not found: {args.target_skills}", file=sys.stderr)
        sys.exit(1)

    for skill_file in sorted(skills_dir.glob("*.py")):
        print(f"Evolving: {skill_file}")
        pr = engine.evolve(str(skill_file), trajectory_data=trajectory_data)
        print(pr)
        print("---")

    print(f"Evolution complete. Processed {len(trajectory_data)} trajectories.")
