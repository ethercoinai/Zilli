from typing import List, Dict, Any, Set, Tuple


class TrajectoryCleaner:
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.max_dedup_similarity = self.config.get("max_dedup_similarity", 0.85)
        self.max_anomaly_std = self.config.get("max_anomaly_std", 2.0)
        self.min_trajectory_length = self.config.get("min_trajectory_length", 2)

    def clean(self, trajectory: List[Dict]) -> Tuple[List[Dict], List[str]]:
        steps = []
        warnings = []

        steps = self._remove_contaminated(trajectory, warnings)
        steps = self._repair_errors(steps, warnings)
        steps = self._deduplicate_steps(steps, warnings)

        return steps, warnings

    def validate(self, trajectory: List[Dict]) -> Dict[str, Any]:
        issues = []
        if len(trajectory) < self.min_trajectory_length:
            issues.append(f"trajectory too short ({len(trajectory)} < {self.min_trajectory_length})")

        if not trajectory:
            return {"valid": False, "issues": issues, "score": 0.0}

        tools_used = set()
        has_contamination = False
        error_count = 0
        for step in trajectory:
            action = step.get("action", {})
            tool = action.get("tool_name", "")
            if tool:
                tools_used.add(tool)
            obs = step.get("observation", {})
            if isinstance(obs, dict):
                if not obs.get("success", True):
                    error_count += 1
                err = obs.get("error", "")
                if "contaminated" in str(err).lower() or "corrupted" in str(err).lower():
                    has_contamination = True

        completeness = min(1.0, len(trajectory) / 10.0)
        tool_diversity = min(1.0, len(tools_used) / 3.0)
        error_penalty = max(0.0, 1.0 - (error_count / max(len(trajectory), 1)) * 2)
        score = (completeness * 0.4 + tool_diversity * 0.3 + error_penalty * 0.3)

        if has_contamination:
            issues.append("contamination detected")
            score *= 0.5
        if error_count > len(trajectory) * 0.5:
            issues.append("high error rate")

        return {
            "valid": len(issues) == 0 and score >= 0.3,
            "issues": issues,
            "score": round(score, 3),
            "tool_diversity": round(tool_diversity, 3),
            "error_count": error_count,
            "total_steps": len(trajectory),
        }

    def find_anomalies(self, trajectories: List[List[Dict]]) -> List[int]:
        if not trajectories:
            return []

        lengths = [len(t) for t in trajectories]
        if not lengths:
            return []

        mean = sum(lengths) / len(lengths)
        variance = sum((x - mean) ** 2 for x in lengths) / len(lengths)
        std = variance ** 0.5 if variance > 0 else 1.0

        threshold = self.max_anomaly_std * std
        anomalous = []
        for i, length in enumerate(lengths):
            if abs(length - mean) > threshold:
                anomalous.append(i)
        return anomalous

    def batch_clean(self, trajectories: List[List[Dict]]) -> Tuple[List[List[Dict]], List[str]]:
        cleaned = []
        all_warnings = []
        for traj in trajectories:
            c, w = self.clean(traj)
            cleaned.append(c)
            all_warnings.extend(w)
        deduped = self._batch_deduplicate(cleaned, all_warnings)
        return deduped, all_warnings

    def _remove_contaminated(self, trajectory: List[Dict], warnings: List[str]) -> List[Dict]:
        clean = []
        for i, step in enumerate(trajectory):
            obs = step.get("observation", {})
            if isinstance(obs, dict):
                err = obs.get("error", "")
                if "contaminated" in str(err).lower() or "corrupted" in str(err).lower():
                    warnings.append(f"step {i}: removed contaminated step")
                    continue
            clean.append(step)
        return clean

    def _repair_errors(self, trajectory: List[Dict], warnings: List[str]) -> List[Dict]:
        repaired = []
        for i, step in enumerate(trajectory):
            action = dict(step.get("action", {}))
            obs = step.get("observation", {})

            if isinstance(obs, dict) and not obs.get("success", True):
                err_msg = obs.get("error", "")
                if "not found" in str(err_msg).lower() and action.get("tool_name") == "file_read":
                    action["path"] = action.get("path", "") + "_recovered"
                    warnings.append(f"step {i}: repaired file_read path")
                    step = {"action": action, "observation": obs}

            repaired.append(step)
        return repaired

    def _deduplicate_steps(self, trajectory: List[Dict], warnings: List[str]) -> List[Dict]:
        if len(trajectory) < 2:
            return trajectory

        seen_signatures: Set[str] = set()
        deduped = []
        for i, step in enumerate(trajectory):
            action = step.get("action", {})
            tool = action.get("tool_name", "") if action else ""
            if not tool:
                deduped.append(step)
                continue
            sig = f"{tool}:{action.get('command','')}:{action.get('path','')}:{action.get('key','')}"
            if sig in seen_signatures:
                warnings.append(f"step {i}: removed duplicate ({sig[:50]})")
                continue
            seen_signatures.add(sig)
            deduped.append(step)
        return deduped

    def _batch_deduplicate(self, trajectories: List[List[Dict]], warnings: List[str]) -> List[List[Dict]]:
        if not trajectories:
            return trajectories

        seen_full: Set[str] = set()
        result = []
        for i, traj in enumerate(trajectories):
            sig = "|".join(
                s.get("action", {}).get("tool_name", "") + ":" + str(s.get("action", {}).get("command", ""))
                for s in traj[:5]
            )
            if sig in seen_full:
                warnings.append(f"trajectory {i}: removed duplicate trajectory")
                continue
            seen_full.add(sig)
            result.append(traj)
        return result


__all__ = ["TrajectoryCleaner"]
