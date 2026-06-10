import logging
import time
import json
import math
import random
from pathlib import Path
from typing import List, Dict, Optional, Callable, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from enum import Enum, auto

logger = logging.getLogger("zilli.arena")


class ArenaStatus(Enum):
    CHAMPION = auto()
    CHALLENGER = auto()
    CONTENDER = auto()
    RETIRED = auto()


@dataclass
class ArenaModel:
    name: str
    version: str
    status: ArenaStatus
    deployed_at: float = 0.0
    retired_at: Optional[float] = None
    metrics: Dict = field(default_factory=dict)


@dataclass
class ArenaMatch:
    match_id: str
    timestamp: float
    champion: str
    challenger: str
    champion_score: float
    challenger_score: float
    champion_std: float = 0.0
    challenger_std: float = 0.0
    num_tasks: int = 0
    p_value: float = 1.0
    significant: bool = False
    winner: Optional[str] = None
    effect_size: float = 0.0


class ChampionChallenger:
    def __init__(
        self,
        min_win_gap: float = 0.05,
        significance_level: float = 0.05,
        min_eval_tasks: int = 10,
        warmup_rounds: int = 3,
        history_size: int = 20,
        log_dir: str = "",
        deploy_callback: Optional[Callable] = None,
        rollback_callback: Optional[Callable] = None,
    ):
        self.min_win_gap = min_win_gap
        self.significance_level = significance_level
        self.min_eval_tasks = min_eval_tasks
        self.warmup_rounds = warmup_rounds
        self.deploy_callback = deploy_callback
        self.rollback_callback = rollback_callback
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "arena_logs"

        self._models: Dict[str, ArenaModel] = {}
        self._champion: Optional[str] = None
        self._matches: List[ArenaMatch] = []
        self._recent_scores: Dict[str, deque] = {}
        self._total_matches = 0
        self._champion_wins = 0
        self._challenger_wins = 0
        self._deployments = 0
        self._rollbacks = 0

    def register_model(self, name: str, version: str,
                       status: ArenaStatus = ArenaStatus.CONTENDER):
        now = time.time()
        model = ArenaModel(
            name=name,
            version=version,
            status=status,
            deployed_at=now if status == ArenaStatus.CHAMPION else 0.0,
        )
        self._models[name] = model
        self._recent_scores.setdefault(name, deque(maxlen=100))

        if status == ArenaStatus.CHAMPION:
            self._champion = name
            logger.info("Champion registered: %s (v%s)", name, version)

        logger.info("Model registered: %s (v%s) status=%s", name, version, status.name)

    def add_score(self, model_name: str, score: float):
        if model_name in self._recent_scores:
            self._recent_scores[model_name].append(score)
        if model_name in self._models:
            old = self._models[model_name].metrics.get("avg_score", 0.0)
            n = len(self._recent_scores[model_name])
            self._models[model_name].metrics["avg_score"] = (
                (old * (n - 1) + score) / n if n > 0 else score
            )

    def run_match(self, challenger_name: str,
                  eval_fn: Callable[[str], List[float]]) -> Optional[ArenaMatch]:
        if self._champion is None:
            logger.warning("No champion registered, cannot run match")
            return None
        if challenger_name not in self._models:
            logger.warning("Challenger %s not registered", challenger_name)
            return None

        champion_scores = eval_fn(self._champion)
        challenger_scores = eval_fn(challenger_name)

        for s in champion_scores:
            self.add_score(self._champion, s)
        for s in challenger_scores:
            self.add_score(challenger_name, s)

        match = self._evaluate_match(
            champion=self._champion,
            challenger=challenger_name,
            champion_scores=champion_scores,
            challenger_scores=challenger_scores,
        )

        self._matches.append(match)
        self._total_matches += 1

        if match.winner == challenger_name:
            self._challenger_wins += 1
            if self._should_deploy(match):
                self._promote_challenger(challenger_name)
        else:
            self._champion_wins += 1

        self._log_match(match)
        return match

    def _evaluate_match(self, champion: str, challenger: str,
                        champion_scores: List[float],
                        challenger_scores: List[float]) -> ArenaMatch:
        n = min(len(champion_scores), len(challenger_scores))
        if n == 0:
            return ArenaMatch(
                match_id=f"{champion}_vs_{challenger}_{int(time.time())}",
                timestamp=time.time(),
                champion=champion,
                challenger=challenger,
                champion_score=0.0,
                challenger_score=0.0,
                num_tasks=0,
            )

        c_mean = sum(champion_scores[:n]) / n
        ch_mean = sum(challenger_scores[:n]) / n
        c_std = self._std(champion_scores[:n], c_mean)
        ch_std = self._std(challenger_scores[:n], ch_mean)

        effect = (ch_mean - c_mean)
        p_value = self._bootstrap_p(champion_scores[:n], challenger_scores[:n])
        significant = p_value < self.significance_level and abs(effect) > self.min_win_gap

        winner = None
        if significant:
            winner = challenger if effect > 0 else champion

        return ArenaMatch(
            match_id=f"{champion}_vs_{challenger}_{int(time.time())}",
            timestamp=time.time(),
            champion=champion,
            challenger=challenger,
            champion_score=round(c_mean, 4),
            challenger_score=round(ch_mean, 4),
            champion_std=round(c_std, 4),
            challenger_std=round(ch_std, 4),
            num_tasks=n,
            p_value=round(p_value, 6),
            significant=significant,
            winner=winner,
            effect_size=round(effect, 4),
        )

    def _bootstrap_p(self, a: List[float], b: List[float],
                     n_iter: int = 10000) -> float:
        combined = a + b
        n_a, n_b = len(a), len(b)
        observed = sum(b) / n_b - sum(a) / n_a
        count_extreme = 0

        for _ in range(n_iter):
            random.shuffle(combined)
            perm_a = combined[:n_a]
            perm_b = combined[n_a:]
            perm_diff = sum(perm_b) / n_b - sum(perm_a) / n_a
            if abs(perm_diff) >= abs(observed):
                count_extreme += 1

        return (count_extreme + 1) / (n_iter + 1)

    def _should_deploy(self, match: ArenaMatch) -> bool:
        if not match.significant:
            return False
        if match.winner != match.challenger:
            return False

        champion_matches = [m for m in self._matches
                            if m.champion == self._champion]
        if len(champion_matches) < self.warmup_rounds:
            return False

        recent = [m for m in self._matches[-5:]
                  if m.challenger == match.challenger]
        wins = sum(1 for m in recent if m.winner == match.challenger)
        if len(recent) >= 2 and wins < len(recent) * 0.6:
            return False

        return True

    def _promote_challenger(self, challenger_name: str):
        old_champion = self._champion
        challenger = self._models.get(challenger_name)
        if challenger is None:
            return

        if old_champion and old_champion in self._models:
            self._models[old_champion].status = ArenaStatus.RETIRED
            self._models[old_champion].retired_at = time.time()

        challenger.status = ArenaStatus.CHAMPION
        challenger.deployed_at = time.time()
        self._champion = challenger_name
        self._deployments += 1

        logger.info(
            "Champion promoted: %s (replaces %s) | deployments=%d",
            challenger_name, old_champion, self._deployments,
        )

        if self.deploy_callback:
            try:
                self.deploy_callback(challenger_name)
            except Exception as e:
                logger.error("Deploy callback failed: %s", e)

    def rollback(self) -> Optional[str]:
        if not self._matches:
            return None

        retired = [
            (n, m) for n, m in self._models.items()
            if m.status == ArenaStatus.RETIRED
        ]
        if not retired:
            logger.warning("No retired champion to roll back to")
            return None

        retired.sort(key=lambda x: x[1].retired_at or 0, reverse=True)
        target_name, target_model = retired[0]

        if self._champion and self._champion in self._models:
            self._models[self._champion].status = ArenaStatus.RETIRED
            self._models[self._champion].retired_at = time.time()

        target_model.status = ArenaStatus.CHAMPION
        target_model.deployed_at = time.time()
        old_champion = self._champion
        self._champion = target_name
        self._rollbacks += 1

        logger.info(
            "Rollback: %s → %s | rollbacks=%d",
            old_champion, target_name, self._rollbacks,
        )

        if self.rollback_callback:
            try:
                self.rollback_callback(target_name)
            except Exception as e:
                logger.error("Rollback callback failed: %s", e)

        return target_name

    def get_champion(self) -> Optional[str]:
        return self._champion

    def leaderboard(self) -> List[Dict]:
        entries = []
        for name, model in self._models.items():
            total_matches = sum(
                1 for m in self._matches
                if m.champion == name or m.challenger == name
            )
            wins = sum(
                1 for m in self._matches
                if m.winner == name
            )
            recent_scores = list(self._recent_scores.get(name, []))
            entries.append({
                "name": name,
                "version": model.version,
                "status": model.status.name,
                "avg_score": model.metrics.get("avg_score", 0.0),
                "total_matches": total_matches,
                "wins": wins,
                "win_rate": round(wins / total_matches, 3) if total_matches > 0 else 0.0,
                "recent_scores": recent_scores[-10:],
                "deployed_at": model.deployed_at,
            })
        entries.sort(key=lambda x: x["avg_score"], reverse=True)
        return entries

    def match_history(self, limit: int = 10) -> List[Dict]:
        return [
            {
                "match_id": m.match_id,
                "timestamp": m.timestamp,
                "champion": m.champion,
                "challenger": m.challenger,
                "champion_score": m.champion_score,
                "challenger_score": m.challenger_score,
                "p_value": m.p_value,
                "significant": m.significant,
                "winner": m.winner,
                "effect_size": m.effect_size,
            }
            for m in self._matches[-limit:]
        ]

    def stats(self) -> Dict:
        return {
            "current_champion": self._champion,
            "total_models": len(self._models),
            "total_matches": self._total_matches,
            "champion_wins": self._champion_wins,
            "challenger_wins": self._challenger_wins,
            "deployments": self._deployments,
            "rollbacks": self._rollbacks,
            "min_win_gap": self.min_win_gap,
            "significance_level": self.significance_level,
        }

    def _std(self, values: List[float], mean: float) -> float:
        if len(values) < 2:
            return 0.0
        var = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(var) if var > 0 else 0.0

    def _log_match(self, match: ArenaMatch):
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / "arena_matches.jsonl"
        entry = {
            "match_id": match.match_id,
            "timestamp": match.timestamp,
            "champion": match.champion,
            "challenger": match.challenger,
            "champion_score": match.champion_score,
            "challenger_score": match.challenger_score,
            "p_value": match.p_value,
            "significant": match.significant,
            "winner": match.winner,
            "effect_size": match.effect_size,
            "num_tasks": match.num_tasks,
        }
        with open(log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")


__all__ = [
    "ChampionChallenger", "ArenaMatch", "ArenaModel", "ArenaStatus",
]
