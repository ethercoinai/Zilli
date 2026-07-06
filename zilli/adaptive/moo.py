from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger("zilli.adaptive.moo")


@dataclass
class CandidateSolution:
    model_name: str
    objectives: dict[str, float]
    constraints_met: bool = True
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def objective_vector(self) -> np.ndarray:
        return np.array(list(self.objectives.values()))


@dataclass
class ParetoFront:
    solutions: list[CandidateSolution]
    objective_names: list[str]

    def dominance_count(self) -> int:
        return len(self.solutions)

    def quality_spread(self) -> float:
        if len(self.solutions) < 2:
            return 0.0
        vectors = np.array([s.objective_vector for s in self.solutions])
        diffs = np.diff(vectors, axis=0)
        return float(np.mean(np.abs(diffs)))


@dataclass
class OptimizationResult:
    pareto_front: ParetoFront
    iterations: int
    dominated_count: int
    best_solution: Optional[CandidateSolution] = None
    convergence_history: list[float] = field(default_factory=list)


ObjectiveFunc = Callable[[CandidateSolution], float]
ConstraintFunc = Callable[[CandidateSolution], bool]


class MultiObjectiveOptimizer:
    def __init__(
        self,
        objective_names: list[str],
        objectives_to_minimize: Optional[list[str]] = None,
        population_size: int = 20,
        max_generations: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.7,
    ):
        self.objective_names = objective_names
        self.minimize_set = set(objectives_to_minimize or [])
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self._objective_fns: dict[str, Callable[[CandidateSolution], float]] = {}
        self._constraints: list[ConstraintFunc] = []
        self._history: list[OptimizationResult] = []

    def register_objective(self, name: str, fn: Callable[[CandidateSolution], float], minimize: bool = False):
        self._objective_fns[name] = fn
        if minimize:
            self.minimize_set.add(name)
        elif name in self.minimize_set:
            self.minimize_set.discard(name)

    def add_constraint(self, fn: ConstraintFunc):
        self._constraints.append(fn)

    def _dominates(self, a: CandidateSolution, b: CandidateSolution) -> bool:
        a_obj, b_obj = a.objectives, b.objectives
        better_or_equal = True
        strictly_better = False
        for name in self.objective_names:
            a_val = a_obj.get(name, 0.0)
            b_val = b_obj.get(name, 0.0)
            if name in self.minimize_set:
                if a_val > b_val:
                    better_or_equal = False
                    break
                if a_val < b_val:
                    strictly_better = True
            else:
                if a_val < b_val:
                    better_or_equal = False
                    break
                if a_val > b_val:
                    strictly_better = True
        return better_or_equal and strictly_better

    def _fast_non_dominated_sort(self, population: list[CandidateSolution]) -> list[list[int]]:
        n = len(population)
        domination_count = [0] * n
        dominated_set: list[list[int]] = [[] for _ in range(n)]
        fronts: list[list[int]] = [[]]

        for i in range(n):
            for j in range(i + 1, n):
                if self._dominates(population[i], population[j]):
                    dominated_set[i].append(j)
                    domination_count[j] += 1
                elif self._dominates(population[j], population[i]):
                    dominated_set[j].append(i)
                    domination_count[i] += 1

        for i in range(n):
            if domination_count[i] == 0:
                fronts[0].append(i)

        current_front = 0
        while fronts[current_front]:
            next_front = []
            for i in fronts[current_front]:
                for j in dominated_set[i]:
                    domination_count[j] -= 1
                    if domination_count[j] == 0:
                        next_front.append(j)
            current_front += 1
            if next_front:
                fronts.append(next_front)
            else:
                break

        return fronts

    def _crowding_distance(self, population: list[CandidateSolution], front_indices: list[int]) -> dict[int, float]:
        distances: dict[int, float] = {i: 0.0 for i in front_indices}
        if len(front_indices) <= 2:
            for i in front_indices:
                distances[i] = float("inf")
            return distances

        for obj_name in self.objective_names:
            sorted_indices = sorted(front_indices, key=lambda i: population[i].objectives.get(obj_name, 0.0))
            distances[sorted_indices[0]] = float("inf")
            distances[sorted_indices[-1]] = float("inf")

            obj_vals = [population[i].objectives.get(obj_name, 0.0) for i in sorted_indices]
            obj_range = obj_vals[-1] - obj_vals[0]
            if obj_range == 0:
                continue

            for k in range(1, len(sorted_indices) - 1):
                idx = sorted_indices[k]
                distances[idx] += (obj_vals[k + 1] - obj_vals[k - 1]) / obj_range

        return distances

    def _evaluate_solution(self, sol: CandidateSolution) -> CandidateSolution:
        for name, fn in self._objective_fns.items():
            sol.objectives[name] = fn(sol)
        sol.constraints_met = all(c(sol) for c in self._constraints)
        return sol

    def optimize(
        self,
        candidates: list[CandidateSolution],
        max_generations: Optional[int] = None,
    ) -> OptimizationResult:
        gen = max_generations or self.max_generations
        population = list(candidates)[:self.population_size]

        while len(population) < self.population_size:
            population.append(CandidateSolution(
                model_name=f"random_{len(population)}",
                objectives={n: random.random() for n in self.objective_names},
            ))

        for sol in population:
            self._evaluate_solution(sol)

        convergence: list[float] = []

        for g in range(gen):
            fronts = self._fast_non_dominated_sort(population)
            new_population: list[CandidateSolution] = []

            for front in fronts:
                cd = self._crowding_distance(population, front)
                sorted_front = sorted(front, key=lambda i: (-cd[i],))
                for idx in sorted_front:
                    if len(new_population) < self.population_size:
                        new_population.append(population[idx])

            while len(new_population) < self.population_size:
                new_population.append(random.choice(population))

            offspring: list[CandidateSolution] = []
            while len(offspring) < self.population_size:
                p1, p2 = random.sample(new_population, 2)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                child = self._evaluate_solution(child)
                offspring.append(child)

            combined = population + offspring
            fronts = self._fast_non_dominated_sort(combined)
            final_pop: list[CandidateSolution] = []
            for front in fronts:
                cd = self._crowding_distance(combined, front)
                sorted_front = sorted(front, key=lambda i: (-cd[i],))
                for idx in sorted_front:
                    if len(final_pop) < self.population_size:
                        final_pop.append(combined[idx])

            population = final_pop
            cur_fronts = self._fast_non_dominated_sort(population)
            pareto_front = [population[i] for i in cur_fronts[0]] if cur_fronts else []
            hypervolume = self._approx_hypervolume(pareto_front)
            convergence.append(hypervolume)

        final_fronts = self._fast_non_dominated_sort(population)
        pareto_indices = final_fronts[0] if final_fronts else []
        pareto_solutions = [population[i] for i in pareto_indices]

        best = min(pareto_solutions, key=lambda s: sum(s.objective_vector)) if pareto_solutions else None

        result = OptimizationResult(
            pareto_front=ParetoFront(solutions=pareto_solutions, objective_names=self.objective_names),
            iterations=gen,
            dominated_count=len(population) - len(pareto_solutions),
            best_solution=best,
            convergence_history=convergence,
        )
        self._history.append(result)
        return result

    def _crossover(self, p1: CandidateSolution, p2: CandidateSolution) -> CandidateSolution:
        if random.random() > self.crossover_rate:
            return CandidateSolution(
                model_name=p1.model_name,
                objectives=dict(p1.objectives),
            )
        child_obj = {}
        for name in self.objective_names:
            child_obj[name] = p1.objectives.get(name, 0.0) if random.random() < 0.5 else p2.objectives.get(name, 0.0)
        return CandidateSolution(
            model_name=random.choice([p1.model_name, p2.model_name]),
            objectives=child_obj,
        )

    def _mutate(self, sol: CandidateSolution) -> CandidateSolution:
        mutated_obj = dict(sol.objectives)
        for name in self.objective_names:
            if random.random() < self.mutation_rate:
                mutated_obj[name] *= (1 + random.gauss(0, 0.1))
        return CandidateSolution(
            model_name=sol.model_name,
            objectives=mutated_obj,
            metadata=dict(sol.metadata),
        )

    def _approx_hypervolume(self, front: list[CandidateSolution], reference: Optional[np.ndarray] = None) -> float:
        if not front:
            return 0.0
        if reference is None:
            reference = np.ones(len(self.objective_names)) * 1e6
        vol = 0.0
        for sol in front:
            diff = reference - sol.objective_vector
            if np.all(diff > 0):
                vol += float(np.prod(diff))
        return vol

    def select_for_task(
        self,
        task_features: dict[str, Any],
        candidates: list[CandidateSolution],
        weights: Optional[dict[str, float]] = None,
    ) -> CandidateSolution:
        if not candidates:
            raise ValueError("No candidates provided")

        if weights is None:
            weights = {n: 1.0 / len(self.objective_names) for n in self.objective_names}

        scored = []
        for c in candidates:
            score = 0.0
            for name in self.objective_names:
                sign = -1.0 if name not in self.minimize_set else 1.0
                score += sign * weights.get(name, 0.0) * c.objectives.get(name, 0.0)
            scored.append((score, c))

        scored.sort(key=lambda x: x[0])
        return scored[0][1]


__all__ = [
    "CandidateSolution", "ParetoFront", "OptimizationResult",
    "MultiObjectiveOptimizer",
]
