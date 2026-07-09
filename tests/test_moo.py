import numpy as np

from zilli.adaptive.moo import (
    CandidateSolution,
    MultiObjectiveOptimizer,
)


class TestCandidateSolution:
    def test_objective_vector(self):
        sol = CandidateSolution(model_name="m1", objectives={"cost": 0.5, "quality": 0.8})
        vec = sol.objective_vector
        assert len(vec) == 2
        assert vec[0] == 0.5


class TestMultiObjectiveOptimizer:
    def test_init(self):
        opt = MultiObjectiveOptimizer(objective_names=["cost", "quality"])
        assert len(opt.objective_names) == 2

    def test_register_objective(self):
        opt = MultiObjectiveOptimizer(objective_names=["a", "b"])
        opt.register_objective("a", lambda s: 0.5, minimize=True)
        assert "a" in opt._objective_fns
        assert "a" in opt.minimize_set

    def test_add_constraint(self):
        opt = MultiObjectiveOptimizer(objective_names=["a"])
        opt.add_constraint(lambda s: True)
        assert len(opt._constraints) == 1

    def test_domination(self):
        opt = MultiObjectiveOptimizer(objective_names=["cost", "quality"], objectives_to_minimize=["cost"])
        a = CandidateSolution(model_name="a", objectives={"cost": 0.3, "quality": 0.9})
        b = CandidateSolution(model_name="b", objectives={"cost": 0.5, "quality": 0.7})
        assert opt._dominates(a, b)
        assert not opt._dominates(b, a)

    def test_non_dominated_sort(self):
        opt = MultiObjectiveOptimizer(objective_names=["cost", "quality"])
        pop = [
            CandidateSolution(model_name="a", objectives={"cost": 0.3, "quality": 0.9}),
            CandidateSolution(model_name="b", objectives={"cost": 0.5, "quality": 0.7}),
            CandidateSolution(model_name="c", objectives={"cost": 0.7, "quality": 0.5}),
        ]
        fronts = opt._fast_non_dominated_sort(pop)
        assert len(fronts) >= 1
        assert 0 in fronts[0]

    def test_optimize(self):
        opt = MultiObjectiveOptimizer(
            objective_names=["cost", "quality"],
            objectives_to_minimize=["cost"],
            population_size=10,
            max_generations=5,
        )
        candidates = [
            CandidateSolution(model_name=f"m{i}", objectives={"cost": np.random.random(), "quality": np.random.random()})
            for i in range(10)
        ]
        result = opt.optimize(candidates, max_generations=5)
        assert result.iterations == 5
        assert len(result.pareto_front.solutions) >= 1

    def test_select_for_task(self):
        opt = MultiObjectiveOptimizer(objective_names=["cost", "quality"])
        candidates = [
            CandidateSolution(model_name="cheap", objectives={"cost": 0.1, "quality": 0.5}),
            CandidateSolution(model_name="good", objectives={"cost": 0.8, "quality": 0.9}),
        ]
        selected = opt.select_for_task({}, candidates, weights={"cost": 0.5, "quality": 0.5})
        assert selected.model_name in ["cheap", "good"]

    def test_crossover(self):
        opt = MultiObjectiveOptimizer(objective_names=["a", "b"], crossover_rate=1.0)
        p1 = CandidateSolution(model_name="p1", objectives={"a": 0.3, "b": 0.7})
        p2 = CandidateSolution(model_name="p2", objectives={"a": 0.6, "b": 0.4})
        child = opt._crossover(p1, p2)
        assert len(child.objectives) == 2

    def test_mutation(self):
        opt = MultiObjectiveOptimizer(objective_names=["a"], mutation_rate=1.0)
        sol = CandidateSolution(model_name="s", objectives={"a": 1.0})
        mutated = opt._mutate(sol)
        assert mutated.objectives["a"] != 1.0

    def test_hypervolume(self):
        opt = MultiObjectiveOptimizer(objective_names=["a", "b"])
        front = [
            CandidateSolution(model_name="x", objectives={"a": 0.3, "b": 0.7}),
            CandidateSolution(model_name="y", objectives={"a": 0.6, "b": 0.4}),
        ]
        hv = opt._approx_hypervolume(front, reference=np.array([1.0, 1.0]))
        assert hv > 0


__all__ = ["TestCandidateSolution", "TestMultiObjectiveOptimizer"]
