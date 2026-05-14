import pytest
import asyncio
import yaml
from pathlib import Path

from zilli.schema.actions import (
    BaseAction, MemoryWriteAction, MemoryReadAction,
    SkillCreateAction, BashRunAction, FileWriteAction, FinishAction,
)
from zilli.envs import HermesSandbox
from zilli.data import TrajectoryStore
from zilli.tasks import load_tasks, TaskRunner
from zilli.training.cispo import CISPO_Trainer
from zilli.training.grpo import GRPO_Trainer
from zilli.training.rl_trainer import RLTrainer
from zilli.rewards import VerifiableReward
from zilli.infra import LengthElasticController
from zilli.evolution import SkillEvolutionEngine
from zilli.learner import ContinuousLearner


class TestSchema:
    def test_base_action_strict(self):
        with pytest.raises(ValueError):
            BaseAction(action_id="1", reasoning="test", tool_name="x", extra_field="nope")

    def test_memory_write_action(self):
        a = MemoryWriteAction(action_id="1", reasoning="store", key="k", value="v")
        d = a.model_dump()
        assert d["tool_name"] == "memory_write"
        assert d["key"] == "k"
        assert d["value"] == "v"


class TestSandbox:
    @pytest.mark.asyncio
    async def test_memory_rw(self):
        s = HermesSandbox()
        r1 = await s.step(MemoryWriteAction(action_id="1", reasoning="store", key="name", value="Hermes"))
        assert r1["reward"] > 0
        r2 = await s.step(MemoryReadAction(action_id="2", reasoning="recall", key="name"))
        assert r2["observation"]["value"] == "Hermes"

    @pytest.mark.asyncio
    async def test_finish_triggers_done(self):
        s = HermesSandbox()
        r = await s.step(FinishAction(action_id="3", reasoning="done", summary="ok"))
        assert r["done"] is True

    @pytest.mark.asyncio
    async def test_unknown_tool(self):
        class BadAction(BaseAction):
            tool_name: str = "nonexistent"
        s = HermesSandbox()
        r = await s.step(BadAction(action_id="bad", reasoning="oops", tool_name="nonexistent"))
        assert r["reward"] == -1.0

    @pytest.mark.asyncio
    async def test_skill_create(self):
        s = HermesSandbox()
        r = await s.step(SkillCreateAction(
            action_id="s1", reasoning="create skill",
            name="test_skill", code="def foo(): pass",
        ))
        assert r["observation"]["success"]

    @pytest.mark.asyncio
    async def test_bash_run(self):
        s = HermesSandbox()
        r = await s.step(BashRunAction(
            action_id="b1", reasoning="run cmd",
            command="echo hello",
        ))
        assert r["observation"]["success"]

    @pytest.mark.asyncio
    async def test_trajectory_recording(self):
        s = HermesSandbox()
        await s.step(MemoryWriteAction(action_id="1", reasoning="a", key="k", value="v"))
        await s.step(MemoryReadAction(action_id="2", reasoning="b", key="k"))
        assert len(s.get_trajectory()) == 2


class TestTrajectoryStore:
    def test_add_golden(self):
        store = TrajectoryStore()
        store.add_trajectory([{"step": 1}], 0.9)
        assert len(store.golden_trajectories) == 1

    def test_add_failure(self):
        store = TrajectoryStore()
        store.add_trajectory([{"step": 1, "observation": {"error": "fail"}}], 0.1)
        assert len(store.failure_trajectories) == 1

    def test_sample_batch(self):
        store = TrajectoryStore()
        for i in range(10):
            store.add_trajectory([{"step": i}], 0.9 + i * 0.01)
        for i in range(10):
            store.add_trajectory([{"step": i, "observation": {"error": "x"}}], 0.1)
        batch = store.sample_batch(8, golden_ratio=0.5)
        assert len(batch) == 8

    def test_purify(self):
        store = TrajectoryStore()
        store.add_trajectory([{"observation": {"error": "contaminated data"}}], 0.9)
        store.add_trajectory([{"observation": {"success": True}}], 0.95)
        count = store.purify()
        assert count >= 1


class TestTasks:
    def test_load_all_tasks(self):
        tasks = load_tasks()
        assert len(tasks) >= 4

    def test_load_basic(self):
        tasks = load_tasks(category="basic")
        assert len(tasks) >= 4

    def test_load_benchmark(self):
        tasks = load_tasks(category="benchmark")
        assert len(tasks) >= 3

    def test_task_runner_truncate(self):
        task = load_tasks(category="basic")[0]
        runner = TaskRunner(task)
        assert not runner.should_truncate()

    def test_task_runner_evaluate(self):
        task = load_tasks(category="basic")[0]
        runner = TaskRunner(task)
        score = runner.evaluate({"task_completed": True, "memory_recalled": True})
        assert 0 <= score <= 1.0


class TestTraining:
    def test_cispo_loss(self):
        trainer = CISPO_Trainer({"clip_range": 0.2, "kl_penalty": 0.01})
        trajs = [
            {"log_prob": -0.5, "old_log_prob": -0.6},
            {"log_prob": -1.0, "old_log_prob": -0.9},
        ]
        advs = [1.0, -0.5]
        metrics = trainer.compute_loss(trajs, advs)
        assert "loss" in metrics
        assert "kl" in metrics

    def test_cispo_advantages(self):
        trainer = CISPO_Trainer({"gamma": 0.99})
        advs = trainer.compute_advantages([1.0, 0.5, 0.0], [False, False, True])
        assert len(advs) == 3

    def test_grpo_advantages(self):
        trainer = GRPO_Trainer({})
        group = [{"reward": 1.0}, {"reward": 0.5}, {"reward": 0.0}]
        advs = trainer.compute_advantages(group)
        assert len(advs) == 3

    def test_rl_trainer_cispo(self):
        config = {"algorithm": "CISPO"}
        trainer = RLTrainer(config)
        batch = [
            {"log_prob": -1.0, "old_log_prob": -1.1, "reward": 1.0, "done": False},
            {"log_prob": -2.0, "old_log_prob": -1.9, "reward": 0.0, "done": True},
        ]
        metrics = trainer.update(batch)
        assert "loss" in metrics

    def test_rl_trainer_grpo(self):
        config = {"algorithm": "GRPO"}
        trainer = RLTrainer(config)
        batch = [
            {"log_prob": -1.0, "old_log_prob": -1.1, "reward": 1.0},
            {"log_prob": -2.0, "old_log_prob": -1.9, "reward": 0.0},
        ]
        metrics = trainer.update(batch)
        assert "loss" in metrics


class TestRewards:
    def test_verifiable_reward_success(self):
        rw = VerifiableReward()
        traj = [
            MemoryWriteAction(action_id="1", reasoning="a", key="k", value="v"),
            FinishAction(action_id="2", reasoning="b", summary="ok"),
        ]
        score = rw.compute(traj, {"task_completed": True})
        assert score > 0

    def test_verifiable_reward_forbidden(self):
        rw = VerifiableReward()
        score = rw.compute([], {"forbidden_action_executed": True})
        assert score < 0


class TestInfra:
    def test_length_controller_no_change(self):
        lc = LengthElasticController()
        lc.adapt([100, 200, 300])
        assert lc.current_cap == 8192

    def test_length_controller_expand(self):
        lc = LengthElasticController()
        lc.adapt([8000, 8100, 8200, 9000])
        assert lc.current_cap > 8192

    def test_length_controller_switch_to_mp(self):
        lc = LengthElasticController()
        lc.adapt([40000])
        assert lc.parallel_mode == "mp"


class TestEvolution:
    def test_skill_evolution(self):
        engine = SkillEvolutionEngine()
        pr = engine.evolve("test_skill.py", [{"observation": {"error": "bug"}}])
        assert "Auto-evolved" in pr


class TestLearner:
    def test_continuous_learner_init(self):
        store = TrajectoryStore()
        learner = ContinuousLearner(store, interval_hours=24)
        assert learner.interval == 24
