import pytest
import asyncio
import yaml
from pathlib import Path

import math
from zilli.schema.actions import (
    BaseAction, MemoryWriteAction, MemoryReadAction,
    SkillCreateAction, BashRunAction, FileWriteAction, FinishAction,
)
from zilli.envs import HermesSandbox
from zilli.data import TrajectoryStore, TrajectoryCleaner
from zilli.tasks import load_tasks, TaskRunner
from zilli.training.cispo import CISPO_Trainer
from zilli.training.grpo import GRPO_Trainer
from zilli.training.rl_trainer import RLTrainer
from zilli.rewards import VerifiableReward
from zilli.infra import LengthElasticController, LayoutAwareDispatcher
from zilli.infra.async_scheduler import AsyncRolloutScheduler, RolloutResult, RolloutStatus
from zilli.run_training import TrainingExperiment
from zilli.evolution import SkillEvolutionEngine
from zilli.learner import ContinuousLearner
from zilli.training.distillation import DistillationScheduler, DistillationSample, DistillationCycle
from zilli.training.champion_challenger import (
    ChampionChallenger, ArenaMatch, ArenaModel, ArenaStatus,
)


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
        store.add_trajectory(
            [{"action": {"tool_name": "bash_run", "command": "echo 1"}},
             {"action": {"tool_name": "finish"}}],
            0.9,
        )
        assert len(store.golden_trajectories) == 1

    def test_add_failure(self):
        store = TrajectoryStore()
        store.add_trajectory(
            [{"action": {"tool_name": "bash_run", "command": "bad"},
              "observation": {"error": "fail", "success": False}},
             {"action": {"tool_name": "finish"},
              "observation": {"success": True}}],
            0.1,
        )
        assert len(store.failure_trajectories) == 1

    def test_sample_batch(self):
        store = TrajectoryStore()
        for i in range(10):
            store.add_trajectory(
                [{"action": {"tool_name": "bash_run", "command": f"echo {i}"}},
                 {"action": {"tool_name": "finish"}}],
                0.9 + i * 0.01,
            )
        for i in range(10):
            store.add_trajectory(
                [{"action": {"tool_name": "bash_run", "command": f"fail_{i}"},
                  "observation": {"error": f"err_{i}", "success": False}},
                 {"action": {"tool_name": "finish"}}],
                0.1,
            )
        batch = store.sample_batch(8, golden_ratio=0.5)
        assert len(batch) == 8

    def test_purify(self):
        store = TrajectoryStore()
        store.add_trajectory(
            [{"action": {"tool_name": "bash_run", "command": "clean"},
              "observation": {"success": True}},
             {"action": {"tool_name": "finish"}}],
            0.95,
        )
        store.add_trajectory(
            [{"action": {"tool_name": "bash_run", "command": "bad"},
              "observation": {"error": "contaminated", "success": False}},
             {"action": {"tool_name": "finish"},
              "observation": {"error": "contaminated too", "success": False}}],
            0.9,
        )
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


class TestTrajectoryCleaner:
    def test_clean_removes_contaminated(self):
        cleaner = TrajectoryCleaner()
        traj = [
            {"step": 1, "action": {"tool_name": "memory_write"}, "observation": {"success": True}},
            {"step": 2, "action": {"tool_name": "file_read"}, "observation": {"error": "contaminated data"}},
            {"step": 3, "action": {"tool_name": "finish"}, "observation": {"success": True}},
        ]
        cleaned, warnings = cleaner.clean(traj)
        assert len([s for s in cleaned if s["step"] == 2]) == 0
        assert any("contaminated" in w for w in warnings)

    def test_validate_short_trajectory(self):
        cleaner = TrajectoryCleaner()
        result = cleaner.validate([])
        assert result["valid"] is False

    def test_find_anomalies(self):
        cleaner = TrajectoryCleaner({"max_anomaly_std": 1.0})
        trajs = [
            [{"step": i} for i in range(5)],
            [{"step": i} for i in range(7)],
            [{"step": i} for i in range(200)],
        ]
        anomalous = cleaner.find_anomalies(trajs)
        assert 2 in anomalous

    def test_deduplicate_batch(self):
        cleaner = TrajectoryCleaner()
        traj_a = [{"action": {"tool_name": "memory_write", "command": "write A"}},
                  {"action": {"tool_name": "finish", "command": ""}}]
        traj_b = [{"action": {"tool_name": "memory_write", "command": "write A"}},
                  {"action": {"tool_name": "finish", "command": ""}}]
        traj_c = [{"action": {"tool_name": "bash_run", "command": "run B"}},
                  {"action": {"tool_name": "finish", "command": ""}}]
        deduped, warnings = cleaner.batch_clean([traj_a, traj_b, traj_c])
        assert len(deduped) == 2


class TestSandboxEnhanced:
    @pytest.mark.asyncio
    async def test_web_search(self):
        s = HermesSandbox()
        r = await s.step({
            "action_id": "w1", "reasoning": "search",
            "tool_name": "web_search", "query": "test",
        })
        assert r["observation"]["success"]

    @pytest.mark.asyncio
    async def test_code_interpreter(self):
        s = HermesSandbox()
        r = await s.step({
            "action_id": "c1", "reasoning": "run code",
            "tool_name": "code_interpreter", "code": "print('hello')",
            "language": "python",
        })
        assert r["observation"]["success"]

    @pytest.mark.asyncio
    async def test_error_probability(self):
        s = HermesSandbox(scenario={"error_probability": 1.0})
        r = await s.step({
            "action_id": "e1", "reasoning": "store",
            "tool_name": "memory_write", "key": "k", "value": "v",
        })
        assert r["observation"]["success"] is False

    @pytest.mark.asyncio
    async def test_max_turns(self):
        s = HermesSandbox(scenario={"max_turns": 2})
        r1 = await s.step(MemoryWriteAction(action_id="1", reasoning="a", key="k1", value="v1"))
        assert r1["done"] is False
        r2 = await s.step(MemoryWriteAction(action_id="2", reasoning="b", key="k2", value="v2"))
        assert r2["done"] is True

    @pytest.mark.asyncio
    async def test_scenario_initial_files(self):
        s = HermesSandbox(scenario={"initial_files": {"/test.txt": "hello"}})
        r = await s.step({
            "action_id": "f1", "reasoning": "read",
            "tool_name": "file_read", "path": "/test.txt",
        })
        assert r["observation"]["content"] == "hello"

    def test_get_stats(self):
        s = HermesSandbox()
        stats = s.get_stats()
        assert "turns" in stats
        assert "trajectory_length" in stats


class TestTrajectoryStore2:
    def test_add_neutral_trajectory(self):
        store = TrajectoryStore()
        store.add_trajectory([{"step": 1}, {"step": 2}], 0.5)
        assert len(store.rollout_buffer) == 1

    def test_priority_sample(self):
        store = TrajectoryStore()
        for i in range(10):
            store.add_trajectory([{"step": i}, {"step": i + 1}], 0.9 + i * 0.01)
        for i in range(10):
            store.add_trajectory(
                [{"step": i, "observation": {"error": "fail"}},
                 {"step": i + 1, "observation": {"error": "fail"}}],
                0.1,
            )
        batch = store.sample_batch(8, use_priority=True)
        assert len(batch) == 8

    def test_augment_batch(self):
        store = TrajectoryStore()
        store.add_trajectory([{"step": 1}, {"step": 2}, {"step": 3}, {"step": 4}], 0.9)
        batch = store.sample_batch(1)
        aug = store.augment_batch(batch)
        assert len(aug) >= 1

    def test_stats_extended(self):
        store = TrajectoryStore()
        store.add_trajectory([{"step": 1}, {"step": 2}], 0.9)
        stats = store.stats()
        assert "avg_golden_reward" in stats
        assert "avg_trajectory_length" in stats


class TestInfra2:
    def test_layout_aware_dispatch(self):
        dispatcher = LayoutAwareDispatcher()
        data = [1, 2, 3, 4, 5]
        chunks = dispatcher.dispatch(data, 3)
        assert len(chunks) == 3
        assert sum(len(c) for c in chunks) == 5

    def test_layout_aware_dispatch_sorted(self):
        dispatcher = LayoutAwareDispatcher()
        data = ["a", "b", "c", "d"]
        lengths = [100, 50, 200, 10]
        chunks = dispatcher.dispatch_layout_aware(data, 2, lengths)
        assert len(chunks) == 2
        assert sum(len(c) for c in chunks) == 4

    def test_length_controller_shrink(self):
        lc = LengthElasticController()
        lc.current_cap = 32000
        lc.adapt([100, 200, 300, 400, 500] * 10)
        assert lc.current_cap < 32000

    def test_length_controller_stats(self):
        lc = LengthElasticController()
        lc.adapt([1000, 2000, 3000])
        stats = lc.get_stats()
        assert "p50_length" in stats
        assert "current_cap" in stats

    def test_training_experiment_init(self):
        exp = TrainingExperiment("test", {"lr": 0.01}, "/tmp/zilli_test_exp")
        assert exp.name == "test"
        assert exp.best_reward == float("-inf")

    def test_training_experiment_log_epoch(self):
        exp = TrainingExperiment("test2", {}, "/tmp/zilli_test_exp")
        exp.log_epoch(0, {"loss": 0.5})
        assert len(exp.metrics) == 1

    def test_training_experiment_summary(self):
        exp = TrainingExperiment("test3", {}, "/tmp/zilli_test_exp")
        s = exp.summary()
        assert s["name"] == "test3"
        assert s["epochs"] == 0


class TestTraining2:
    def test_cispo_entropy(self):
        trainer = CISPO_Trainer({"entropy_coef": 0.01})
        trajs = [
            {"log_prob": -0.5, "old_log_prob": -0.6},
            {"log_prob": -1.0, "old_log_prob": -0.9},
        ]
        metrics = trainer.compute_loss(trajs, [1.0, -0.5])
        assert "entropy" in metrics
        assert metrics["entropy"] >= 0

    def test_cispo_advantage_normalization(self):
        trainer = CISPO_Trainer({})
        trajs = [{"log_prob": -0.5, "old_log_prob": -0.6},
                 {"log_prob": -1.0, "old_log_prob": -0.9}]
        metrics = trainer.compute_loss(trajs, [100.0, -100.0])
        assert abs(metrics["mean_advantage"]) < 1e-6

    def test_cispo_value_loss(self):
        trainer = CISPO_Trainer({"vf_coef": 0.5})
        trajs = [
            {"log_prob": -0.5, "old_log_prob": -0.6, "value": 0.5},
            {"log_prob": -1.0, "old_log_prob": -0.9, "value": 0.3},
        ]
        metrics = trainer.compute_loss(trajs, [1.0, -0.5])
        assert "value_loss" in metrics

    def test_cispo_gae(self):
        trainer = CISPO_Trainer({"gamma": 0.99, "gae_lambda": 0.95})
        advs = trainer.compute_gae_advantages(
            [1.0, 0.5, 0.0], [0.8, 0.6, 0.4], [False, False, True]
        )
        assert len(advs) == 3


class TestRewards2:
    def test_template_match_perfect(self):
        rw = VerifiableReward()
        traj = [
            {"action": {"tool_name": "memory_write"}},
            {"action": {"tool_name": "finish"}},
        ]
        template = [
            {"tool": "memory_write", "reward_weight": 1.0},
            {"tool": "finish", "reward_weight": 1.0},
        ]
        score = rw.compute_template_match(traj, template)
        assert score == 1.0

    def test_template_match_partial(self):
        rw = VerifiableReward()
        traj = [
            {"action": {"tool_name": "bash_run"}},
            {"action": {"tool_name": "finish"}},
        ]
        template = [
            {"tool": "memory_write", "reward_weight": 1.0},
            {"tool": "finish", "reward_weight": 1.0},
        ]
        score = rw.compute_template_match(traj, template)
        assert score == 0.5

    def test_efficiency_bonus(self):
        rw = VerifiableReward()
        score = rw.compute_efficiency([{"step": 1}, {"step": 2}], 10)
        assert score > 0.5

    def test_diversity_high(self):
        rw = VerifiableReward()
        traj = [
            {"action": {"tool_name": "memory_write"}},
            {"action": {"tool_name": "bash_run"}},
            {"action": {"tool_name": "file_read"}},
            {"action": {"tool_name": "skill_create"}},
        ]
        score = rw.compute_diversity(traj)
        assert score == 1.0

    def test_diversity_low(self):
        rw = VerifiableReward()
        traj = [{"action": {"tool_name": "memory_write"}},
                {"action": {"tool_name": "memory_write"}}]
        score = rw.compute_diversity(traj)
        assert score == 0.0

    def test_trajectory_reward_with_template(self):
        rw = VerifiableReward()
        traj = [{"step": 1, "action": {"tool_name": "bash_run"},
                 "observation": {"success": True}}]
        score = rw.compute_trajectory(traj, {
            "task_completed": True,
            "template_match_score": 0.8,
            "efficiency": 0.6,
        })
        assert score > 0


class TestEvolution2:
    def test_multi_strategy_evolution(self):
        engine = SkillEvolutionEngine()
        prs = engine.evolve_multi_strategy("test_skill.py",
                                           [{"observation": {"error": "bug"}}])
        assert len(prs) == 4
        for pr in prs:
            assert "Auto-evolved" in pr

    def test_select_strategy_with_errors(self):
        engine = SkillEvolutionEngine()
        module = {"source": "def foo(): pass", "functions": ["foo"]}
        strategy = engine._select_strategy(module, ["Error: something"])
        assert strategy == "error_handling"

    def test_select_strategy_no_source(self):
        engine = SkillEvolutionEngine()
        strategy = engine._select_strategy({"source": ""}, [])
        assert strategy == "tool_addiction"


class TestLearner2:
    def test_continuous_learner_stats(self):
        store = TrajectoryStore()
        learner = ContinuousLearner(store, interval_hours=12,
                                    data_dir="/tmp/zilli_learn_test")
        stats = learner.stats()
        assert stats["interval_hours"] == 12
        assert stats["running"] is False

    def test_continuous_learner_should_sft(self):
        store = TrajectoryStore()
        learner = ContinuousLearner(store, sft_threshold=5, sft_callback=lambda x: {})
        for i in range(3):
            store.add_trajectory([{"step": 1}, {"step": 2}], 0.9)
        assert learner._should_trigger_sft() is False
        for i in range(3):
            store.add_trajectory([{"step": 1}, {"step": 2}], 0.9)
        assert learner._should_trigger_sft() is True


class TestScheduler:
    @pytest.mark.asyncio
    async def test_rollout_result_defaults(self):
        r = RolloutResult(task_id="t1", trajectory=[], reward=0.0, tokens=0, completed=True)
        assert r.status == RolloutStatus.COMPLETED
        assert r.retry_count == 0

    @pytest.mark.asyncio
    async def test_scheduler_stats(self):
        sched = AsyncRolloutScheduler()
        stats = sched.get_stats()
        assert stats["total_scheduled"] == 0
        assert stats["total_errors"] == 0

    @pytest.mark.asyncio
    async def test_scheduler_cancel(self):
        sched = AsyncRolloutScheduler()
        sched.cancel("test_task")
        # no error expected

    @pytest.mark.asyncio
    async def test_scheduler_with_tasks(self):
        async def fake_rollout(task):
            await asyncio.sleep(0.01)
            return {"trajectory": [{"step": 1}], "reward": 1.0, "tokens": 256}

        sched = AsyncRolloutScheduler(window_sec=30)
        tasks = [{"id": "a"}, {"id": "b"}]
        results = await sched.schedule(fake_rollout, tasks, timeout_per_task=5)
        assert len(results) == 2
        for r in results:
            assert r.completed
            assert r.reward == 1.0
            assert r.tokens == 256

    @pytest.mark.asyncio
    async def test_scheduler_timeout(self):
        async def slow_rollout(task):
            await asyncio.sleep(100)
            return {"trajectory": [], "reward": 0.0, "tokens": 0}

        sched = AsyncRolloutScheduler(window_sec=1)
        tasks = [{"id": "slow"}]
        results = await sched.schedule(slow_rollout, tasks, timeout_per_task=1)
        assert len(results) == 1
        assert not results[0].completed


class TestDistillation:
    def test_distillation_sample_creation(self):
        s = DistillationSample(
            executor_action={"tool": "bash"},
            planner_action={"tool": "bash"},
            executor_log_prob=0.7,
            planner_log_prob=0.85,
            executor_reward=0.6,
            planner_reward=0.9,
        )
        assert s.executor_log_prob == 0.7
        assert s.planner_log_prob == 0.85

    def test_bc_loss_basic(self):
        d = DistillationScheduler()
        bc, kl = d.compute_bc_loss([0.7, 0.8], [0.85, 0.9])
        assert bc > 0
        assert kl > 0
        assert bc >= kl

    def test_bc_loss_perfect_match(self):
        d = DistillationScheduler()
        bc, kl = d.compute_bc_loss([0.9, 0.8], [0.9, 0.8])
        assert bc > 0
        assert kl == 0.0

    def test_rl_loss(self):
        d = DistillationScheduler(reward_gamma=0.5)
        loss = d.compute_rl_loss([0.0, 0.0], [1.0, 1.0])
        assert loss > 0

    def test_rl_loss_executor_wins(self):
        d = DistillationScheduler(reward_gamma=0.2)
        loss = d.compute_rl_loss([0.9, 0.8], [0.5, 0.6])
        assert loss < 0

    def test_reg_loss_no_embeddings(self):
        d = DistillationScheduler()
        samples = [DistillationSample(
            executor_action={}, planner_action={},
            executor_log_prob=0.7, planner_log_prob=0.85,
            executor_reward=0.6, planner_reward=0.9,
        )]
        loss = d.compute_reg_loss(samples)
        assert loss == 0.0

    def test_reg_loss_with_embeddings(self):
        d = DistillationScheduler(embedding_delta=0.5)
        samples = [
            DistillationSample(
                executor_action={}, planner_action={},
                executor_log_prob=0.7, planner_log_prob=0.85,
                executor_reward=0.6, planner_reward=0.9,
                executor_embedding=[0.1, 0.2, 0.3],
                planner_embedding=[0.5, 0.6, 0.7],
            )
        ]
        loss = d.compute_reg_loss(samples)
        assert loss > 0

    def test_full_cycle(self):
        d = DistillationScheduler(lora_threshold=10)
        for i in range(20):
            d.add_sample(DistillationSample(
                executor_action={"step": i}, planner_action={"step": i},
                executor_log_prob=0.6 + i * 0.002,
                planner_log_prob=0.85,
                executor_reward=0.5,
                planner_reward=0.9,
            ))
        assert d.should_distill()
        cycle = d.run_cycle()
        assert cycle is not None
        assert cycle.samples == 20
        assert cycle.total_loss > 0
        assert cycle.kl_divergence > 0

    def test_add_batch(self):
        d = DistillationScheduler()
        samples = [DistillationSample(
            executor_action={}, planner_action={},
            executor_log_prob=0.7, planner_log_prob=0.85,
            executor_reward=0.6, planner_reward=0.9,
        ) for _ in range(20)]
        d.add_batch(samples)
        assert d.stats()["total_samples"] == 20

    def test_stats(self):
        d = DistillationScheduler(lambda_bc=0.8, lambda_rl=0.3)
        stats = d.stats()
        assert stats["lambda_bc"] == 0.8
        assert stats["lambda_rl"] == 0.3
        assert stats["cycles_completed"] == 0
        assert stats["total_samples"] == 0


class TestChampionChallenger:
    def test_register_champion(self):
        arena = ChampionChallenger(min_eval_tasks=1)
        arena.register_model("v1", "1.0", ArenaStatus.CHAMPION)
        assert arena.get_champion() == "v1"

    def test_register_contender(self):
        arena = ChampionChallenger()
        arena.register_model("v2", "2.0", ArenaStatus.CONTENDER)
        assert arena.get_champion() is None

    def test_add_score(self):
        arena = ChampionChallenger()
        arena.register_model("m1", "1.0", ArenaStatus.CHAMPION)
        arena.add_score("m1", 0.85)
        arena.add_score("m1", 0.90)
        lb = arena.leaderboard()
        assert lb[0]["avg_score"] == pytest.approx(0.875, abs=0.01)

    def test_run_match_champion_wins(self):
        arena = ChampionChallenger(min_win_gap=0.1, warmup_rounds=0)
        arena.register_model("champ", "1.0", ArenaStatus.CHAMPION)
        arena.register_model("chal", "2.0", ArenaStatus.CONTENDER)

        def eval_fn(name):
            if name == "champ":
                return [0.9, 0.85, 0.88, 0.92, 0.87, 0.9, 0.86, 0.91, 0.89, 0.88]
            return [0.6, 0.55, 0.58, 0.62, 0.57, 0.6, 0.56, 0.61, 0.59, 0.58]

        match = arena.run_match("chal", eval_fn)
        assert match is not None
        assert match.champion_score > match.challenger_score
        assert match.winner == "champ"

    def test_run_match_challenger_wins(self):
        arena = ChampionChallenger(min_win_gap=0.1, warmup_rounds=0)
        arena.register_model("champ", "1.0", ArenaStatus.CHAMPION)
        arena.register_model("chal", "2.0", ArenaStatus.CONTENDER)

        def eval_fn(name):
            if name == "champ":
                return [0.5, 0.55, 0.48, 0.52, 0.51] * 2
            return [0.9, 0.85, 0.88, 0.92, 0.87] * 2

        match = arena.run_match("chal", eval_fn)
        assert match is not None
        assert match.challenger_score > match.champion_score
        assert match.winner == "chal"

    def test_rollback(self):
        arena = ChampionChallenger(min_eval_tasks=1, warmup_rounds=0)
        arena.register_model("v1", "1.0", ArenaStatus.CHAMPION)
        arena.register_model("v2", "2.0", ArenaStatus.CONTENDER)

        def eval_fn(name):
            if name == "v1":
                return [0.5] * 10
            return [0.9] * 10

        arena.run_match("v2", eval_fn)
        champions = [m for m in arena._models.values() if m.status == ArenaStatus.CHAMPION]
        assert any(m.name == "v2" for m in champions)

        rolled = arena.rollback()
        assert rolled == "v1"
        assert arena.get_champion() == "v1"

    def test_leaderboard_ordering(self):
        arena = ChampionChallenger()
        arena.register_model("a", "1.0", ArenaStatus.CHAMPION)
        arena.register_model("b", "2.0", ArenaStatus.CONTENDER)
        arena.add_score("a", 0.9)
        arena.add_score("b", 0.8)
        arena.add_score("a", 0.85)
        lb = arena.leaderboard()
        assert lb[0]["name"] == "a"
        assert lb[0]["avg_score"] >= lb[1]["avg_score"]

    def test_stats_output(self):
        arena = ChampionChallenger(min_win_gap=0.1)
        arena.register_model("v1", "1.0", ArenaStatus.CHAMPION)
        arena.register_model("v2", "2.0", ArenaStatus.CONTENDER)

        def eval_fn(name):
            return [0.8, 0.85, 0.9, 0.88, 0.82, 0.87, 0.83, 0.86, 0.84, 0.89]

        arena.run_match("v2", eval_fn)
        stats = arena.stats()
        assert stats["total_matches"] == 1
        assert stats["current_champion"] == "v1"
        assert stats["champion_wins"] == 1
