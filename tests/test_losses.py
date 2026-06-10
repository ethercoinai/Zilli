import numpy as np
import pytest

from zilli.distillation.losses import DualModelDistillationLoss


class TestDualModelDistillationLoss:
    def test_bc_loss_basic(self):
        loss_fn = DualModelDistillationLoss(lambda_bc=1.0, beta=0.1)
        executor_probs = np.array([0.2, 0.3, 0.5])
        planner_probs = np.array([0.1, 0.2, 0.7])
        planner_action_id = np.array([0, 0, 1])
        result = loss_fn.compute_bc_loss(executor_probs, planner_probs, planner_action_id)
        assert isinstance(result, float)
        assert result > 0

    def test_bc_loss_planner_action_id_as_int(self):
        loss_fn = DualModelDistillationLoss()
        executor_probs = np.array([0.8, 0.1, 0.1])
        planner_probs = np.array([0.3, 0.3, 0.4])
        result = loss_fn.compute_bc_loss(executor_probs, planner_probs, 1)
        assert isinstance(result, float)

    def test_rl_loss_equal_rewards(self):
        loss_fn = DualModelDistillationLoss(gamma=0.2)
        result = loss_fn.compute_rl_loss(executor_reward=0.5, planner_reward=0.5)
        expected = -0.5 + 0.2 * 0.0
        assert result == pytest.approx(expected)

    def test_rl_loss_gap(self):
        loss_fn = DualModelDistillationLoss(gamma=0.5)
        result = loss_fn.compute_rl_loss(executor_reward=0.8, planner_reward=0.3)
        expected = -0.8 + 0.5 * (0.5) ** 2
        assert result == pytest.approx(expected)

    def test_regularization_loss_within_delta(self):
        loss_fn = DualModelDistillationLoss(delta=1.0)
        embed_a = np.array([0.5, 0.5])
        embed_b = np.array([0.6, 0.4])
        distance = np.linalg.norm(embed_a - embed_b)
        result = loss_fn.compute_regularization_loss(embed_a, embed_b)
        assert result == pytest.approx(max(0.0, distance - 1.0))
        assert result == 0.0

    def test_regularization_loss_exceeds_delta(self):
        loss_fn = DualModelDistillationLoss(delta=0.1)
        embed_a = np.array([1.0, 0.0])
        embed_b = np.array([0.0, 1.0])
        result = loss_fn.compute_regularization_loss(embed_a, embed_b)
        assert result > 0

    def test_total_loss_happy_path(self):
        loss_fn = DualModelDistillationLoss(lambda_bc=1.0, lambda_rl=0.5, lambda_reg=0.1)
        step = {
            "executor_probs": np.array([0.2, 0.3, 0.5]),
            "planner_probs": np.array([0.1, 0.2, 0.7]),
            "planner_action_id": np.array([2]),
            "executor_reward": 0.8,
            "planner_reward": 0.3,
            "executor_embed": np.array([0.1, 0.2]),
            "planner_embed": np.array([0.3, 0.4]),
        }
        total = loss_fn.total_loss(step)
        assert isinstance(total, float)
        assert total > 0

    def test_total_loss_zero_components(self):
        loss_fn = DualModelDistillationLoss(lambda_bc=0, lambda_rl=0, lambda_reg=0)
        step = {
            "executor_probs": np.array([0.5, 0.5]),
            "planner_probs": np.array([0.5, 0.5]),
            "planner_action_id": np.array([0]),
            "executor_reward": 0.0,
            "planner_reward": 0.0,
            "executor_embed": np.array([0.0, 0.0]),
            "planner_embed": np.array([0.0, 0.0]),
        }
        assert loss_fn.total_loss(step) == 0.0
