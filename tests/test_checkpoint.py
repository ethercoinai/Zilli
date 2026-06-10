import json
from pathlib import Path

import pytest

from zilli.training.distillation import DistillationSample, DistillationScheduler


def _s(exec_reward=0.5, plan_reward=0.8):
    return DistillationSample(
        executor_action={"tool": "write"},
        planner_action={"tool": "write"},
        executor_log_prob=-1.0,
        planner_log_prob=-1.5,
        executor_reward=exec_reward,
        planner_reward=plan_reward,
        executor_embedding=[0.1, 0.2],
        planner_embedding=[0.3, 0.4],
    )


class TestSaveCheckpoint:
    def test_save_checkpoint_creates_file(self, tmp_path):
        scheduler = DistillationScheduler(log_dir=str(tmp_path))
        scheduler.add_batch([_s() for _ in range(5)])
        path = scheduler.save_checkpoint(str(tmp_path / "ckpt.json"))
        assert Path(path).exists()
        with open(path) as f:
            data = json.load(f)
        assert "scheduler_params" in data
        assert "state" in data
        assert data["state"]["total_samples"] == 5

    def test_save_checkpoint_default_path(self, tmp_path):
        scheduler = DistillationScheduler(log_dir=str(tmp_path))
        scheduler.add_sample(_s())
        path = scheduler.save_checkpoint()
        assert Path(path).exists()
        assert "checkpoint.json" in path

    def test_save_checkpoint_after_cycle(self, tmp_path):
        scheduler = DistillationScheduler(log_dir=str(tmp_path))
        scheduler.add_batch([_s() for _ in range(10)])
        scheduler.run_cycle()
        path = scheduler.save_checkpoint(str(tmp_path / "after_cycle.json"))
        with open(path) as f:
            data = json.load(f)
        assert isinstance(data["state"]["cycles"][0]["total_loss"], float)

    def test_save_preserves_kl_deque(self, tmp_path):
        scheduler = DistillationScheduler(log_dir=str(tmp_path))
        scheduler.add_batch([_s() for _ in range(20)])
        scheduler.run_cycle()
        scheduler.add_batch([_s(exec_reward=1.0) for _ in range(20)])
        scheduler.run_cycle()
        path = scheduler.save_checkpoint(str(tmp_path / "kl.json"))
        with open(path) as f:
            data = json.load(f)
        assert len(data["state"]["recent_kl"]) == 2


class TestLoadCheckpoint:
    def test_load_restores_state(self, tmp_path):
        scheduler = DistillationScheduler(log_dir=str(tmp_path))
        scheduler.add_batch([_s() for _ in range(15)])
        scheduler.run_cycle()
        scheduler.add_batch([_s() for _ in range(5)])
        path = scheduler.save_checkpoint(str(tmp_path / "restore.json"))

        loaded = DistillationScheduler.load_checkpoint(path)
        assert loaded._total_samples == 20
        assert len(loaded._cycles) == 1
        assert len(loaded._buffer) == 5
        assert loaded.stats()["buffer_size"] == 5

    def test_load_restores_params(self, tmp_path):
        scheduler = DistillationScheduler(
            lambda_bc=0.5, lambda_rl=0.3, lora_threshold=500,
            log_dir=str(tmp_path),
        )
        scheduler.add_batch([_s() for _ in range(3)])
        path = scheduler.save_checkpoint(str(tmp_path / "params.json"))

        loaded = DistillationScheduler.load_checkpoint(path)
        assert loaded.lambda_bc == 0.5
        assert loaded.lambda_rl == 0.3
        assert loaded.lora_threshold == 500

    def test_load_restores_timestamps(self, tmp_path):
        scheduler = DistillationScheduler(log_dir=str(tmp_path))
        scheduler._last_lora_time = 12345.0
        scheduler._last_full_sft_time = 67890.0
        scheduler.add_batch([_s() for _ in range(3)])
        path = scheduler.save_checkpoint(str(tmp_path / "ts.json"))

        loaded = DistillationScheduler.load_checkpoint(path)
        assert loaded._last_lora_time == 12345.0
        assert loaded._last_full_sft_time == 67890.0

    def test_load_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            DistillationScheduler.load_checkpoint("/nonexistent/path.json")

    def test_load_empty_buffer(self, tmp_path):
        scheduler = DistillationScheduler(log_dir=str(tmp_path))
        path = scheduler.save_checkpoint(str(tmp_path / "empty.json"))
        loaded = DistillationScheduler.load_checkpoint(path)
        assert loaded._total_samples == 0
        assert len(loaded._buffer) == 0
        assert len(loaded._cycles) == 0

    def test_load_continues_training(self, tmp_path):
        scheduler = DistillationScheduler(log_dir=str(tmp_path))
        scheduler.add_batch([_s() for _ in range(10)])
        scheduler.run_cycle()
        path = scheduler.save_checkpoint(str(tmp_path / "cont.json"))

        loaded = DistillationScheduler.load_checkpoint(path)
        loaded.add_batch([_s() for _ in range(5)])
        cycle = loaded.run_cycle()
        assert cycle is not None
        assert loaded._total_samples == 15
        assert len(loaded._cycles) == 2
