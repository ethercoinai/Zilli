from unittest.mock import MagicMock, patch

from zilli.training.distillation import DistillationSample, DistillationScheduler


def _sample(exec_log_prob=-1.0, plan_log_prob=-1.5, exec_reward=0.5, plan_reward=0.8,
             exec_embed=None, plan_embed=None):
    return DistillationSample(
        executor_action={"tool": "write"},
        planner_action={"tool": "write"},
        executor_log_prob=exec_log_prob,
        planner_log_prob=plan_log_prob,
        executor_reward=exec_reward,
        planner_reward=plan_reward,
        executor_embedding=exec_embed,
        planner_embedding=plan_embed,
    )


class MockTorchTensor:
    def __init__(self, val=0.0):
        self._val = val

    def to(self, *args, **kwargs):
        return self

    def cpu(self):
        return self

    def mean(self):
        return MockTorchTensor(self._val)

    def sum(self):
        return MockTorchTensor(self._val)

    def __add__(self, other):
        v = other._val if isinstance(other, MockTorchTensor) else other
        return MockTorchTensor(self._val + v)

    def __radd__(self, other):
        return MockTorchTensor(self._val + other)

    def __sub__(self, other):
        v = other._val if isinstance(other, MockTorchTensor) else other
        return MockTorchTensor(self._val - v)

    def __mul__(self, other):
        v = other._val if isinstance(other, MockTorchTensor) else other
        return MockTorchTensor(self._val * v)

    def __rmul__(self, other):
        return MockTorchTensor(self._val * other)

    def __truediv__(self, other):
        v = other._val if isinstance(other, MockTorchTensor) else other
        return MockTorchTensor(self._val / v)

    def __neg__(self):
        return MockTorchTensor(-self._val)

    def __pow__(self, other):
        return MockTorchTensor(self._val ** other)

    def item(self):
        return self._val

    def clamp(self, min=None, max=None):
        return self

    def norm(self, *args, **kwargs):
        return MockTorchTensor(0.3)

    def log(self):
        return MockTorchTensor(-1.0)


class MockTorchModule:
    Tensor = MockTorchTensor

    @staticmethod
    def tensor(data, device=None):
        if isinstance(data, (int, float)):
            return MockTorchTensor(float(data))
        if isinstance(data, list):
            if data:
                return MockTorchTensor(sum(data) / len(data))
        return MockTorchTensor(0.0)

    @staticmethod
    def zeros(*args, **kwargs):
        return MockTorchTensor(0.0)

    @staticmethod
    def log(x):
        if isinstance(x, MockTorchTensor):
            return MockTorchTensor(-1.0)
        return MockTorchTensor(0.0)

    @staticmethod
    def norm(x, *args, **kwargs):
        return MockTorchTensor(0.3)

    @staticmethod
    def clamp(x, min=None, max=None):
        return MockTorchTensor(0.0)

    @staticmethod
    def is_tensor(obj):
        return isinstance(obj, MockTorchTensor)

    cuda = MagicMock()
    backends = MagicMock()


class TestGPULossComputation:
    def test_cpu_fallback_returns_none(self):
        scheduler = DistillationScheduler()
        samples = [_sample()]
        with patch("zilli.training.distillation.is_gpu_available", return_value=False):
            result = scheduler.compute_loss_torch(samples)
        assert result is None

    def _with_mock_torch(self):
        return patch.dict("sys.modules", {"torch": MockTorchModule()})

    def _patch_gpu(self):
        return (
            patch("zilli.training.distillation.is_gpu_available", return_value=True),
            patch("zilli.training.distillation.get_device", return_value="cuda"),
        )

    def test_compute_loss_torch_with_embeddings(self):
        scheduler = DistillationScheduler(lambda_bc=1.0, lambda_rl=0.5, lambda_reg=0.1)
        samples = [
            _sample(exec_embed=[0.1, 0.2], plan_embed=[0.3, 0.4]),
            _sample(exec_embed=[0.5, 0.6], plan_embed=[0.7, 0.8]),
        ]
        gpu_patch, dev_patch = self._patch_gpu()
        with gpu_patch, dev_patch, self._with_mock_torch():
            result = scheduler.compute_loss_torch(samples)
        assert result is not None
        assert "bc_loss" in result
        assert "rl_loss" in result
        assert "reg_loss" in result
        assert "total" in result
        assert "kl" in result
        for v in result.values():
            assert isinstance(v, float)

    def test_compute_loss_torch_no_embeddings(self):
        scheduler = DistillationScheduler(lambda_reg=0.0)
        samples = [
            _sample(exec_embed=None, plan_embed=None),
            _sample(exec_embed=None, plan_embed=None),
        ]
        gpu_patch, dev_patch = self._patch_gpu()
        with gpu_patch, dev_patch, self._with_mock_torch():
            result = scheduler.compute_loss_torch(samples)
        assert result is not None
        assert result["reg_loss"] == 0.0

    def test_no_torch_fallback(self):
        scheduler = DistillationScheduler()
        with patch("zilli.training.distillation.is_gpu_available", return_value=True), \
             patch.dict("sys.modules", {"torch": None}), \
             patch("builtins.__import__", lambda name, *a, **kw: (_ for _ in ()).throw(ImportError("no torch")) if name == "torch" else __import__(name, *a, **kw)):
            result = scheduler.compute_loss_torch([_sample()])
        assert result is None

    def test_torch_import_error_inside_gpu(self):
        scheduler = DistillationScheduler()
        with patch("zilli.training.distillation.is_gpu_available", return_value=True), \
             patch.dict("sys.modules", {"torch": None}), \
             patch("builtins.__import__", lambda name, *a, **kw: (_ for _ in ()).throw(ImportError("no torch")) if name == "torch" else __import__(name, *a, **kw)):
            result = scheduler.compute_loss_torch([_sample()])
        assert result is None

    def test_compute_loss_torch_single_sample(self):
        scheduler = DistillationScheduler()
        samples = [_sample(exec_log_prob=-0.5, plan_log_prob=-1.0)]
        gpu_patch, dev_patch = self._patch_gpu()
        with gpu_patch, dev_patch, self._with_mock_torch():
            result = scheduler.compute_loss_torch(samples)
        assert result is not None
        assert isinstance(result["total"], float)
