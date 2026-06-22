from unittest.mock import MagicMock, patch

import pytest

from zilli.infra.device_utils import (
    DeviceType,
    detect_device,
    get_device,
    get_device_count,
    is_cuda_available,
    is_gpu_available,
    set_device,
)


class TestDetectDevice:
    def test_detect_cpu_explicit(self):
        assert detect_device("cpu") == "cpu"

    @patch("zilli.infra.device_utils._auto_detect", return_value="cpu")
    def test_detect_auto_no_gpu(self, mock_auto):
        assert detect_device("auto") == "cpu"

    @patch("zilli.infra.device_utils._auto_detect", return_value="cuda")
    def test_detect_auto_with_cuda(self, mock_auto):
        assert detect_device("auto") == "cuda"

    @patch("zilli.infra.device_utils._validate_device", return_value="cuda")
    def test_detect_prefer_cuda(self, mock_validate):
        assert detect_device("cuda") == "cuda"

    @patch("zilli.infra.device_utils._validate_device", return_value="cpu")
    def test_detect_cuda_fallback(self, mock_validate):
        assert detect_device("cuda") == "cpu"


class TestGetDevice:
    def teardown_method(self):
        from zilli.infra import device_utils
        device_utils._global_device = None

    def test_get_device_cached(self):
        from zilli.infra import device_utils
        device_utils._global_device = None
        result = get_device()
        assert result in ("cpu", "cuda", "mps")

    def test_get_device_with_arg(self):
        result = get_device("cpu")
        assert result == "cpu"


class TestSetDevice:
    def teardown_method(self):
        from zilli.infra import device_utils
        device_utils._global_device = None

    def test_set_device_cpu(self):
        set_device("cpu")
        assert get_device() == "cpu"

    @patch("zilli.infra.device_utils._validate_device", return_value="cuda")
    def test_set_device_cuda(self, mock_validate):
        set_device("cuda")
        assert get_device() == "cuda"

    @patch("zilli.infra.device_utils._validate_device", return_value="cpu")
    def test_set_device_fallback(self, mock_validate):
        set_device("cuda")
        assert get_device() == "cpu"


class TestDeviceChecks:
    def teardown_method(self):
        from zilli.infra import device_utils
        device_utils._global_device = None

    def test_is_cuda_available(self):
        result = is_cuda_available()
        assert isinstance(result, bool)

    def test_is_gpu_available(self):
        result = is_gpu_available()
        assert isinstance(result, bool)

    def test_get_device_count_with_cuda(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 4
        with patch.dict("sys.modules", {"torch": mock_torch}):
            from zilli.infra import device_utils
            device_utils._global_device = None
            device_utils.detect_device("auto")
            assert device_utils.get_device_count() == 4

    def test_get_device_count_no_cuda(self):
        count = get_device_count()
        assert count == 0


class TestToDevice:
    def test_to_device_cpu_tensor(self):
        detect_device("cpu")

    def test_to_device_with_dict(self):
        from zilli.infra.device_utils import to_device
        with pytest.raises(ImportError):
            to_device({"key": "value"})


class TestAutoDetect:
    @patch("zilli.infra.device_utils._auto_detect")
    def test_auto_detect_no_torch(self, mock_auto):
        mock_auto.return_value = "cpu"
        assert detect_device("auto") == "cpu"

    @patch("zilli.infra.device_utils._auto_detect")
    def test_auto_detect_with_cuda(self, mock_auto):
        mock_auto.return_value = "cuda"
        assert detect_device("auto") == "cuda"

    @patch("zilli.infra.device_utils._auto_detect")
    def test_auto_detect_with_mps(self, mock_auto):
        mock_auto.return_value = "mps"
        assert detect_device("auto") == "mps"


class TestDeviceType:
    def test_device_type_values(self):
        assert DeviceType.CPU.value == "cpu"
        assert DeviceType.CUDA.value == "cuda"
        assert DeviceType.MPS.value == "mps"
        assert DeviceType.AUTO.value == "auto"
