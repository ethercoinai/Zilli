import logging
import threading
from enum import Enum
from typing import Optional

logger = logging.getLogger("zilli.device_utils")


class DeviceType(Enum):
    CPU = "cpu"
    CUDA = "cuda"
    MPS = "mps"
    AUTO = "auto"


_global_device: Optional[str] = None
_global_device_lock = threading.Lock()


def detect_device(prefer: str = "auto") -> str:
    if prefer and prefer != "auto":
        normalized = prefer.lower()
        if normalized in ("cpu", "cuda", "mps"):
            return _validate_device(normalized)
    return _auto_detect()


def _validate_device(device: str) -> str:
    if device == "cuda":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        logger.warning("CUDA requested but not available, falling back to CPU")
        return "cpu"
    if device == "mps":
        try:
            import torch
            if torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        logger.warning("MPS requested but not available, falling back to CPU")
        return "cpu"
    return device


def _auto_detect() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("CUDA detected, using GPU")
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            logger.info("MPS detected, using Apple Silicon GPU")
            return "mps"
    except ImportError:
        pass
    logger.info("No GPU detected, using CPU")
    return "cpu"


def get_device(device: Optional[str] = None) -> str:
    global _global_device
    if device:
        return detect_device(device)
    with _global_device_lock:
        if _global_device is None:
            _global_device = detect_device("auto")
        return _global_device


def set_device(device: str):
    global _global_device
    validated = _validate_device(device)
    with _global_device_lock:
        _global_device = validated
    logger.info("Device set to %s", _global_device)


def is_cuda_available() -> bool:
    return get_device() == "cuda"


def is_mps_available() -> bool:
    return get_device() == "mps"


def is_gpu_available() -> bool:
    return is_cuda_available() or is_mps_available()


def get_device_count() -> int:
    if is_cuda_available():
        import torch
        return torch.cuda.device_count()
    return 0


def to_device(tensor, device: Optional[str] = None):
    try:
        import torch
    except ImportError:
        raise ImportError("torch is required for to_device()")
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"Expected torch.Tensor, got {type(tensor).__name__}")
    d = get_device(device)
    if d == "cpu":
        return tensor.cpu()
    return tensor.to(d)


__all__ = [
    "DeviceType", "detect_device", "get_device", "set_device",
    "is_cuda_available", "is_mps_available", "is_gpu_available",
    "get_device_count", "to_device",
]
