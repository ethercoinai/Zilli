import logging
from typing import Optional

from zilli.security.pii import PIIDetector, Sanitizer

logger = logging.getLogger("zilli.security.sanitizer")

_INJECTION_SIGNATURES = [
    "ignore all previous", "ignore all above", "ignore all prior",
    "forget all previous", "forget all above",
    "you are now", "you are not an", "you are free", "you are released",
    "system prompt", "system message", "system instruction",
    "<|im_start|>", "<|im_end|>",
    "you must obey", "override", "new instructions",
]


def _has_injection(text: str) -> bool:
    lowered = text.lower()
    # Normalize Unicode confusables
    normalized = lowered.replace("\u201c", '"').replace("\u201d", '"')
    normalized = normalized.replace("\u2018", "'").replace("\u2019", "'")
    normalized = normalized.replace("\u00a0", " ")
    # Strip common padding
    normalized = " ".join(normalized.split())
    for sig in _INJECTION_SIGNATURES:
        if sig in normalized:
            return True
    return False


class InputSanitizer:
    def __init__(self, pii_detector: Optional[PIIDetector] = None):
        self.pii_sanitizer = Sanitizer(detector=pii_detector)

    def sanitize(self, text: Optional[str], strip_pii: bool = True) -> str:
        if not text:
            return ""

        clean = text.strip()
        if len(clean) > 1_000_000:
            clean = clean[:1_000_000]
            logger.warning("Input truncated to 1M chars")

        if _has_injection(clean):
            clean = "[REDACTED - potential prompt injection]"
            logger.info("Prompt injection detected in input")

        if strip_pii and not _has_injection(clean):
            result = self.pii_sanitizer.sanitize(clean)
            return result.sanitized

        return clean

    def classify_safe(self, text: str) -> bool:
        if len(text) > 1_000_000:
            return False
        if _has_injection(text):
            return False
        return True


def safe_format(template: str, /, max_length: int = 100_000, **kwargs) -> str:
    for k, v in kwargs.items():
        if isinstance(v, str) and len(v) > max_length:
            kwargs[k] = v[:max_length] + "... [TRUNCATED]"
    return template.format(**kwargs)


__all__ = ["InputSanitizer", "safe_format"]
