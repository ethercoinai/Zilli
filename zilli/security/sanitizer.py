import logging
import re
from typing import Optional

from zilli.security.pii import PIIDetector, Sanitizer

logger = logging.getLogger("zilli.security.sanitizer")

_PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?(?:previous|above|prior)\s+instructions", re.I),
    re.compile(r"forget\s+(?:all\s+)?(?:previous|above|prior)", re.I),
    re.compile(r"you\s+are\s+(?:now|not\s+an?\s+ai|free|released)", re.I),
    re.compile(r"system\s+(?:prompt|message|instruction)", re.I),
    re.compile(r"<\|im_start\|>|<\|im_end\|>", re.I),
]


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

        for pattern in _PROMPT_INJECTION_PATTERNS:
            if pattern.search(clean):
                clean = pattern.sub("[REDACTED]", clean)
                logger.info("Prompt injection pattern removed from input")

        if strip_pii:
            result = self.pii_sanitizer.sanitize(clean)
            return result.sanitized

        return clean

    def classify_safe(self, text: str) -> bool:
        if len(text) > 1_000_000:
            return False
        for pattern in _PROMPT_INJECTION_PATTERNS:
            if pattern.search(text):
                return False
        return True


def safe_format(template: str, /, max_length: int = 100_000, **kwargs) -> str:
    for k, v in kwargs.items():
        if isinstance(v, str) and len(v) > max_length:
            kwargs[k] = v[:max_length] + "... [TRUNCATED]"
        elif isinstance(v, str):
            kwargs[k] = v.replace("<", "&lt;").replace(">", "&gt;")
    return template.format(**kwargs)


__all__ = ["InputSanitizer", "safe_format"]
