import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger("zilli.security.pii")


class PIICategory(str, Enum):
    NAME = "name"
    ID_NUMBER = "id_number"
    PHONE = "phone"
    EMAIL = "email"
    ADDRESS = "address"
    SSN = "ssn"
    CREDIT_CARD = "credit_card"
    PASSPORT = "passport"
    DOB = "date_of_birth"
    MEDICAL_RECORD = "medical_record"
    BANK_ACCOUNT = "bank_account"
    IP_ADDRESS = "ip_address"
    API_KEY = "api_key"
    CHINESE_ID = "chinese_id"  # 中国身份证号


@dataclass
class PIEFinding:
    category: PIICategory
    text: str
    start: int
    end: int
    confidence: float = 1.0


@dataclass
class SanitizationResult:
    original: str
    sanitized: str
    findings: list[PIEFinding] = field(default_factory=list)
    mask_char: str = "***"


class PIIDetector:
    _PATTERNS: dict[PIICategory, str] = {
        PIICategory.EMAIL: r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        PIICategory.PHONE: r"(?:\+?86)?1[3-9]\d{9}",
        PIICategory.SSN: r"\b(?!000|666|9\d{2})\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b",
        PIICategory.CREDIT_CARD: r"\b(?:\d[ -]*?){13,16}\b",  # Note: results are validated with Luhn in detect()
        PIICategory.IP_ADDRESS: r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        PIICategory.CHINESE_ID: r"\b[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx]\b",
    }

    def __init__(self, custom_patterns: Optional[dict[PIICategory, str]] = None):
        self._compiled: dict[PIICategory, re.Pattern] = {}
        all_patterns = {**self._PATTERNS, **(custom_patterns or {})}
        for cat, pat in all_patterns.items():
            try:
                self._compiled[cat] = re.compile(pat)
            except re.error as e:
                logger.warning("Invalid regex for %s: %s", cat, e)

    @staticmethod
    def _luhn_check(digits: str) -> bool:
        nums = [int(c) for c in digits if c.isdigit()]
        if len(nums) < 13 or len(nums) > 16:
            return False
        checksum = 0
        for i, n in enumerate(reversed(nums)):
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            checksum += n
        return checksum % 10 == 0

    def detect(self, text: str) -> list[PIEFinding]:
        findings: list[PIEFinding] = []
        seen: set[tuple[int, int, str]] = set()

        for category, pattern in self._compiled.items():
            for match in pattern.finditer(text):
                key = (match.start(), match.end(), category.value)
                if key not in seen:
                    seen.add(key)
                    if category == PIICategory.CREDIT_CARD:
                        if not self._luhn_check(match.group()):
                            continue
                    findings.append(PIEFinding(
                        category=category,
                        text=match.group(),
                        start=match.start(),
                        end=match.end(),
                    ))

        findings.sort(key=lambda f: f.start)
        return findings

    def has_pii(self, text: str) -> bool:
        return len(self.detect(text)) > 0


class Sanitizer:
    def __init__(self, detector: Optional[PIIDetector] = None):
        self.detector = detector or PIIDetector()

    def sanitize(self, text: str, mask_char: str = "***") -> SanitizationResult:
        findings = self.detector.detect(text)
        if not findings:
            return SanitizationResult(original=text, sanitized=text, findings=[])

        sanitized = text
        for f in reversed(findings):
            sanitized = sanitized[:f.start] + mask_char + sanitized[f.end:]
        return SanitizationResult(
            original=text,
            sanitized=sanitized,
            findings=findings,
            mask_char=mask_char,
        )

    def sanitize_for_log(self, text: str) -> str:
        result = self.sanitize(text)
        return result.sanitized

    def sanitize_text(self, text: str) -> str:
        findings = self.detector.detect(text)
        if not findings:
            return text
        sanitized = text
        for f in reversed(findings):
            sanitized = sanitized[:f.start] + "***" + sanitized[f.end:]
        return sanitized
