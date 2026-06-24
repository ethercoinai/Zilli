from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("zilli.privacy.reid")


class ReIDRisk(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


_RISK_ORDER = {
    ReIDRisk.NONE: 0,
    ReIDRisk.LOW: 1,
    ReIDRisk.MEDIUM: 2,
    ReIDRisk.HIGH: 3,
    ReIDRisk.CRITICAL: 4,
}

_QUASI_IDENTIFIER_PATTERNS = [
    (r"\b\d{4}\b", "4-digit year/zip"),
    (r"\b(?:male|female|man|woman|boy|girl|non-binary)\b", "gender"),
    (r"\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b", "date"),
    (r"\b\d{3}-?\d{2,4}-?\d{4}\b", "phone-like pattern"),
    (r"\b[A-Z]{2,3}\s+\d{1,5}\b", "license plate"),
    (r"\b(?:Dr\.|Mr\.|Ms\.|Mrs\.|Prof\.)\s+[A-Z][a-z]+\b", "name prefix"),
]


@dataclass
class ReIDAssessment:
    risk: ReIDRisk = ReIDRisk.NONE
    score: float = 0.0
    quasi_identifiers: list[str] = field(default_factory=list)
    direct_identifiers: int = 0
    population_uniqueness: float = 0.0
    recommendation: str = ""


class ReIDAssessor:
    def __init__(self, pii_detector=None):
        if pii_detector:
            self.pii_detector = pii_detector
        else:
            from zilli.security.pii import PIIDetector
            self.pii_detector = PIIDetector()
        self._qid_patterns = [(re.compile(p, re.IGNORECASE), desc) for p, desc in _QUASI_IDENTIFIER_PATTERNS]

    def assess(self, sanitized_text: str, original_pii_count: int = 0) -> ReIDAssessment:
        score = 0.0
        quasi_ids: list[str] = []

        for pattern, desc in self._qid_patterns:
            matches = pattern.findall(sanitized_text)
            if matches:
                quasi_ids.append(f"{desc} (x{len(matches)})")
                score += 0.1 * len(matches)

        remaining_pii = self.pii_detector.detect(sanitized_text)
        direct_ids = len(remaining_pii)
        score += direct_ids * 0.5

        if direct_ids > 0:
            risk = ReIDRisk.CRITICAL
            recommendation = "Direct identifiers remain in text. Sanitization incomplete."
        elif score >= 2.0:
            risk = ReIDRisk.HIGH
            recommendation = "High re-identification risk. Additional sanitization required."
        elif score >= 1.0:
            risk = ReIDRisk.MEDIUM
            recommendation = "Moderate re-identification risk. Consider further sanitization."
        elif score >= 0.3:
            risk = ReIDRisk.LOW
            recommendation = "Low re-identification risk. Acceptable for most use cases."
        else:
            risk = ReIDRisk.NONE
            recommendation = "No re-identification risk detected."

        if original_pii_count > 10 and score < 0.5:
            score += 0.2
            if _RISK_ORDER[ReIDRisk(risk)] < _RISK_ORDER[ReIDRisk.LOW]:
                risk = ReIDRisk.LOW
                recommendation = "Low risk, but high original PII volume suggests caution."

        return ReIDAssessment(
            risk=risk,
            score=round(score, 3),
            quasi_identifiers=quasi_ids,
            direct_identifiers=direct_ids,
            recommendation=recommendation,
        )

    def is_safe_for_cloud(self, assessment: ReIDAssessment, threshold: ReIDRisk = ReIDRisk.LOW) -> bool:
        return _RISK_ORDER[assessment.risk] <= _RISK_ORDER[threshold]
