from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from zilli.security.pii import PIICategory, PIIDetector

logger = logging.getLogger("zilli.privacy.classifier")


class DataClass(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    RESTRICTED = "restricted"
    REGULATED = "regulated"


CLASS_LEVEL: dict[DataClass, int] = {
    DataClass.PUBLIC: 0,
    DataClass.INTERNAL: 1,
    DataClass.CONFIDENTIAL: 2,
    DataClass.RESTRICTED: 3,
    DataClass.REGULATED: 4,
}


@dataclass
class ClassificationResult:
    data_class: DataClass
    pii_categories: list[PIICategory] = field(default_factory=list)
    risk_score: float = 0.0
    reasons: list[str] = field(default_factory=list)
    requires_sanitization: bool = False
    can_use_cloud: bool = True


_REGULATED_KEYWORDS: dict[str, DataClass] = {
    "hipaa": DataClass.REGULATED,
    "phi": DataClass.REGULATED,
    "protected health": DataClass.REGULATED,
    "gdpr": DataClass.REGULATED,
    "pci": DataClass.REGULATED,
    "payment card": DataClass.REGULATED,
    "sox": DataClass.RESTRICTED,
    "ferpa": DataClass.REGULATED,
    "student record": DataClass.REGULATED,
    "attorney-client": DataClass.RESTRICTED,
    "privileged": DataClass.RESTRICTED,
    "trade secret": DataClass.RESTRICTED,
    "classified": DataClass.RESTRICTED,
    "confidential": DataClass.CONFIDENTIAL,
    "internal only": DataClass.INTERNAL,
    "proprietary": DataClass.CONFIDENTIAL,
    "secret": DataClass.RESTRICTED,
    "top secret": DataClass.RESTRICTED,
    "commercial in confidence": DataClass.CONFIDENTIAL,
    "personal data": DataClass.CONFIDENTIAL,
    "sensitive personal": DataClass.RESTRICTED,
    "special category": DataClass.RESTRICTED,
    "patient data": DataClass.REGULATED,
    "medical record": DataClass.REGULATED,
    "credit card": DataClass.REGULATED,
    "bank account": DataClass.RESTRICTED,
}

_REGULATED_PII_MAP: dict[PIICategory, DataClass] = {
    PIICategory.MEDICAL_RECORD: DataClass.REGULATED,
    PIICategory.PASSPORT: DataClass.RESTRICTED,
    PIICategory.SSN: DataClass.RESTRICTED,
    PIICategory.CREDIT_CARD: DataClass.REGULATED,
    PIICategory.BANK_ACCOUNT: DataClass.RESTRICTED,
    PIICategory.CHINESE_ID: DataClass.RESTRICTED,
}


class DataClassifier:
    def __init__(self, pii_detector: Optional[PIIDetector] = None):
        self.pii_detector = pii_detector or PIIDetector()

    def classify(self, text: str, context_hint: str = "") -> ClassificationResult:
        pii_findings = self.pii_detector.detect(text)
        pii_categories = list({f.category for f in pii_findings})
        reasons: list[str] = []
        data_class = DataClass.PUBLIC

        for category in pii_categories:
            mapped = _REGULATED_PII_MAP.get(category)
            if mapped and CLASS_LEVEL[mapped] > CLASS_LEVEL[data_class]:
                data_class = mapped
                reasons.append(f"Contains {category.value}")
            elif not mapped and CLASS_LEVEL[DataClass.CONFIDENTIAL] > CLASS_LEVEL[data_class]:
                data_class = DataClass.CONFIDENTIAL
                reasons.append(f"Contains PII: {category.value}")

        lowered = (text + "\n" + context_hint).lower()
        max_level = CLASS_LEVEL[data_class]
        for keyword, dc in _REGULATED_KEYWORDS.items():
            if re.search(r"\b" + re.escape(keyword) + r"\b", lowered):
                level = CLASS_LEVEL[dc]
                if level > max_level:
                    max_level = level
                    data_class = dc
                    reasons.append(f"Keyword match: '{keyword}'")

        risk_score = CLASS_LEVEL[data_class] / 4.0
        if pii_findings:
            risk_score = min(1.0, risk_score + 0.1 * len(pii_findings))

        requires_sanitization = data_class in (
            DataClass.CONFIDENTIAL, DataClass.RESTRICTED, DataClass.REGULATED,
        )
        can_use_cloud = data_class in (DataClass.PUBLIC, DataClass.INTERNAL)

        return ClassificationResult(
            data_class=data_class,
            pii_categories=pii_categories,
            risk_score=round(risk_score, 3),
            reasons=reasons,
            requires_sanitization=requires_sanitization,
            can_use_cloud=can_use_cloud,
        )

    def classify_batch(self, texts: list[str]) -> list[ClassificationResult]:
        return [self.classify(t) for t in texts]
