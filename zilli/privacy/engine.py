from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from zilli.privacy.classifier import CLASS_LEVEL, DataClass, DataClassifier
from zilli.privacy.consent import ConsentManager, DataUse
from zilli.privacy.policy import PolicyStore
from zilli.privacy.reid import ReIDAssessor, ReIDRisk
from zilli.security.pii import Sanitizer

logger = logging.getLogger("zilli.privacy.engine")


class SanitizationMode(str, Enum):
    NONE = "none"
    AUTO = "auto"
    FORCE = "force"
    STRICT = "strict"


@dataclass
class PrivacyVerdict:
    passed: bool
    data_class: DataClass
    risk_score: float
    sanitized_text: str
    original_text: str
    reid_risk: ReIDRisk
    mode: SanitizationMode
    requires_cloud: bool = False
    can_proceed_cloud: bool = True
    can_proceed_local: bool = True
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    duration_ms: float = 0.0


class PrivacyEngine:
    def __init__(
        self,
        classifier: Optional[DataClassifier] = None,
        sanitizer: Optional[Sanitizer] = None,
        reid: Optional[ReIDAssessor] = None,
        consent: Optional[ConsentManager] = None,
        policies: Optional[PolicyStore] = None,
    ):
        self.classifier = classifier or DataClassifier()
        self.sanitizer = sanitizer or Sanitizer()
        self.reid = reid or ReIDAssessor()
        self.consent = consent or ConsentManager()
        self.policies = policies or PolicyStore()

    def evaluate(
        self,
        text: str,
        tenant_id: str = "default",
        user_id: str = "",
        context_hint: str = "",
        mode: SanitizationMode = SanitizationMode.AUTO,
        require_cloud: bool = False,
    ) -> PrivacyVerdict:
        start = time.monotonic()
        reasons: list[str] = []
        warnings: list[str] = []

        classification = self.classifier.classify(text, context_hint)
        policy = self.policies.get(tenant_id)

        sanitized = text
        if mode == SanitizationMode.STRICT:
            sanitized = self.sanitizer.sanitize(text).sanitized
            reasons.append("Strict sanitization applied")
        elif mode == SanitizationMode.FORCE:
            sanitized = self.sanitizer.sanitize(text).sanitized
            reasons.append("Forced sanitization applied")
        elif mode == SanitizationMode.AUTO and classification.requires_sanitization:
            sanitized = self.sanitizer.sanitize(text).sanitized
            reasons.append(f"Auto-sanitized ({classification.data_class.value})")
        else:
            reasons.append("No sanitization needed")

        reid_assessment = self.reid.assess(sanitized, len(classification.pii_categories))
        if reid_assessment.risk in (ReIDRisk.HIGH, ReIDRisk.CRITICAL):
            warnings.append(reid_assessment.recommendation)

        passed = True
        can_use_cloud = classification.can_use_cloud
        if require_cloud and not can_use_cloud:
            can_use_cloud = False
            warnings.append("Data class too sensitive for cloud processing")

        if require_cloud and can_use_cloud:
            reasons.append("Cloud processing required by caller")

        if user_id:
            if require_cloud:
                consent_ok = self.consent.check(tenant_id, user_id, DataUse.CLOUD_INFERENCE)
                if not consent_ok:
                    warnings.append("No consent for cloud inference")
                    can_use_cloud = False
            consent_local = self.consent.check(tenant_id, user_id, DataUse.LOCAL_INFERENCE)
            if not consent_local:
                warnings.append("No consent for local inference")

        if policy is not None:
            if CLASS_LEVEL[classification.data_class] > CLASS_LEVEL[policy.max_allowed_class]:
                warnings.append(
                    f"Data class {classification.data_class.value} exceeds policy max {policy.max_allowed_class.value}"
                )
                passed = False
            if policy.require_consent and not user_id:
                warnings.append("User identification required by policy")
                passed = False
            if not require_cloud and not policy.allowed_cloud_providers and can_use_cloud:
                can_use_cloud = False
                warnings.append("Cloud processing restricted by policy")

        is_reid_safe = self.reid.is_safe_for_cloud(reid_assessment,
                                                     ReIDRisk.LOW if require_cloud else ReIDRisk.MEDIUM)
        if require_cloud and not is_reid_safe:
            warnings.append("Re-identification risk too high for cloud")
            can_use_cloud = False

        consent_local_ok = not user_id or self.consent.check(tenant_id, user_id, DataUse.LOCAL_INFERENCE)
        if user_id and not consent_local_ok:
            warnings.append("No consent for local inference")
        can_proceed_local = not require_cloud and consent_local_ok
        can_proceed_cloud = require_cloud and can_use_cloud and is_reid_safe

        if passed:
            passed = can_proceed_local or can_proceed_cloud

        duration = (time.monotonic() - start) * 1000

        return PrivacyVerdict(
            passed=passed,
            data_class=classification.data_class,
            risk_score=classification.risk_score,
            sanitized_text=sanitized,
            original_text=text,
            reid_risk=reid_assessment.risk,
            mode=mode,
            requires_cloud=require_cloud,
            can_proceed_cloud=can_proceed_cloud,
            can_proceed_local=can_proceed_local,
            reasons=reasons,
            warnings=warnings,
            duration_ms=round(duration, 2),
        )
