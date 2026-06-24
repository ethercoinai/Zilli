from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from zilli.privacy.classifier import DataClass
from zilli.privacy.engine import PrivacyEngine, PrivacyVerdict, SanitizationMode
from zilli.privacy.policy import CloudProvider

logger = logging.getLogger("zilli.hybrid.gatekeeper")


class ExecutionTarget(str, Enum):
    LOCAL = "local"
    CLOUD = "cloud"
    REJECTED = "rejected"
    LOCAL_WITH_CLOUD_FALLBACK = "local_with_cloud_fallback"


@dataclass
class GatekeeperDecision:
    target: ExecutionTarget
    verdict: PrivacyVerdict
    cloud_provider: Optional[CloudProvider] = None
    reason: str = ""
    warnings: list[str] = field(default_factory=list)


class PrivacyGatekeeper:
    def __init__(self, privacy_engine: PrivacyEngine):
        self.privacy = privacy_engine

    def decide(
        self,
        text: str,
        tenant_id: str = "default",
        user_id: str = "",
        context_hint: str = "",
        preferred_cloud: Optional[CloudProvider] = None,
        allow_cloud_fallback: bool = True,
    ) -> GatekeeperDecision:
        warnings: list[str] = []

        verdict = self.privacy.evaluate(
            text=text,
            tenant_id=tenant_id,
            user_id=user_id,
            context_hint=context_hint,
            mode=SanitizationMode.AUTO,
        )

        if verdict.data_class in (DataClass.RESTRICTED, DataClass.REGULATED):
            sanitized_verdict = self.privacy.evaluate(
                text=text,
                tenant_id=tenant_id,
                user_id=user_id,
                context_hint=context_hint,
                mode=SanitizationMode.STRICT,
                require_cloud=True,
            )
            if sanitized_verdict.can_proceed_cloud:
                if not self._is_provider_allowed(tenant_id, preferred_cloud):
                    warnings.append(f"Cloud provider {preferred_cloud} not allowed for tenant")
                    return GatekeeperDecision(
                        target=ExecutionTarget.REJECTED,
                        verdict=verdict,
                        reason="No allowed cloud provider for sanitized data",
                        warnings=warnings,
                    )
                return GatekeeperDecision(
                    target=ExecutionTarget.CLOUD,
                    verdict=sanitized_verdict,
                    cloud_provider=preferred_cloud or CloudProvider.CUSTOM,
                    reason=f"Sensitive data ({verdict.data_class.value}) → sanitize → cloud",
                    warnings=warnings,
                )
            else:
                warnings.append("Cannot process restricted/regulated data even with sanitization")
                return GatekeeperDecision(
                    target=ExecutionTarget.REJECTED,
                    verdict=verdict,
                    reason="Data too sensitive for any processing path",
                    warnings=warnings,
                )

        if verdict.can_proceed_cloud and verdict.can_proceed_local:
            if preferred_cloud and allow_cloud_fallback:
                return GatekeeperDecision(
                    target=ExecutionTarget.LOCAL_WITH_CLOUD_FALLBACK,
                    verdict=verdict,
                    cloud_provider=preferred_cloud,
                    reason="Safe for both local and cloud; local default with cloud fallback",
                )
            return GatekeeperDecision(
                target=ExecutionTarget.LOCAL,
                verdict=verdict,
                reason="Data safe for local execution",
            )

        if verdict.can_proceed_local:
            return GatekeeperDecision(
                target=ExecutionTarget.LOCAL,
                verdict=verdict,
                reason="Local execution only",
            )

        return GatekeeperDecision(
            target=ExecutionTarget.REJECTED,
            verdict=verdict,
            reason="No valid execution path",
            warnings=warnings,
        )

    def _is_provider_allowed(self, tenant_id: str, provider: Optional[CloudProvider]) -> bool:
        if provider is None:
            return True
        policy = self.privacy.policies.get(tenant_id)
        return provider in policy.allowed_cloud_providers
