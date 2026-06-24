from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from zilli.hybrid.gatekeeper import ExecutionTarget, PrivacyGatekeeper
from zilli.models.base import GenerationResult
from zilli.models.registry import ModelRegistry
from zilli.privacy.engine import PrivacyVerdict

logger = logging.getLogger("zilli.hybrid.executor")


@dataclass
class HybridResult:
    text: str
    target: ExecutionTarget
    verdict: PrivacyVerdict
    model_name: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)


class HybridExecutor:
    def __init__(
        self,
        gatekeeper: PrivacyGatekeeper,
        registry: ModelRegistry,
    ):
        self.gatekeeper = gatekeeper
        self.registry = registry

    async def execute(
        self,
        prompt: str,
        tenant_id: str = "default",
        user_id: str = "",
        context_hint: str = "",
        preferred_cloud_provider: Optional[str] = None,
    ) -> HybridResult:
        from zilli.privacy.policy import CloudProvider

        provider = CloudProvider(preferred_cloud_provider) if preferred_cloud_provider else None
        decision = self.gatekeeper.decide(
            text=prompt,
            tenant_id=tenant_id,
            user_id=user_id,
            context_hint=context_hint,
            preferred_cloud=provider,
        )

        if decision.target == ExecutionTarget.REJECTED:
            logger.warning("Request rejected by gatekeeper: %s", decision.reason)
            return HybridResult(
                text="",
                target=ExecutionTarget.REJECTED,
                verdict=decision.verdict,
                error=f"Rejected by privacy gatekeeper: {decision.reason}",
                warnings=decision.warnings,
            )

        use_sanitized = decision.verdict.sanitized_text != prompt
        input_text = decision.verdict.sanitized_text if use_sanitized else prompt

        try:
            if decision.target == ExecutionTarget.CLOUD:
                result = await self._call_cloud(input_text, provider)
            elif decision.target == ExecutionTarget.LOCAL_WITH_CLOUD_FALLBACK:
                result = await self._call_local(input_text)
                if result.error:
                    logger.info("Local failed, falling back to cloud: %s", result.error)
                    result = await self._call_cloud(input_text, provider)
            else:
                result = await self._call_local(input_text)

            return HybridResult(
                text=result.text,
                target=decision.target,
                verdict=decision.verdict,
                model_name=result.model_name,
                tokens_in=result.tokens_in,
                tokens_out=result.tokens_out,
                error=result.error,
                warnings=decision.warnings,
            )
        except Exception as e:
            logger.error("Hybrid execution failed: %s", e)
            return HybridResult(
                text="",
                target=decision.target,
                verdict=decision.verdict,
                error=str(e),
                warnings=decision.warnings,
            )

    async def _call_local(self, prompt: str) -> GenerationResult:
        return await self.registry.generate_local(prompt)

    async def _call_cloud(self, prompt: str, provider: Optional[object] = None) -> GenerationResult:
        return await self.registry.generate_cloud(prompt, provider)
