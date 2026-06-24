from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, Optional

from zilli.models.base import GenerationResult, ModelBackend
from zilli.models.config import DeploymentType, ModelConfig, ModelProfile, ModelRole
from zilli.models.llamacpp import LlamaCppBackend
from zilli.models.ollama import OllamaBackend
from zilli.models.vllm import VLLMBackend

if TYPE_CHECKING:
    from zilli.configs import ZilliConfig

logger = logging.getLogger("zilli.models.registry")

BACKEND_BUILDERS: dict[str, Callable] = {
    "ollama": lambda cfg: OllamaBackend(
        name=cfg.name, model_id=cfg.model_id, base_url=cfg.base_url,
    ),
    "vllm": lambda cfg: VLLMBackend(
        name=cfg.name, model_id=cfg.model_id, base_url=cfg.base_url,
    ),
    "llamacpp": lambda cfg: LlamaCppBackend(
        name=cfg.name, model_id=cfg.model_id, base_url=cfg.base_url,
    ),
}


class ModelRegistry:
    def __init__(self, profile: Optional[ModelProfile] = None,
                 config: Optional["ZilliConfig"] = None):
        if config is not None:
            profile = config.to_model_profile()
        if profile is None:
            profile = ModelProfile()
        self.profile = profile
        self._backends: dict[str, ModelBackend] = {}
        self._role_map: dict[ModelRole, list[str]] = {
            role: [] for role in ModelRole
        }
        self._fallback_chain: dict[ModelRole, list[str]] = {
            role: [] for role in ModelRole
        }

        for cfg in self.profile.models:
            self._register(cfg)

    def _register(self, cfg: ModelConfig) -> None:
        builder = BACKEND_BUILDERS.get(cfg.backend)
        if builder is None:
            logger.warning("Unknown backend %r for model %s, skipping", cfg.backend, cfg.name)
            return

        backend = builder(cfg)
        self._backends[cfg.name] = backend
        self._role_map[cfg.role].append(cfg.name)

        if cfg.is_fallback:
            self._fallback_chain[cfg.role].append(cfg.name)
        else:
            self._fallback_chain[cfg.role].insert(0, cfg.name)

        logger.info(
            "Registered model %s (backend=%s, role=%s, model_id=%s)",
            cfg.name, cfg.backend, cfg.role.value, cfg.model_id,
        )

    def get_model(self, name: str) -> Optional[ModelBackend]:
        return self._backends.get(name)

    async def get_model_for_role(self, role: ModelRole) -> Optional[ModelBackend]:
        import asyncio

        chain = self._fallback_chain.get(role, [])
        if not chain:
            return None

        async def _check(name: str) -> Optional[ModelBackend]:
            backend = self._backends.get(name)
            if backend and await backend.health_check():
                return backend
            return None

        results = await asyncio.gather(*[_check(n) for n in chain], return_exceptions=True)
        for r in results:
            if isinstance(r, ModelBackend):
                return r
        return None

    async def generate(
        self,
        role: ModelRole,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> GenerationResult:
        model_cfgs = [c for c in self.profile.models if c.role == role]
        errors: list[str] = []

        for cfg in model_cfgs:
            backend = self._backends.get(cfg.name)
            if not backend:
                continue
            try:
                if not await backend.health_check():
                    logger.warning("Model %s unhealthy, trying next", cfg.name)
                    errors.append(f"{cfg.name}: unhealthy")
                    continue
                return await backend.generate(
                    prompt=prompt,
                    max_tokens=max_tokens or cfg.max_tokens,
                    temperature=temperature if temperature is not None else cfg.temperature,
                )
            except Exception as e:
                logger.warning("Model %s failed: %s, trying next", cfg.name, e)
                errors.append(f"{cfg.name}: {e}")

        return GenerationResult(
            text="",
            model_name="none",
            error=f"All models for role {role.value} failed: {'; '.join(errors)}",
        )

    async def _generate_by_deployment(
        self, deployment: DeploymentType, prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> GenerationResult:
        candidates = [c for c in self.profile.models if c.deployment == deployment]
        errors: list[str] = []
        for cfg in candidates:
            backend = self._backends.get(cfg.name)
            if not backend:
                continue
            try:
                if not await backend.health_check():
                    errors.append(f"{cfg.name}: unhealthy")
                    continue
                return await backend.generate(
                    prompt=prompt,
                    max_tokens=max_tokens or cfg.max_tokens,
                    temperature=temperature if temperature is not None else cfg.temperature,
                )
            except Exception as e:
                errors.append(f"{cfg.name}: {e}")
        return GenerationResult(
            text="", model_name="none",
            error=f"No {deployment.value} model available: {'; '.join(errors)}",
        )

    async def generate_local(
        self, prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> GenerationResult:
        return await self._generate_by_deployment(DeploymentType.LOCAL, prompt, max_tokens, temperature)

    async def generate_cloud(
        self, prompt: str,
        provider: Optional[object] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
    ) -> GenerationResult:
        return await self._generate_by_deployment(DeploymentType.CLOUD, prompt, max_tokens, temperature)

    def list_models(self) -> list[dict]:
        return [
            {
                "name": cfg.name,
                "backend": cfg.backend,
                "model_id": cfg.model_id,
                "role": cfg.role.value,
                "alive": self._backends.get(cfg.name, None) is not None,
            }
            for cfg in self.profile.models
        ]

    def summary(self) -> dict:
        return {
            "total_models": len(self._backends),
            "per_role": {
                role.value: len(names) for role, names in self._role_map.items()
            },
            "profile_budget": self.profile.monthly_budget_usd,
        }
