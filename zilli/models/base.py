from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import AsyncIterator, Optional


@dataclass
class GenerationResult:
    text: str
    model_name: str
    tokens_in: int = 0
    tokens_out: int = 0
    duration_ms: float = 0.0
    error: Optional[str] = None


class ModelBackend(ABC):
    def __init__(self, name: str, model_id: str, base_url: str):
        self.name = name
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> GenerationResult:
        ...

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> AsyncIterator[str]:
        result = await self.generate(prompt, max_tokens, temperature)
        if result.text:
            yield result.text

    async def generate_batch(
        self,
        prompts: list[str],
        max_tokens: int = 2048,
        temperature: float = 0.1,
        max_concurrency: int = 4,
    ) -> list[GenerationResult]:
        import asyncio
        sem = asyncio.Semaphore(max_concurrency)

        async def _one(p: str) -> GenerationResult:
            async with sem:
                return await self.generate(p, max_tokens, temperature)

        return await asyncio.gather(*[ _one(p) for p in prompts ], return_exceptions=True)

    @abstractmethod
    async def health_check(self) -> bool:
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, model_id={self.model_id!r})"
