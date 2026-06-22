from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator

from zilli.models.base import GenerationResult, ModelBackend

logger = logging.getLogger("zilli.models.llamacpp")

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    httpx = None


class LlamaCppBackend(ModelBackend):
    def __init__(self, name: str, model_id: str, base_url: str = "http://127.0.0.1:8080"):
        super().__init__(name, model_id, base_url)
        self._completion_url = f"{self.base_url}/completion"
        self._health_url = f"{self.base_url}/health"
        self._client: httpx.AsyncClient | None = None

    def _get_client(self) -> httpx.AsyncClient | None:
        if self._client is None and HAS_HTTPX:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))
        return self._client

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> GenerationResult:
        if not HAS_HTTPX:
            return GenerationResult(
                text="", model_name=self.model_id,
                error="httpx not installed",
            )

        client = self._get_client()
        if client is None:
            return GenerationResult(
                text="", model_name=self.model_id,
                error="Failed to create HTTP client",
            )

        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "cache_prompt": True,
        }

        start = time.monotonic()
        try:
            response = await client.post(self._completion_url, json=payload)
            duration_ms = (time.monotonic() - start) * 1000

            if response.status_code != 200:
                return GenerationResult(
                    text="", model_name=self.model_id,
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                    duration_ms=duration_ms,
                )

            data = response.json()
            return GenerationResult(
                text=data.get("content", ""),
                model_name=self.model_id,
                tokens_in=data.get("tokens_cached", 0) or data.get("tokens_evaluated", 0),
                tokens_out=data.get("tokens_predicted", 0),
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.monotonic() - start) * 1000
            return GenerationResult(
                text="", model_name=self.model_id,
                error=str(e), duration_ms=duration_ms,
            )

    async def generate_stream(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> AsyncIterator[str]:
        if not HAS_HTTPX:
            yield ""
            return

        client = self._get_client()
        if client is None:
            yield ""
            return

        payload = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": temperature,
            "cache_prompt": True,
            "stream": True,
        }

        try:
            async with client.stream("POST", self._completion_url, json=payload) as response:
                if response.status_code != 200:
                    logger.error("llama.cpp stream HTTP %d", response.status_code)
                    yield ""
                    return
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        chunk = data.get("content", "")
                        if chunk:
                            yield chunk
                        if data.get("stop", False):
                            break
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error("llama.cpp stream error: %s", e)
            yield ""

    async def health_check(self) -> bool:
        if not HAS_HTTPX:
            return False
        try:
            client = self._get_client()
            if client is None:
                return False
            resp = await client.get(self._health_url, timeout=httpx.Timeout(5.0))
            return resp.status_code == 200
        except Exception as e:
            logger.warning("llama.cpp health check failed: %s", e)
            return False
