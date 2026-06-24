from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator, Optional

from zilli.models.base import GenerationResult, ModelBackend

logger = logging.getLogger("zilli.models.vllm")

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    httpx = None


class VLLMBackend(ModelBackend):
    def __init__(self, name: str, model_id: str, base_url: str = "http://127.0.0.1:8000"):
        super().__init__(name, model_id, base_url)
        self._completions_url = f"{self.base_url}/v1/completions"
        self._models_url = f"{self.base_url}/v1/models"
        self._client: Optional["httpx.AsyncClient"] = None

    async def _ensure_client(self) -> "httpx.AsyncClient":
        if self._client is None and HAS_HTTPX:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(300.0),
                limits=httpx.Limits(max_keepalive_connections=8, max_connections=16),
            )
        return self._client  # type: ignore

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

        client = await self._ensure_client()
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        start = time.monotonic()
        try:
            response = await client.post(self._completions_url, json=payload, timeout=httpx.Timeout(60.0))
            duration_ms = (time.monotonic() - start) * 1000

            if response.status_code != 200:
                return GenerationResult(
                    text="", model_name=self.model_id,
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                    duration_ms=duration_ms,
                )

            data = response.json()
            choices = data.get("choices") or []
            text = choices[0].get("text", "") if len(choices) > 0 else ""
            usage = data.get("usage", {})
            return GenerationResult(
                text=text,
                model_name=self.model_id,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
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

        client = await self._ensure_client()
        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        try:
            async with client.stream("POST", self._completions_url, json=payload) as response:
                if response.status_code != 200:
                    logger.error("vLLM stream HTTP %d", response.status_code)
                    return
                async for line in response.aiter_lines():
                    if not line.strip() or line.startswith(":") or line == "data: [DONE]":
                        continue
                    if line.startswith("data: "):
                        try:
                            data = json.loads(line[6:])
                            delta = data.get("choices", [{}])[0].get("text", "")
                            if delta:
                                yield delta
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error("vLLM stream error: %s", e)

    async def generate_chat(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> GenerationResult:
        if not HAS_HTTPX:
            return GenerationResult(
                text="", model_name=self.model_id,
                error="httpx not installed",
            )

        client = await self._ensure_client()
        chat_url = f"{self.base_url}/v1/chat/completions"
        payload = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        start = time.monotonic()
        try:
            response = await client.post(chat_url, json=payload)
            duration_ms = (time.monotonic() - start) * 1000

            if response.status_code != 200:
                return GenerationResult(
                    text="", model_name=self.model_id,
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                    duration_ms=duration_ms,
                )

            data = response.json()
            choices = data.get("choices", [])
            text = choices[0].get("message", {}).get("content", "") if choices else ""
            usage = data.get("usage", {})
            return GenerationResult(
                text=text,
                model_name=self.model_id,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.monotonic() - start) * 1000
            return GenerationResult(
                text="", model_name=self.model_id,
                error=str(e), duration_ms=duration_ms,
            )

    async def health_check(self) -> bool:
        if not HAS_HTTPX:
            return False
        try:
            client = self._get_client()
            if client is None:
                return False
            client = await self._ensure_client()
            resp = await client.get(self._models_url, timeout=httpx.Timeout(5.0))
            if resp.status_code != 200:
                return False
            models = resp.json().get("data", [])
            return any(m["id"] == self.model_id for m in models)
        except Exception as e:
            logger.warning("vLLM health check failed: %s", e)
            return False
