import json
import logging
import time
from typing import AsyncIterator, Optional

from zilli.models.base import GenerationResult, ModelBackend

logger = logging.getLogger("zilli.models.ollama")

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False
    httpx = None  # type: ignore


class OllamaBackend(ModelBackend):
    def __init__(self, name: str, model_id: str, base_url: str = "http://127.0.0.1:11434"):
        super().__init__(name, model_id, base_url)
        self._generate_url = f"{self.base_url}/api/generate"
        self._chat_url = f"{self.base_url}/api/chat"
        self._client: Optional["httpx.AsyncClient"] = None

    def _get_client(self) -> "httpx.AsyncClient":
        if self._client is None and HAS_HTTPX:
            self._client = httpx.AsyncClient(timeout=httpx.Timeout(300.0))
        return self._client  # type: ignore

    async def generate(
        self,
        prompt: str,
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> GenerationResult:
        if not HAS_HTTPX:
            return GenerationResult(
                text="",
                model_name=self.model_id,
                error="httpx not installed. Install with: pip install httpx",
            )

        client = self._get_client()
        if client is None:
            return GenerationResult(
                text="",
                model_name=self.model_id,
                error="Failed to create HTTP client",
            )

        payload = {
            "model": self.model_id,
            "prompt": prompt,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
            "stream": False,
        }

        start = time.monotonic()
        try:
            response = await client.post(self._generate_url, json=payload)
            duration_ms = (time.monotonic() - start) * 1000

            if response.status_code != 200:
                return GenerationResult(
                    text="",
                    model_name=self.model_id,
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                    duration_ms=duration_ms,
                )

            data = response.json()
            return GenerationResult(
                text=data.get("response", ""),
                model_name=self.model_id,
                tokens_in=data.get("prompt_eval_count", 0),
                tokens_out=data.get("eval_count", 0),
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.monotonic() - start) * 1000
            return GenerationResult(
                text="",
                model_name=self.model_id,
                error=str(e),
                duration_ms=duration_ms,
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
            "model": self.model_id,
            "prompt": prompt,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
            "stream": True,
        }

        try:
            async with client.stream("POST", self._generate_url, json=payload) as response:
                if response.status_code != 200:
                    logger.error("Ollama stream HTTP %d", response.status_code)
                    yield ""
                    return
                async for line in response.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        chunk = data.get("response", "")
                        if chunk:
                            yield chunk
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.error("Ollama stream error: %s", e)
            yield ""

    async def generate_chat(
        self,
        messages: list[dict],
        max_tokens: int = 2048,
        temperature: float = 0.1,
    ) -> GenerationResult:
        if not HAS_HTTPX:
            return GenerationResult(
                text="",
                model_name=self.model_id,
                error="httpx not installed",
            )

        client = self._get_client()
        if client is None:
            return GenerationResult(
                text="",
                model_name=self.model_id,
                error="Failed to create HTTP client",
            )

        payload = {
            "model": self.model_id,
            "messages": messages,
            "options": {
                "num_predict": max_tokens,
                "temperature": temperature,
            },
            "stream": False,
        }

        start = time.monotonic()
        try:
            response = await client.post(self._chat_url, json=payload)
            duration_ms = (time.monotonic() - start) * 1000

            if response.status_code != 200:
                return GenerationResult(
                    text="",
                    model_name=self.model_id,
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                    duration_ms=duration_ms,
                )

            data = response.json()
            return GenerationResult(
                text=data.get("message", {}).get("content", ""),
                model_name=self.model_id,
                tokens_in=data.get("prompt_eval_count", 0),
                tokens_out=data.get("eval_count", 0),
                duration_ms=duration_ms,
            )
        except Exception as e:
            duration_ms = (time.monotonic() - start) * 1000
            return GenerationResult(
                text="",
                model_name=self.model_id,
                error=str(e),
                duration_ms=duration_ms,
            )

    async def health_check(self) -> bool:
        if not HAS_HTTPX:
            return False
        try:
            client = self._get_client()
            if client is None:
                return False
            resp = await client.get(f"{self.base_url}/api/tags", timeout=httpx.Timeout(5.0))
            if resp.status_code != 200:
                logger.warning("Ollama health check failed: HTTP %d", resp.status_code)
                return False
            models = resp.json().get("models", [])
            available = any(m["name"] == self.model_id for m in models)
            if not available:
                logger.warning(
                    "Model %s not found in Ollama. Available: %s",
                    self.model_id,
                    [m["name"] for m in models],
                )
            return available
        except Exception as e:
            logger.warning("Ollama health check failed: %s", e)
            return False
