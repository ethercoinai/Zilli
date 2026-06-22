import pytest

from zilli.models.base import GenerationResult, ModelBackend


class MockStreamBackend(ModelBackend):
    def __init__(self):
        super().__init__(name="mock", model_id="mock-model", base_url="http://mock:9999")

    async def generate(self, prompt: str, max_tokens: int = 2048,
                       temperature: float = 0.1) -> GenerationResult:
        return GenerationResult(text="mock full response", model_name=self.model_id)

    async def generate_stream(self, prompt: str, max_tokens: int = 2048,
                              temperature: float = 0.1):
        words = prompt.split()
        for w in words:
            yield w

    async def health_check(self) -> bool:
        return True


class FailingBackend(ModelBackend):
    def __init__(self):
        super().__init__(name="fail", model_id="fail", base_url="http://mock:9999")

    async def generate(self, prompt: str, max_tokens: int = 2048,
                       temperature: float = 0.1) -> GenerationResult:
        return GenerationResult(text="", model_name="fail", error="fail")
    async def generate_stream(self, prompt: str, max_tokens: int = 2048,
                              temperature: float = 0.1):
        yield ""
        raise RuntimeError("stream error")


    def health_check(self) -> bool:
        return False


class TestModelBackendStreaming:
    @pytest.mark.asyncio
    async def test_generate_stream_base_class(self):
        backend = MockStreamBackend()
        collected = []
        async for chunk in backend.generate_stream("test prompt"):
            collected.append(chunk)
        assert len(collected) > 0

    @pytest.mark.asyncio
    async def test_generate_stream_yields_chunks(self):
        backend = MockStreamBackend()
        collected = []
        async for chunk in backend.generate_stream("hello world foo bar baz"):
            collected.append(chunk)
        assert len(collected) >= 2

    @pytest.mark.asyncio
    async def test_generate_stream_empty_prompt(self):
        backend = MockStreamBackend()
        collected = []
        async for chunk in backend.generate_stream(""):
            collected.append(chunk)
        assert isinstance(collected, list)

    @pytest.mark.asyncio
    async def test_generate_stream_handles_exception(self):
        backend = FailingBackend()
        collected = []
        try:
            async for chunk in backend.generate_stream("test"):
                collected.append(chunk)
        except RuntimeError:
            collected.append("error")
        assert len(collected) > 0


class TestGenerateBatch:
    @pytest.mark.asyncio
    async def test_batch_returns_list(self):
        backend = MockStreamBackend()
        prompts = ["hello", "world"]
        results = await backend.generate_batch(prompts)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, GenerationResult)

    @pytest.mark.asyncio
    async def test_batch_with_empty_list(self):
        backend = MockStreamBackend()
        results = await backend.generate_batch([])
        assert results == []


class TestGenerationResult:
    def test_default_values(self):
        r = GenerationResult(text="hello", model_name="test")
        assert r.text == "hello"
        assert r.model_name == "test"
        assert r.tokens_in == 0
        assert r.tokens_out == 0
        assert r.duration_ms == 0.0
        assert r.error is None

    def test_with_error(self):
        r = GenerationResult(text="", model_name="test", error="timeout")
        assert r.error == "timeout"

    def test_with_all_fields(self):
        r = GenerationResult(
            text="response", model_name="m", tokens_in=10, tokens_out=20,
            duration_ms=100.0, error=None,
        )
        assert r.tokens_in == 10
        assert r.tokens_out == 20
        assert r.duration_ms == 100.0
