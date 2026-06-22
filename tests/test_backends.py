from zilli.models.base import ModelBackend
from zilli.models.llamacpp import LlamaCppBackend
from zilli.models.vllm import VLLMBackend


class TestVLLMBackend:
    def test_init(self):
        b = VLLMBackend(name="test", model_id="qwen2:7b")
        assert b.name == "test"
        assert b.model_id == "qwen2:7b"
        assert b.base_url == "http://127.0.0.1:8000"

    def test_custom_url(self):
        b = VLLMBackend(name="t", model_id="m", base_url="http://10.0.0.1:8001")
        assert b.base_url == "http://10.0.0.1:8001"

    def test_is_backend(self):
        b = VLLMBackend(name="t", model_id="m")
        assert isinstance(b, ModelBackend)

    def test_repr(self):
        b = VLLMBackend(name="test", model_id="m")
        r = repr(b)
        assert "VLLMBackend" in r
        assert "test" in r

    def test_health_no_httpx(self, monkeypatch):
        monkeypatch.setattr("zilli.models.vllm.HAS_HTTPX", False)
        b = VLLMBackend(name="t", model_id="m")
        import asyncio
        assert asyncio.run(b.health_check()) is False

    def test_generate_no_httpx(self, monkeypatch):
        monkeypatch.setattr("zilli.models.vllm.HAS_HTTPX", False)
        b = VLLMBackend(name="t", model_id="m")
        import asyncio
        result = asyncio.run(b.generate("hello"))
        assert result.error is not None
        assert "httpx" in result.error


class TestLlamaCppBackend:
    def test_init(self):
        b = LlamaCppBackend(name="test", model_id="llama3:8b")
        assert b.name == "test"
        assert b.model_id == "llama3:8b"
        assert b.base_url == "http://127.0.0.1:8080"

    def test_custom_url(self):
        b = LlamaCppBackend(name="t", model_id="m", base_url="http://10.0.0.1:8081")
        assert b.base_url == "http://10.0.0.1:8081"

    def test_is_backend(self):
        b = LlamaCppBackend(name="t", model_id="m")
        assert isinstance(b, ModelBackend)

    def test_repr(self):
        b = LlamaCppBackend(name="test", model_id="m")
        r = repr(b)
        assert "LlamaCppBackend" in r
        assert "test" in r

    def test_health_no_httpx(self, monkeypatch):
        monkeypatch.setattr("zilli.models.llamacpp.HAS_HTTPX", False)
        b = LlamaCppBackend(name="t", model_id="m")
        import asyncio
        assert asyncio.run(b.health_check()) is False

    def test_generate_no_httpx(self, monkeypatch):
        monkeypatch.setattr("zilli.models.llamacpp.HAS_HTTPX", False)
        b = LlamaCppBackend(name="t", model_id="m")
        import asyncio
        result = asyncio.run(b.generate("hello"))
        assert result.error is not None


class TestBackendRegistration:
    def test_vllm_registered(self):
        from zilli.models.registry import BACKEND_BUILDERS
        assert "vllm" in BACKEND_BUILDERS

    def test_llamacpp_registered(self):
        from zilli.models.registry import BACKEND_BUILDERS
        assert "llamacpp" in BACKEND_BUILDERS

    def test_ollama_still_registered(self):
        from zilli.models.registry import BACKEND_BUILDERS
        assert "ollama" in BACKEND_BUILDERS

    def test_vllm_config_creates_backend(self):
        from zilli.models.config import ModelConfig, ModelProfile, ModelRole
        from zilli.models.registry import ModelRegistry
        cfg = ModelConfig(
            name="v", model_id="qwen2:7b", role=ModelRole.EXECUTOR,
            backend="vllm", base_url="http://10.0.0.1:8000",
        )
        registry = ModelRegistry(ModelProfile(models=[cfg]))
        backend = registry.get_model("v")
        assert backend is not None
        assert backend.base_url == "http://10.0.0.1:8000"

    def test_llamacpp_config_creates_backend(self):
        from zilli.models.config import ModelConfig, ModelProfile, ModelRole
        from zilli.models.registry import ModelRegistry
        cfg = ModelConfig(
            name="l", model_id="llama3:8b", role=ModelRole.EXECUTOR,
            backend="llamacpp", base_url="http://10.0.0.1:8080",
        )
        registry = ModelRegistry(ModelProfile(models=[cfg]))
        backend = registry.get_model("l")
        assert backend is not None
        assert backend.base_url == "http://10.0.0.1:8080"
