import pytest

from zilli.models.base import GenerationResult, ModelBackend
from zilli.models.config import ModelConfig, ModelProfile, ModelRole
from zilli.models.registry import ModelRegistry

# ── Mock backend for testing ─────────────────────────────────────────────


class MockBackend(ModelBackend):
    def __init__(self, name: str, model_id: str, base_url: str = "http://mock:9999",
                 health: bool = True, fail_every: int = 0):
        super().__init__(name, model_id, base_url)
        self._health = health
        self._call_count = 0
        self._fail_every = fail_every

    async def generate(self, prompt: str, max_tokens: int = 2048,
                       temperature: float = 0.1) -> GenerationResult:
        self._call_count += 1
        if self._fail_every and self._call_count % self._fail_every == 0:
            return GenerationResult(text="", model_name=self.model_id,
                                    error="mock failure")
        return GenerationResult(
            text=f"mock response to: {prompt[:40]}",
            model_name=self.model_id,
            tokens_in=len(prompt),
            tokens_out=50,
            duration_ms=100.0,
        )

    async def health_check(self) -> bool:
        return self._health


# ── ModelConfig tests ──────────────────────────────────────────────────


class TestModelConfig:
    def test_default_planner(self):
        cfg = ModelConfig(name="planner", model_id="qwen3:27b", role=ModelRole.PLANNER)
        assert cfg.role == ModelRole.PLANNER
        assert cfg.model_id == "qwen3:27b"
        assert cfg.base_url == "http://127.0.0.1:11434"
        assert cfg.temperature == 0.1
        assert cfg.max_tokens == 2048

    def test_default_executor(self):
        cfg = ModelConfig(name="executor", model_id="qwen3:7b", role=ModelRole.EXECUTOR)
        assert cfg.role == ModelRole.EXECUTOR
        assert cfg.model_id == "qwen3:7b"

    def test_default_reviewer(self):
        cfg = ModelConfig(name="reviewer", model_id="qwen3:27b", role=ModelRole.REVIEWER)
        assert cfg.role == ModelRole.REVIEWER

    def test_extra_forbidden(self):
        with pytest.raises(Exception):
            ModelConfig(name="test", model_id="m", role=ModelRole.EXECUTOR, extra_field="x")

    def test_custom_values(self):
        cfg = ModelConfig(
            name="test", model_id="llama3:8b", role=ModelRole.EXECUTOR,
            base_url="http://10.0.0.1:11434", temperature=0.3, max_tokens=8192,
            is_fallback=True,
        )
        assert cfg.is_fallback is True
        assert cfg.base_url == "http://10.0.0.1:11434"

    def test_enum_str(self):
        assert str(ModelRole.PLANNER) == "planner"
        assert str(ModelRole.EXECUTOR) == "executor"
        assert str(ModelRole.REVIEWER) == "reviewer"


class TestModelProfile:
    def test_default_profile_has_three_models(self):
        profile = ModelProfile()
        roles = {m.role for m in profile.models}
        assert ModelRole.PLANNER in roles
        assert ModelRole.EXECUTOR in roles
        assert ModelRole.REVIEWER in roles
        assert profile.monthly_budget_usd == 500.0

    def test_custom_models(self):
        profile = ModelProfile(
            models=[
                ModelConfig(name="p", model_id="deepseek-r1:32b", role=ModelRole.PLANNER),
                ModelConfig(name="e", model_id="gemma2:9b", role=ModelRole.EXECUTOR),
            ],
            monthly_budget_usd=1000.0,
        )
        assert len(profile.models) == 2
        assert profile.monthly_budget_usd == 1000.0


# ── ModelRegistry tests ────────────────────────────────────────────────


class TestModelRegistry:
    def test_register_ollama_backend(self):
        cfg = ModelConfig(name="mock", model_id="mock-model", role=ModelRole.PLANNER)
        profile = ModelProfile(models=[cfg])
        registry = ModelRegistry(profile)
        backend = registry.get_model("mock")
        assert backend is not None
        assert isinstance(backend, ModelBackend)
        assert backend.name == "mock"
        assert backend.model_id == "mock-model"

    def test_unknown_backend_skipped(self):
        cfg = ModelConfig(name="bad", model_id="x", role=ModelRole.PLANNER, backend="nonexistent")
        profile = ModelProfile(models=[cfg])
        registry = ModelRegistry(profile)
        assert registry.get_model("bad") is None

    def test_summary_empty(self):
        registry = ModelRegistry(ModelProfile(models=[]))
        assert registry.summary()["total_models"] == 0

    def test_summary_with_models(self):
        registry = ModelRegistry()
        s = registry.summary()
        assert s["total_models"] == 3
        assert s["per_role"]["planner"] >= 1

    def test_get_model_for_role_with_str(self):
        registry = ModelRegistry()
        backend = registry.get_model("planner")
        assert backend is None or backend.name == "planner"

    def test_list_models_format(self):
        registry = ModelRegistry()
        models = registry.list_models()
        for m in models:
            assert "name" in m
            assert "role" in m
            assert "alive" in m
            assert m["backend"] == "ollama"

    def test_generate_no_healthy_model(self):
        profile = ModelProfile(models=[
            ModelConfig(name="p", model_id="nonexistent", role=ModelRole.PLANNER,
                        base_url="http://nonexistent:11434"),
        ])
        registry = ModelRegistry(profile)
        import asyncio
        result = asyncio.run(registry.generate(ModelRole.PLANNER, "hello"))
        assert result.error != ""
        assert result.text == ""


# ── GenerationResult tests ─────────────────────────────────────────────


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
