
from zilli.models.base import GenerationResult
from zilli.models.config import ModelConfig, ModelProfile, ModelRole
from zilli.models.registry import ModelRegistry
from zilli.routing.classifier import RouteClassifier, RouteDecision, RouteType

# ── Mock backend for routing tests ─────────────────────────────────────


class MockRouteBackend:
    def __init__(self, name: str, model_id: str, base_url: str = "http://mock:9999",
                 health: bool = True):
        self.name = name
        self.model_id = model_id
        self.base_url = base_url
        self._health = health
        self._call_count = 0

    async def generate(self, prompt: str, max_tokens: int = 2048,
                       temperature: float = 0.1) -> GenerationResult:
        self._call_count += 1
        return GenerationResult(
            text=f"[{self.name.upper()}] response to: {prompt[:60]}",
            model_name=self.model_id,
            tokens_in=len(prompt),
            tokens_out=50,
            duration_ms=50.0,
        )

    async def health_check(self) -> bool:
        return self._health


def _make_registry() -> ModelRegistry:
    profile = ModelProfile(models=[
        ModelConfig(name="planner", model_id="p-model", role=ModelRole.PLANNER),
        ModelConfig(name="executor", model_id="e-model", role=ModelRole.EXECUTOR),
        ModelConfig(name="reviewer", model_id="r-model", role=ModelRole.REVIEWER),
    ])
    import zilli.models.registry as reg
    reg.BACKEND_BUILDERS = {
        "ollama": lambda cfg: MockRouteBackend(cfg.name, cfg.model_id),
    }
    return ModelRegistry(profile)


# ── RouteClassifier tests ──────────────────────────────────────────────


class TestRouteClassifier:
    def setup_method(self):
        self.classifier = RouteClassifier()

    def test_full_route_complex(self):
        d = self.classifier.classify("我们需要设计一个复杂的金融风控系统")
        assert d.route == RouteType.FULL_ROUTE

    def test_full_route_english(self):
        d = self.classifier.classify("Design a comprehensive audit plan")
        assert d.route == RouteType.FULL_ROUTE

    def test_fast_lane_simple(self):
        d = self.classifier.classify("你好，今天天气怎么样")
        assert d.route == RouteType.FAST_LANE

    def test_fast_lane_basic(self):
        d = self.classifier.classify("What is machine learning")
        assert d.route == RouteType.FAST_LANE

    def test_long_request(self):
        d = self.classifier.classify("x" * 600)
        assert d.route == RouteType.FULL_ROUTE

    def test_decision_repr(self):
        d = RouteDecision(RouteType.FULL_ROUTE, "test reason")
        r = repr(d)
        assert "full_route" in r
        assert "test reason" in r


# ── LocalHybridRouter tests ────────────────────────────────────────────


class TestLocalHybridRouter:
    def test_classify_full_route(self):
        classifier = RouteClassifier()
        d = classifier.classify("设计一个数据分析方案")
        assert d.route == RouteType.FULL_ROUTE

    def test_classify_fast_lane(self):
        classifier = RouteClassifier()
        d = classifier.classify("Hello world")
        assert d.route == RouteType.FAST_LANE

    def test_force_full_route(self):
        classifier = RouteClassifier()
        d = classifier.classify("hi")
        assert d.route == RouteType.FAST_LANE
        forced = RouteDecision(RouteType.FULL_ROUTE, "forced")
        assert forced.route == RouteType.FULL_ROUTE
