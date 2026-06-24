from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import AsyncIterator, Optional

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse

from zilli.configs import ZilliConfig, load_config
from zilli.models import ModelRegistry
from zilli.privacy.engine import PrivacyEngine, SanitizationMode
from zilli.routing import LocalHybridRouter, RouteClassifier
from zilli.server.dashboard import _DASHBOARD_HTML
from zilli.server.schemas import (
    ChatCompletionChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    CostStatus,
    HealthResponse,
    IndustryRequest,
    IndustryResponse,
    ModelHealth,
    ModelInfo,
    OpenAIModel,
    OpenAIModelList,
    RouteRequest,
    RouteResponse,
    Usage,
)
from zilli.version import version

logger = logging.getLogger("zilli.server")

_metrics: dict = {"requests_total": 0, "tokens_total": 0, "errors_total": 0}

@dataclass
class RateLimiter:
    requests: dict[str, deque[float]] = field(default_factory=lambda: defaultdict(lambda: deque(maxlen=100)))
    max_requests: int = 60
    window_seconds: float = 60.0
    _cleanup_interval: float = 300.0
    _last_cleanup: float = 0.0

    def _periodic_cleanup(self):
        now = time.time()
        if now - self._last_cleanup < self._cleanup_interval:
            return
        self._last_cleanup = now
        cutoff = now - self.window_seconds
        stale_keys = [k for k, dq in self.requests.items() if dq and dq[-1] <= cutoff]
        for k in stale_keys:
            del self.requests[k]

    def check(self, key: str) -> bool:
        now = time.time()
        self._periodic_cleanup()
        window_start = now - self.window_seconds
        dq = self.requests[key]
        while dq and dq[0] <= window_start:
            dq.popleft()
        if len(dq) >= self.max_requests:
            return False
        dq.append(now)
        return True


class ZilliAppState:
    def __init__(self, config: Optional[ZilliConfig] = None):
        self.config = config
        self.registry: Optional[ModelRegistry] = None
        self.classifier: Optional[RouteClassifier] = None
        self.router: Optional[LocalHybridRouter] = None
        self.privacy: Optional[PrivacyEngine] = None
        self.cost_controller = None
        self.api_keys: set[str] = set()
        self.rate_limiter = RateLimiter()

    def ensure_initialized(self):
        if self.registry is not None:
            return
        self.registry = ModelRegistry(config=self.config)
        self.classifier = RouteClassifier(
            model_registry=self.registry,
            config=self.config,
        )
        self.router = LocalHybridRouter(
            registry=self.registry,
            classifier=self.classifier,
            config=self.config,
        )
        self.privacy = PrivacyEngine()
        if self.config is not None:
            from zilli.envs.cost_controller import CostController
            self.cost_controller = CostController(config=self.config)

        keys_env = os.environ.get("ZILLI_API_KEYS", "")
        if keys_env:
            self.api_keys = set(k.strip() for k in keys_env.split(",") if k.strip())

    def verify_api_key(self, request: Request) -> str | None:
        if not self.api_keys:
            return None
        auth = request.headers.get("Authorization", "")
        api_key = request.headers.get("X-API-Key", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
            if token in self.api_keys:
                return token
        if api_key in self.api_keys:
            return api_key
        client_ip = request.client.host if request.client else "unknown"
        logger.warning("Unauthorized API access attempt from %s", client_ip)
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

    def check_rate_limit(self, key: str):
        if not self.rate_limiter.check(key):
            raise HTTPException(status_code=429, detail="Rate limit exceeded. Try again later.")


def create_app(config: Optional[ZilliConfig] = None) -> FastAPI:
    state = ZilliAppState(config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state.ensure_initialized()
        yield
        if state.router and hasattr(state.router, "cache") and state.router.cache:
            state.router.cache.clear()

    app = FastAPI(
        title="Zilli API",
        version=version,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    cors_origins_str = os.environ.get("ZILLI_CORS_ORIGINS", "http://127.0.0.1:8900,http://localhost:3000")
    cors_origins = [o.strip() for o in cors_origins_str.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=bool(cors_origins),
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["X-Request-ID"],
    )

    @app.middleware("http")
    async def _add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4())[:8])
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        _metrics["requests_total"] += 1
        return response

    @app.middleware("http")
    async def _auth_middleware(request: Request, call_next):
        try:
            if request.url.path not in (
                "/healthz", "/v1/health", "/docs", "/redoc", "/openapi.json",
                "/favicon.ico",
            ):
                state.verify_api_key(request)
                client_ip = request.client.host if request.client else "unknown"
                state.check_rate_limit(client_ip)
            return await call_next(request)
        except HTTPException as exc:
            return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})

    # ── Health ──────────────────────────────────────────────────────────

    @app.get("/healthz")
    @app.get("/v1/health")
    async def health():
        state.ensure_initialized()
        models = state.registry.list_models()
        alive_results = await asyncio.gather(*[_check_alive(state.registry, m["name"]) for m in models])
        alive = sum(1 for a in alive_results if a)
        return HealthResponse(
            status="ok",
            version=version,
            models_configured=len(models),
            models_alive=alive,
        )

    # ── Route ──────────────────────────────────────────────────────────

    @app.post("/v1/route", response_model=RouteResponse)
    async def route(body: RouteRequest, x_tenant_id: str = Header("default"),
                    request: Request = None):
        state.ensure_initialized()
        start = time.monotonic()

        verdict = state.privacy.evaluate(
            body.request, tenant_id=x_tenant_id, mode=SanitizationMode.AUTO,
        )
        if not verdict.passed:
            raise HTTPException(status_code=403, detail="Request blocked by privacy policy")
        input_text = verdict.sanitized_text

        try:
            result = await state.router.run(
                request=input_text,
                industry=body.industry,
                force_full_route=body.force_full_route,
            )
        except Exception as e:
            _metrics["errors_total"] += 1
            logger.error("Route error: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")

        duration = (time.monotonic() - start) * 1000

        return RouteResponse(
            final_text=result.final_text,
            route_type=result.route_type.value,
            decision_reason=result.decision.reason,
            planner_output=result.planner_result,
            executor_output=result.executor_result,
            reviewer_output=result.reviewer_result,
            total_duration_ms=duration,
            error=result.error,
        )

    # ── Industry ────────────────────────────────────────────────────────

    @app.post("/v1/industry/{industry_type}", response_model=IndustryResponse)
    async def industry_route(
        industry_type: str,
        body: IndustryRequest,
        x_tenant_id: str = Header("default"),
    ):
        from zilli.industry import IndustryType, WorkflowRegistry

        try:
            ind = IndustryType(industry_type)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown industry: {industry_type}. "
                       f"Supported: legal, medical, financial, education",
            )

        state.ensure_initialized()
        registry = WorkflowRegistry(
            model_registry=state.registry,
            config=state.config,
        )

        try:
            result = await registry.run(
                request=body.request,
                industry=ind,
                tenant_id=x_tenant_id,
                force_full_route=body.force_full_route,
                sanitize=body.sanitize,
            )
        except Exception as e:
            _metrics["errors_total"] += 1
            logger.error("Industry route error: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")

        return IndustryResponse(
            final_text=result.final_text,
            route_type=result.route_type.value,
            decision_reason=result.decision.reason,
            industry=industry_type,
            tenant_id=x_tenant_id,
            total_duration_ms=result.total_duration_ms,
            error=result.error,
        )

    # ── Models ──────────────────────────────────────────────────────────

    @app.get("/v1/models/internal", response_model=list[ModelInfo])
    async def list_models():
        state.ensure_initialized()
        return [
            ModelInfo(**m) for m in state.registry.list_models()
        ]

    @app.get("/v1/models/health", response_model=list[ModelHealth])
    async def model_health():
        state.ensure_initialized()
        results = []
        for cfg in state.registry.profile.models:
            backend = state.registry.get_model(cfg.name)
            alive = backend is not None and await backend.health_check()
            results.append(ModelHealth(
                name=cfg.name,
                model_id=cfg.model_id,
                status="healthy" if alive else "unreachable",
                base_url=cfg.base_url,
            ))
        return results

    # ── Cost ────────────────────────────────────────────────────────────

    @app.get("/v1/cost/status", response_model=CostStatus)
    async def cost_status():
        state.ensure_initialized()
        cc = state.cost_controller
        if cc is None:
            raise HTTPException(status_code=501, detail="Cost controller not configured")
        snap = cc.snapshot()
        return CostStatus(
            remaining_budget=snap.remaining_budget,
            total_calls=snap.total_calls,
            calls_this_hour=snap.calls_this_hour,
            hourly_quota=snap.hourly_quota,
            emergency_mode=snap.emergency_mode,
        )

    @app.post("/v1/cost/reset-month")
    async def cost_reset():
        state.ensure_initialized()
        cc = state.cost_controller
        if cc is None:
            raise HTTPException(status_code=501, detail="Cost controller not configured")
        cc.reset_monthly()
        return {"status": "ok"}

    # ── Metrics ──────────────────────────────────────────────────────────

    @app.get("/v1/metrics")
    async def metrics():
        return {
            **_metrics,
            "uptime_seconds": time.time() - _start_time,
        }

    # ── OpenAI-compatible ─────────────────────────────────────────────

    @app.post("/v1/chat/completions")
    async def chat_completions(body: ChatCompletionRequest,
                               x_tenant_id: str = Header("default"),
                               request: Request = None):
        state.ensure_initialized()
        start = time.time()

        if body.stream:
            return StreamingResponse(
                _stream_chat(body, state),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        prompt = _messages_to_prompt(body.messages)

        verdict = state.privacy.evaluate(
            prompt, tenant_id=x_tenant_id, mode=SanitizationMode.AUTO,
        )
        if not verdict.passed:
            raise HTTPException(status_code=403, detail="Request blocked by privacy policy")

        try:
            result = await state.router.run(request=verdict.sanitized_text, force_full_route=False)
        except Exception as e:
            _metrics["errors_total"] += 1
            logger.error("Chat completion error: %s", e, exc_info=True)
            raise HTTPException(status_code=500, detail="Internal server error")

        response_text = result.final_text or ""

        _metrics["tokens_total"] += (result.executor_tokens or len(prompt) // 4) + len(response_text) // 4

        return ChatCompletionResponse(
            id=f"chatcmpl-{int(start)}",
            created=int(start),
            model=body.model,
            choices=[
                ChatCompletionChoice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response_text),
                ),
            ],
            usage=Usage(
                prompt_tokens=result.executor_tokens or len(prompt) // 4,
                completion_tokens=len(response_text) // 4,
                total_tokens=(result.executor_tokens or len(prompt) // 4) + len(response_text) // 4,
            ),
        )

    @app.get("/v1/models", response_model=OpenAIModelList)
    @app.get("/v1/models/{model_id}", response_model=OpenAIModel)
    async def openai_models(model_id: Optional[str] = None):
        state.ensure_initialized()
        if model_id:
            return OpenAIModel(id=model_id, created=int(time.time()))
        models = []
        for m in state.registry.list_models():
            models.append(OpenAIModel(
                id=m["name"], created=int(time.time()),
            ))
        return OpenAIModelList(data=models)

    # ── Cache ──────────────────────────────────────────────────────────

    @app.get("/v1/cache/stats")
    async def cache_stats():
        state.ensure_initialized()
        if state.router and hasattr(state.router, "cache") and state.router.cache:
            stats = state.router.cache.stats()
            return {
                "entries": stats.entries,
                "hits": stats.hits,
                "misses": stats.misses,
                "memory_entries": stats.memory_entries,
                "disk_entries": stats.disk_entries,
            }
        return {"entries": 0, "hits": 0, "misses": 0}

    @app.post("/v1/cache/clear")
    async def cache_clear():
        state.ensure_initialized()
        if state.router and hasattr(state.router, "cache") and state.router.cache:
            state.router.cache.clear()
        return {"status": "ok"}

    # ── Dashboard ─────────────────────────────────────────────────────

    @app.get("/dashboard")
    async def dashboard():
        return HTMLResponse(content=_DASHBOARD_HTML)

    return app


async def _stream_chat(body: ChatCompletionRequest, state: ZilliAppState) -> AsyncIterator[str]:
    prompt = _messages_to_prompt(body.messages)
    start = time.time()
    request_id = f"chatcmpl-{int(start)}"

    verdict = state.privacy.evaluate(
        prompt, tenant_id="default", mode=SanitizationMode.AUTO,
    )
    if not verdict.passed:
        yield f"data: {json.dumps({'error': {'message': 'Blocked by privacy policy'}})}\n\n"
        yield "data: [DONE]\n\n"
        return
    safe_prompt = verdict.sanitized_text

    yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': int(start), 'model': body.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"

    try:
        result = await state.router.run(request=safe_prompt, force_full_route=False)
        text = result.final_text or ""
        yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': int(start), 'model': body.model, 'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]})}\n\n"
        yield f"data: {json.dumps({'id': request_id, 'object': 'chat.completion.chunk', 'created': int(start), 'model': body.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
        _metrics["tokens_total"] += len(text) // 4
    except Exception as e:
        logger.error("Stream error: %s", e, exc_info=True)
        _metrics["errors_total"] += 1
        yield f"data: {json.dumps({'error': {'message': 'Internal server error'}})}\n\n"

    yield "data: [DONE]\n\n"


def _messages_to_prompt(messages: list[ChatMessage]) -> str:
    parts = []
    for m in messages:
        if m.role == "system":
            parts.append(f"System: {m.content}")
        elif m.role == "user":
            parts.append(f"User: {m.content}")
        elif m.role == "assistant":
            parts.append(f"Assistant: {m.content}")
    return "\n".join(parts)


async def _check_alive(registry, name: str) -> bool:
    backend = registry.get_model(name) if registry else None
    if backend is None:
        return False
    try:
        return await backend.health_check()
    except Exception:
        return False


_start_time = time.time()


def run_server(
    host: str = "127.0.0.1",
    port: int = 8900,
    config: Optional[ZilliConfig] = None,
    config_path: Optional[Path] = None,
):
    import uvicorn
    if config is None and config_path is not None:
        config = load_config(config_path)
    app = create_app(config)
    uvicorn.run(app, host=host, port=port, log_level="info")
