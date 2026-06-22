from __future__ import annotations

from pydantic import BaseModel, Field


class RouteRequest(BaseModel):
    request: str = Field(..., min_length=1, max_length=65536)
    industry: str = ""
    force_full_route: bool = False


class RouteResponse(BaseModel):
    final_text: str
    route_type: str
    decision_reason: str
    planner_output: str | None = None
    executor_output: str | None = None
    reviewer_output: str | None = None
    total_duration_ms: float = 0.0
    error: str | None = None


class IndustryRequest(BaseModel):
    request: str = Field(..., min_length=1, max_length=65536)
    force_full_route: bool = False
    sanitize: bool = True
    tenant_id: str = "default"


class IndustryResponse(BaseModel):
    final_text: str
    route_type: str
    decision_reason: str
    industry: str
    tenant_id: str
    total_duration_ms: float = 0.0
    error: str | None = None


class ModelInfo(BaseModel):
    name: str
    backend: str
    model_id: str
    role: str
    alive: bool


class ModelHealth(BaseModel):
    name: str
    model_id: str
    status: str
    base_url: str


class CostStatus(BaseModel):
    remaining_budget: float
    total_calls: int
    calls_this_hour: int
    hourly_quota: float
    emergency_mode: bool


class HealthResponse(BaseModel):
    status: str
    version: str
    models_configured: int
    models_alive: int
    model_health: list[ModelHealth] = []


# ── OpenAI-compatible schemas ─────────────────────────────────────────


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: str = "zilli"
    messages: list[ChatMessage] = Field(..., min_length=1)
    max_tokens: int = 2048
    temperature: float = 0.1
    top_p: float = 0.9
    stop: list[str] | None = None
    n: int = 1
    stream: bool = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: str = "stop"


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: Usage


class OpenAIModel(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "zilli"


class OpenAIModelList(BaseModel):
    object: str = "list"
    data: list[OpenAIModel]
