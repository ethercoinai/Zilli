from __future__ import annotations

import json
import os

import pytest
from fastapi.testclient import TestClient

from zilli.server.app import create_app


class MockBackend:
    def __init__(self, name: str, model_id: str, base_url: str = "http://mock:9999",
                 health: bool = True):
        self.name = name
        self.model_id = model_id
        self.base_url = base_url
        self._health = health

    async def generate(self, prompt: str, max_tokens: int = 2048,
                       temperature: float = 0.1):
        from zilli.models.base import GenerationResult
        return GenerationResult(
            text=f"[{self.name.upper()}] response",
            model_name=self.model_id,
            tokens_in=len(prompt),
            tokens_out=50,
            duration_ms=10.0,
        )

    async def generate_stream(self, prompt: str, max_tokens: int = 2048,
                              temperature: float = 0.1):
        for word in prompt.split()[:5]:
            yield word + " "

    async def health_check(self) -> bool:
        return self._health


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


@pytest.fixture
def client_with_auth():
    os.environ["ZILLI_API_KEYS"] = "test-key-123,test-key-456"
    app = create_app()
    with TestClient(app) as c:
        yield c
    del os.environ["ZILLI_API_KEYS"]


class TestHealth:
    def test_healthz(self, client):
        resp = client.get("/healthz")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    def test_v1_health(self, client):
        resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "version" in data
        assert "models_configured" in data


class TestModels:
    def test_list_models(self, client):
        resp = client.get("/v1/models/internal")
        assert resp.status_code == 200
        models = resp.json()
        assert isinstance(models, list)

    def test_model_health(self, client):
        resp = client.get("/v1/models/health")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)


class TestRoute:
    def test_route_fast_lane(self, client):
        resp = client.post("/v1/route", json={
            "request": "hello world",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "route_type" in data
        assert data["route_type"] in ("fast_lane", "full_route")

    def test_route_full_force(self, client):
        resp = client.post("/v1/route", json={
            "request": "design a system",
            "force_full_route": True,
        })
        assert resp.status_code in (200, 500)
        if resp.status_code == 500:
            assert "detail" in resp.json()

    def test_route_with_industry(self, client):
        resp = client.post("/v1/route", json={
            "request": "analyze this contract",
            "industry": "legal",
        })
        assert resp.status_code in (200, 500)

    def test_route_empty_request(self, client):
        resp = client.post("/v1/route", json={"request": ""})
        assert resp.status_code == 422

    def test_route_with_tenant(self, client):
        resp = client.post("/v1/route",
                           json={"request": "help"},
                           headers={"X-Tenant-ID": "acme_corp"})
        assert resp.status_code == 200


class TestIndustry:
    def test_industry_list_types(self, client):
        for ind in ("legal", "medical", "financial", "education"):
            resp = client.post(f"/v1/industry/{ind}", json={
                "request": f"a {ind} query",
            })
            assert resp.status_code in (200, 500)

    def test_industry_unknown(self, client):
        resp = client.post("/v1/industry/unknown", json={
            "request": "test",
        })
        assert resp.status_code == 400
        data = resp.json()
        assert "unknown" in data["detail"].lower()

    def test_industry_with_tenant(self, client):
        resp = client.post("/v1/industry/legal",
                           json={"request": "contract review"},
                           headers={"X-Tenant-ID": "law_firm_a"})
        assert resp.status_code in (200, 500)
        if resp.status_code == 200:
            data = resp.json()
            assert data["tenant_id"] == "law_firm_a"


class TestCost:
    def test_cost_status_not_configured(self, client):
        resp = client.get("/v1/cost/status")
        assert resp.status_code == 501

    def test_cost_reset_not_configured(self, client):
        resp = client.post("/v1/cost/reset-month")
        assert resp.status_code == 501


class TestCORS:
    def test_cors_headers(self, client):
        resp = client.options("/v1/health", headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        })
        assert resp.status_code == 200
        assert "access-control-allow-origin" in resp.headers

    def test_cors_blocks_unexpected_origin(self, client):
        resp = client.get("/v1/health", headers={
            "Origin": "http://evil.com",
        })
        assert "access-control-allow-origin" not in resp.headers


class TestRequestID:
    def test_response_has_request_id(self, client):
        resp = client.get("/v1/health")
        assert "x-request-id" in resp.headers

    def test_custom_request_id(self, client):
        resp = client.get("/v1/health", headers={"X-Request-ID": "my-custom-id"})
        assert resp.headers.get("x-request-id") == "my-custom-id"


class TestValidation:
    def test_route_request_too_long(self, client):
        resp = client.post("/v1/route", json={
            "request": "x" * 70000,
        })
        assert resp.status_code == 422

    def test_invalid_json(self, client):
        resp = client.post("/v1/route", content=b"not json",
                           headers={"Content-Type": "application/json"})
        assert resp.status_code == 422


class TestOpenAIChat:
    def test_chat_basic(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "zilli",
            "messages": [{"role": "user", "content": "hello"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"
        assert "choices" in data
        assert len(data["choices"]) > 0
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "usage" in data

    def test_chat_with_system(self, client):
        resp = client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "hi"},
            ],
        })
        assert resp.status_code == 200

    def test_chat_multi_turn(self, client):
        resp = client.post("/v1/chat/completions", json={
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "hi there"},
                {"role": "user", "content": "how are you"},
            ],
        })
        assert resp.status_code == 200

    def test_chat_empty_messages(self, client):
        resp = client.post("/v1/chat/completions", json={
            "messages": [],
        })
        assert resp.status_code == 422

    def test_chat_model_field(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "custom-model",
            "messages": [{"role": "user", "content": "test"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["model"] == "custom-model"

    def test_chat_usage(self, client):
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hello world"}],
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["usage"]["total_tokens"] > 0

    def test_chat_id_format(self, client):
        resp = client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "hi"}],
        })
        data = resp.json()
        assert data["id"].startswith("chatcmpl-")


class TestStreamingChat:
    def _get_lines(self, resp):
        return [line for line in resp.text.strip().split("\n") if line.startswith("data: ")]

    def test_stream_basic(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "zilli",
            "messages": [{"role": "user", "content": "hello world"}],
            "stream": True,
        })
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "text/event-stream; charset=utf-8"
        assert resp.headers.get("x-accel-buffering") == "no"

    def test_stream_events_are_json(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "zilli",
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
        })
        lines = self._get_lines(resp)
        assert len(lines) >= 2
        for line in lines:
            assert line.startswith("data: ")

    def test_stream_first_event(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "zilli",
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
        })
        lines = self._get_lines(resp)
        first_data = lines[0][6:]
        parsed = json.loads(first_data)
        assert parsed["object"] == "chat.completion.chunk"
        assert parsed["choices"][0]["delta"]["role"] == "assistant"
        assert parsed["choices"][0]["finish_reason"] is None

    def test_stream_ends_with_done(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "zilli",
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
        })
        lines = self._get_lines(resp)
        assert lines[-1] == "data: [DONE]"

    def test_stream_finish_reason(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "zilli",
            "messages": [{"role": "user", "content": "test"}],
            "stream": True,
        })
        lines = self._get_lines(resp)
        found_stop = False
        for line in reversed(lines):
            if line.startswith("data: ") and line != "data: [DONE]":
                data = json.loads(line[6:])
                if "choices" in data and "error" not in data:
                    assert data["choices"][0]["finish_reason"] == "stop"
                    found_stop = True
                    break
        assert found_stop, "No stop event found in stream"

    def test_stream_non_stream_fallback(self, client):
        resp = client.post("/v1/chat/completions", json={
            "model": "zilli",
            "messages": [{"role": "user", "content": "test"}],
            "stream": False,
        })
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "chat.completion"


class TestOpenAIModels:
    def test_list_models_openai_format(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)

    def test_get_single_model(self, client):
        resp = client.get("/v1/models/planner")
        assert resp.status_code == 200
        data = resp.json()
        assert data["id"] == "planner"
        assert data["object"] == "model"


class TestMetrics:
    def test_metrics_endpoint(self, client):
        resp = client.get("/v1/metrics")
        assert resp.status_code == 200
        data = resp.json()
        assert "requests_total" in data
        assert "uptime_seconds" in data
        assert "errors_total" in data


class TestCache:
    def test_cache_stats(self, client):
        resp = client.get("/v1/cache/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "entries" in data

    def test_cache_clear(self, client):
        resp = client.post("/v1/cache/clear")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"


class TestAuth:
    def test_auth_required_for_protected(self, client_with_auth):
        resp = client_with_auth.get("/v1/models/internal")
        assert resp.status_code == 401

    def test_auth_with_api_key_header(self, client_with_auth):
        resp = client_with_auth.get("/v1/models/internal", headers={"X-API-Key": "test-key-123"})
        assert resp.status_code == 200

    def test_auth_with_bearer_token(self, client_with_auth):
        resp = client_with_auth.get("/v1/models/internal", headers={"Authorization": "Bearer test-key-456"})
        assert resp.status_code == 200

    def test_auth_invalid_key(self, client_with_auth):
        resp = client_with_auth.get("/v1/models/internal", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 401

    def test_auth_health_bypasses_auth(self, client_with_auth):
        resp = client_with_auth.get("/v1/health")
        assert resp.status_code == 200

    def test_auth_docs_bypasses_auth(self, client_with_auth):
        resp = client_with_auth.get("/docs")
        assert resp.status_code == 200


class TestSwagger:
    def test_docs_available(self, client):
        resp = client.get("/docs")
        assert resp.status_code == 200
        assert "swagger" in resp.text.lower()

    def test_redoc_available(self, client):
        resp = client.get("/redoc")
        assert resp.status_code == 200

    def test_openapi_json(self, client):
        resp = client.get("/openapi.json")
        assert resp.status_code == 200
        data = resp.json()
        assert data["info"]["title"] == "Zilli API"


class TestDashboard:
    def test_dashboard_returns_html(self, client):
        resp = client.get("/dashboard")
        assert resp.status_code == 200
        assert "text/html" in resp.headers["content-type"]
        assert "Zilli" in resp.text

    def test_dashboard_contains_endpoints(self, client):
        resp = client.get("/dashboard")
        assert "/v1/health" in resp.text
        assert "/v1/route" in resp.text
        assert "/v1/chat/completions" in resp.text


class TestRunServer:
    def test_run_server_import(self):
        from zilli.server.app import run_server
        assert callable(run_server)
