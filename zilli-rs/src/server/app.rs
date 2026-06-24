use axum::{
    extract::State,
    routing::{get, post},
    Json, Router,
};

use super::schemas::*;
use crate::models::config::ModelRole;
use crate::models::registry::ModelRegistry;
use crate::security::isolation::DataIsolation;
use crate::security::sanitizer::Sanitizer;

const DEFAULT_TENANT: &str = "default";
const DEFAULT_ROLE: &str = "user";

#[derive(Clone)]
pub struct AppState {
    pub model_registry: std::sync::Arc<ModelRegistry>,
    pub data_isolation: std::sync::Arc<DataIsolation>,
    pub sanitizer: std::sync::Arc<Sanitizer>,
}

async fn health_handler(
    State(state): State<AppState>,
) -> Json<HealthResponse> {
    let models_healthy = state.model_registry.health_check().await;

    Json(HealthResponse {
        status: "ok".into(),
        version: "0.3.0".into(),
        models_healthy,
    })
}

async fn models_handler(
    State(state): State<AppState>,
) -> Json<Vec<serde_json::Value>> {
    Json(state.model_registry.list_models())
}

async fn chat_completion(
    State(state): State<AppState>,
    Json(req): Json<ChatCompletionRequest>,
) -> Json<ChatCompletionResponse> {
    let prompt = req
        .messages
        .last()
        .map(|m| m.content.as_str())
        .unwrap_or("");

    if !state.data_isolation.check_access(DEFAULT_TENANT, DEFAULT_ROLE) {
        return Json(ChatCompletionResponse {
            id: format!("chatcmpl-{:x}", rand::random::<u64>()),
            object: "chat.completion".into(),
            created: chrono::Utc::now().timestamp(),
            model: req.model.clone(),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: "Access denied: role not permitted".into(),
                },
                finish_reason: Some("error".into()),
            }],
        });
    }

    if !state.data_isolation.check_input_length(DEFAULT_TENANT, prompt.len()) {
        return Json(ChatCompletionResponse {
            id: format!("chatcmpl-{:x}", rand::random::<u64>()),
            object: "chat.completion".into(),
            created: chrono::Utc::now().timestamp(),
            model: req.model.clone(),
            choices: vec![ChatCompletionChoice {
                index: 0,
                message: ChatMessage {
                    role: "assistant".into(),
                    content: "Input exceeds maximum allowed length".into(),
                },
                finish_reason: Some("error".into()),
            }],
        });
    }

    let sanitized = if state.data_isolation.needs_sanitization(DEFAULT_TENANT) {
        state.sanitizer.sanitize(prompt)
    } else {
        prompt.to_string()
    };

    let result = state
        .model_registry
        .generate(
            ModelRole::Executor,
            &sanitized,
            req.max_tokens,
            req.temperature,
        )
        .await;

    let (content, finish_reason) = match result {
        Ok(res) => (res.text, Some("stop".into())),
        Err(e) => (format!("Error: {}", e), Some("error".into())),
    };

    Json(ChatCompletionResponse {
        id: format!("chatcmpl-{:x}", rand::random::<u64>()),
        object: "chat.completion".into(),
        created: chrono::Utc::now().timestamp(),
        model: req.model,
        choices: vec![ChatCompletionChoice {
            index: 0,
            message: ChatMessage {
                role: "assistant".into(),
                content,
            },
            finish_reason,
        }],
    })
}

async fn cost_handler(State(state): State<AppState>) -> Json<CostStatus> {
    let models = state.model_registry.list_models();
    Json(CostStatus {
        remaining_budget: 1000.0,
        total_calls: models.len() as u64,
        planner_calls: 0,
        executor_calls: models.len() as u64,
        emergency_mode: false,
    })
}

pub fn create_app(state: AppState) -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/models", get(models_handler))
        .route("/v1/chat/completions", post(chat_completion))
        .route("/cost", get(cost_handler))
        .with_state(state)
}

pub async fn run_server(host: &str, port: u16) -> Result<(), Box<dyn std::error::Error>> {
    let model_registry = std::sync::Arc::new(ModelRegistry::new());
    let data_isolation = std::sync::Arc::new(DataIsolation::new());
    let sanitizer = std::sync::Arc::new(Sanitizer::new());
    let state = AppState {
        model_registry,
        data_isolation,
        sanitizer,
    };

    let app = create_app(state);

    let addr = format!("{}:{}", host, port);
    let listener = tokio::net::TcpListener::bind(&addr).await?;

    tracing::info!("Zilli server listening on {}", addr);

    axum::serve(listener, app).await?;
    Ok(())
}
