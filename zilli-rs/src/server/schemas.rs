use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteRequest {
    pub request: String,
    pub industry: Option<String>,
    pub force_full_route: Option<bool>,
    pub sanitize: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteResponse {
    pub final_text: String,
    pub route_type: String,
    pub decision: String,
    pub total_duration_ms: u64,
    pub error: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndustryRequest {
    pub request: String,
    pub industry: String,
    pub tenant_id: Option<String>,
    pub force_full_route: Option<bool>,
    pub sanitize: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    pub max_tokens: Option<i32>,
    pub temperature: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionChoice {
    pub index: i32,
    pub message: ChatMessage,
    pub finish_reason: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChatCompletionChoice>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostStatus {
    pub remaining_budget: f64,
    pub total_calls: u64,
    pub planner_calls: u64,
    pub executor_calls: u64,
    pub emergency_mode: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub models_healthy: Vec<serde_json::Value>,
}
