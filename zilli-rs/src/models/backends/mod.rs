use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

pub mod ollama;
pub mod vllm;
pub mod llamacpp;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerationResult {
    pub text: String,
    pub model_name: String,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub duration_ms: u64,
    pub error: Option<String>,
}

#[async_trait]
pub trait ModelBackend: Debug + Send + Sync {
    fn name(&self) -> &str;
    fn model_id(&self) -> &str;

    async fn generate(
        &self,
        prompt: &str,
        max_tokens: Option<i32>,
        temperature: Option<f64>,
    ) -> Result<GenerationResult, ModelError>;

    async fn generate_stream(
        &self,
        prompt: &str,
        max_tokens: Option<i32>,
        temperature: Option<f64>,
    ) -> Result<Vec<String>, ModelError>;

    async fn health_check(&self) -> Result<bool, ModelError>;
}

#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("HTTP request failed: {0}")]
    HttpError(#[from] reqwest::Error),

    #[error("Backend unavailable: {0}")]
    BackendUnavailable(String),

    #[error("Generation failed: {0}")]
    GenerationFailed(String),

    #[error("Model not found: {0}")]
    ModelNotFound(String),

    #[error("Timeout")]
    Timeout,

    #[error("Stream error: {0}")]
    StreamError(String),
}
