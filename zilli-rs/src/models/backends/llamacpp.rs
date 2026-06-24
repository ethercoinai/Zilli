use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tracing::instrument;

use super::{GenerationResult, ModelBackend, ModelError};

#[derive(Debug)]
pub struct LlamaCppBackend {
    name: String,
    model_id: String,
    base_url: String,
    client: Client,
}

#[derive(Serialize)]
struct CompletionRequest {
    prompt: String,
    n_predict: Option<i32>,
    temperature: Option<f64>,
    stream: bool,
}

#[derive(Deserialize)]
struct CompletionResponse {
    content: String,
    timings: Option<Timings>,
}

#[derive(Deserialize)]
struct Timings {
    predicted_n: Option<u64>,
    _predicted_ms: Option<f64>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct HealthResponse {
    status: Option<String>,
}

impl LlamaCppBackend {
    pub fn new(name: &str, model_id: &str, base_url: &str) -> Self {
        Self {
            name: name.to_string(),
            model_id: model_id.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            client: Client::builder()
                .timeout(Duration::from_secs(60))
                .build()
                .expect("valid reqwest client"),
        }
    }
}

#[async_trait]
impl ModelBackend for LlamaCppBackend {
    fn name(&self) -> &str {
        &self.name
    }

    fn model_id(&self) -> &str {
        &self.model_id
    }

    #[instrument(skip(self, prompt))]
    async fn generate(
        &self,
        prompt: &str,
        max_tokens: Option<i32>,
        temperature: Option<f64>,
    ) -> Result<GenerationResult, ModelError> {
        let start = Instant::now();

        let request = CompletionRequest {
            prompt: prompt.to_string(),
            n_predict: max_tokens,
            temperature,
            stream: false,
        };

        let resp = self
            .client
            .post(format!("{}/completion", self.base_url))
            .json(&request)
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(ModelError::BackendUnavailable(format!(
                "llama.cpp returned status {}",
                resp.status()
            )));
        }

        let body: CompletionResponse = resp.json().await?;
        let duration = start.elapsed().as_millis() as u64;

        Ok(GenerationResult {
            text: body.content,
            model_name: self.model_id.clone(),
            tokens_in: prompt.len() as u64 / 4,
            tokens_out: body.timings.as_ref().and_then(|t| t.predicted_n).unwrap_or(0),
            duration_ms: duration.max(1),
            error: None,
        })
    }

    async fn generate_stream(
        &self,
        prompt: &str,
        max_tokens: Option<i32>,
        temperature: Option<f64>,
    ) -> Result<Vec<String>, ModelError> {
        let request = CompletionRequest {
            prompt: prompt.to_string(),
            n_predict: max_tokens,
            temperature,
            stream: true,
        };

        let resp = self
            .client
            .post(format!("{}/completion", self.base_url))
            .json(&request)
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(ModelError::BackendUnavailable(format!(
                "llama.cpp returned status {}",
                resp.status()
            )));
        }

        let mut chunks = Vec::new();
        use futures::StreamExt;
        let mut stream = resp.bytes_stream();
        let mut buf = String::new();

        while let Some(chunk) = stream.next().await {
            let chunk = chunk?;
            buf.push_str(&String::from_utf8_lossy(&chunk));
            while let Some(line_end) = buf.find('\n') {
                let line: String = buf.drain(..=line_end).collect();
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }
                if let Ok(partial) = serde_json::from_str::<serde_json::Value>(line) {
                    if let Some(content) = partial["content"].as_str() {
                        if !content.is_empty() {
                            chunks.push(content.to_string());
                        }
                    }
                    if partial["stop"].as_bool().unwrap_or(false) {
                        break;
                    }
                }
            }
        }

        Ok(chunks)
    }

    async fn health_check(&self) -> Result<bool, ModelError> {
        let resp = self
            .client
            .get(format!("{}/health", self.base_url))
            .send()
            .await?;

        Ok(resp.status().is_success())
    }
}
