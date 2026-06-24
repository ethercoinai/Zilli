use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tracing::instrument;

use super::{GenerationResult, ModelBackend, ModelError};

#[derive(Debug)]
pub struct OllamaBackend {
    name: String,
    model_id: String,
    base_url: String,
    client: Client,
}

#[derive(Serialize)]
struct GenerateRequest {
    model: String,
    prompt: String,
    stream: bool,
    options: Option<GenerateOptions>,
}

#[derive(Serialize)]
struct GenerateOptions {
    num_predict: Option<i32>,
    temperature: Option<f64>,
}

#[derive(Deserialize)]
struct GenerateResponse {
    response: String,
    done: bool,
    eval_count: Option<u64>,
    _eval_duration: Option<u64>,
}

#[derive(Deserialize)]
#[allow(dead_code)]
struct HealthResponse {
    status: Option<String>,
}

impl OllamaBackend {
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
impl ModelBackend for OllamaBackend {
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
        let request = GenerateRequest {
            model: self.model_id.clone(),
            prompt: prompt.to_string(),
            stream: false,
            options: Some(GenerateOptions {
                num_predict: max_tokens,
                temperature,
            }),
        };

        let resp = self
            .client
            .post(format!("{}/api/generate", self.base_url))
            .json(&request)
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(ModelError::BackendUnavailable(format!(
                "Ollama returned status {}",
                resp.status()
            )));
        }

        let body: GenerateResponse = resp.json().await?;
        let duration = start.elapsed().as_millis() as u64;

        Ok(GenerationResult {
            text: body.response,
            model_name: self.model_id.clone(),
            tokens_in: prompt.len() as u64 / 4,
            tokens_out: body.eval_count.unwrap_or(0),
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
        let request = GenerateRequest {
            model: self.model_id.clone(),
            prompt: prompt.to_string(),
            stream: true,
            options: Some(GenerateOptions {
                num_predict: max_tokens,
                temperature,
            }),
        };

        let resp = self
            .client
            .post(format!("{}/api/generate", self.base_url))
            .json(&request)
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(ModelError::BackendUnavailable(format!(
                "Ollama returned status {}",
                resp.status()
            )));
        }

        let mut chunks = Vec::new();
        let mut stream = resp.bytes_stream();

        use futures::StreamExt;
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
                if let Ok(partial) = serde_json::from_str::<GenerateResponse>(line) {
                    if partial.done {
                        break;
                    }
                    chunks.push(partial.response);
                }
            }
        }

        Ok(chunks)
    }

    async fn health_check(&self) -> Result<bool, ModelError> {
        let resp = self
            .client
            .get(format!("{}/api/tags", self.base_url))
            .send()
            .await?;

        Ok(resp.status().is_success())
    }
}
