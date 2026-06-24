use async_trait::async_trait;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::time::{Duration, Instant};
use tracing::instrument;

use super::{GenerationResult, ModelBackend, ModelError};

#[derive(Debug)]
pub struct VLLMBackend {
    name: String,
    model_id: String,
    base_url: String,
    client: Client,
}

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<Message>,
    max_tokens: Option<i32>,
    temperature: Option<f64>,
    stream: bool,
}

#[derive(Serialize)]
struct Message {
    role: String,
    content: String,
}

#[derive(Deserialize)]
struct ChatResponse {
    choices: Vec<Choice>,
    usage: Option<Usage>,
}

#[derive(Deserialize)]
struct Choice {
    message: ResponseMessage,
}

#[derive(Deserialize)]
struct ResponseMessage {
    content: String,
}

#[derive(Deserialize)]
struct Usage {
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct ModelList {
    data: Vec<ModelInfo>,
}

#[derive(Deserialize)]
struct ModelInfo {
    id: String,
}

impl VLLMBackend {
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
impl ModelBackend for VLLMBackend {
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

        let request = ChatRequest {
            model: self.model_id.clone(),
            messages: vec![Message {
                role: "user".into(),
                content: prompt.to_string(),
            }],
            max_tokens,
            temperature,
            stream: false,
        };

        let resp = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&request)
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(ModelError::BackendUnavailable(format!(
                "vLLM returned status {}",
                resp.status()
            )));
        }

        let body: ChatResponse = resp.json().await?;
        let duration = start.elapsed().as_millis() as u64;

        let text = body
            .choices
            .first()
            .map(|c| c.message.content.clone())
            .unwrap_or_default();

        let tokens_in = body.usage.as_ref().and_then(|u| u.prompt_tokens).unwrap_or(0);
        let tokens_out = body
            .usage
            .as_ref()
            .and_then(|u| u.completion_tokens)
            .unwrap_or(0);

        Ok(GenerationResult {
            text,
            model_name: self.model_id.clone(),
            tokens_in,
            tokens_out,
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
        let request = ChatRequest {
            model: self.model_id.clone(),
            messages: vec![Message {
                role: "user".into(),
                content: prompt.to_string(),
            }],
            max_tokens,
            temperature,
            stream: true,
        };

        let resp = self
            .client
            .post(format!("{}/v1/chat/completions", self.base_url))
            .json(&request)
            .send()
            .await?;

        if !resp.status().is_success() {
            return Err(ModelError::BackendUnavailable(format!(
                "vLLM returned status {}",
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
                if line.is_empty() || line == "data: [DONE]" {
                    continue;
                }
                if let Some(data) = line.strip_prefix("data: ") {
                    if let Ok(partial) =
                        serde_json::from_str::<serde_json::Value>(data)
                    {
                        if let Some(choice) = partial["choices"][0]["delta"]["content"]
                            .as_str()
                        {
                            chunks.push(choice.to_string());
                        }
                    }
                }
            }
        }

        Ok(chunks)
    }

    async fn health_check(&self) -> Result<bool, ModelError> {
        let resp = self
            .client
            .get(format!("{}/v1/models", self.base_url))
            .send()
            .await?;

        if !resp.status().is_success() {
            return Ok(false);
        }

        let models: ModelList = resp.json().await?;
        Ok(models.data.iter().any(|m| m.id == self.model_id))
    }
}
