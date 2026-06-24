use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use tracing::instrument;

use super::backends::{GenerationResult, ModelBackend, ModelError};
use super::config::{BackendType, ModelConfig, ModelProfile, ModelRole};

type BackendBuilder = Arc<dyn Fn(&ModelConfig) -> Arc<dyn ModelBackend> + Send + Sync>;

pub struct ModelRegistry {
    profile: RwLock<Option<ModelProfile>>,
    models: RwLock<HashMap<String, Arc<dyn ModelBackend>>>,
    pub(crate) builders: HashMap<BackendType, BackendBuilder>,
}

impl ModelRegistry {
    pub fn new() -> Self {
        let mut builders: HashMap<BackendType, BackendBuilder> = HashMap::new();

        builders.insert(
            BackendType::Ollama,
            Arc::new(|cfg: &ModelConfig| {
                let url = cfg
                    .base_url
                    .clone()
                    .unwrap_or_else(|| "http://localhost:11434".into());
                Arc::new(super::backends::ollama::OllamaBackend::new(
                    &cfg.name, &cfg.model_id, &url,
                ))
            }),
        );

        builders.insert(
            BackendType::Vllm,
            Arc::new(|cfg: &ModelConfig| {
                let url = cfg
                    .base_url
                    .clone()
                    .unwrap_or_else(|| "http://localhost:8000".into());
                Arc::new(super::backends::vllm::VLLMBackend::new(
                    &cfg.name, &cfg.model_id, &url,
                ))
            }),
        );

        builders.insert(
            BackendType::LlamaCpp,
            Arc::new(|cfg: &ModelConfig| {
                let url = cfg
                    .base_url
                    .clone()
                    .unwrap_or_else(|| "http://localhost:8080".into());
                Arc::new(super::backends::llamacpp::LlamaCppBackend::new(
                    &cfg.name, &cfg.model_id, &url,
                ))
            }),
        );

        Self {
            profile: RwLock::new(None),
            models: RwLock::new(HashMap::new()),
            builders,
        }
    }

    pub fn set_profile(&self, profile: ModelProfile) {
        *self.profile.write() = Some(profile);
    }

    pub fn register_model(&self, cfg: ModelConfig) -> Result<(), ModelError> {
        let builder = self
            .builders
            .get(&cfg.backend)
            .ok_or_else(|| ModelError::BackendUnavailable(format!("Unknown backend: {}", cfg.backend)))?;

        let backend = builder(&cfg);
        self.models.write().insert(cfg.name.clone(), backend);
        Ok(())
    }

    pub fn get_model(&self, name: &str) -> Option<Arc<dyn ModelBackend>> {
        self.models.read().get(name).cloned()
    }

    pub fn list_models(&self) -> Vec<serde_json::Value> {
        self.models
            .read()
            .iter()
            .map(|(name, backend)| {
                serde_json::json!({
                    "name": name,
                    "model_id": backend.model_id(),
                    "available": true,
                })
            })
            .collect()
    }

    #[instrument(skip(self, prompt))]
    pub async fn generate(
        &self,
        role: ModelRole,
        prompt: &str,
        max_tokens: Option<i32>,
        temperature: Option<f64>,
    ) -> Result<GenerationResult, ModelError> {
        let profile = self.profile.read().clone();
        let candidates: Vec<Arc<dyn ModelBackend>> = self
            .models
            .read()
            .iter()
            .filter_map(|(_, backend)| {
                let cfg = profile
                    .as_ref()
                    .and_then(|p| p.models.iter().find(|m| m.name == backend.name()));
                cfg.filter(|c| c.role == role).map(|_| backend.clone())
            })
            .collect();

        let backend = candidates
            .first()
            .ok_or(ModelError::ModelNotFound(format!("No model for role {}", role)))?;

        backend.generate(prompt, max_tokens, temperature).await
    }

    pub async fn health_check(&self) -> Vec<serde_json::Value> {
        let backends: Vec<(String, Arc<dyn ModelBackend>)> = self
            .models
            .read()
            .iter()
            .map(|(name, b)| (name.clone(), b.clone()))
            .collect();
        let mut results = Vec::new();
        for (name, backend) in backends {
            let healthy = backend.health_check().await.unwrap_or(false);
            results.push(serde_json::json!({
                "name": name,
                "healthy": healthy,
            }));
        }
        results
    }
}

impl Default for ModelRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::config::ModelProfile;

    fn make_executor_config(name: &str, model_id: &str) -> ModelConfig {
        ModelConfig {
            name: name.into(),
            backend: BackendType::Ollama,
            model_id: model_id.into(),
            role: ModelRole::Executor,
            deployment: crate::models::config::DeploymentType::Local,
            base_url: None,
            max_tokens: None,
            temperature: None,
            cost_per_call: None,
        }
    }

    struct SpyBackend {
        name: String,
        model_id: String,
        response_text: String,
    }

    impl SpyBackend {
        fn new(name: &str, model_id: &str, response_text: &str) -> Self {
            Self {
                name: name.into(),
                model_id: model_id.into(),
                response_text: response_text.into(),
            }
        }
    }

    impl std::fmt::Debug for SpyBackend {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("SpyBackend").field("name", &self.name).finish()
        }
    }

    #[async_trait::async_trait]
    impl ModelBackend for SpyBackend {
        fn name(&self) -> &str { &self.name }
        fn model_id(&self) -> &str { &self.model_id }
        async fn generate(&self, _prompt: &str, _max_tokens: Option<i32>, _temperature: Option<f64>) -> Result<GenerationResult, ModelError> {
            Ok(GenerationResult {
                text: self.response_text.clone(),
                model_name: self.model_id.clone(),
                tokens_in: 10,
                tokens_out: 5,
                duration_ms: 1,
                error: None,
            })
        }
        async fn generate_stream(&self, _prompt: &str, _max_tokens: Option<i32>, _temperature: Option<f64>) -> Result<Vec<String>, ModelError> {
            Ok(vec!["mock".into()])
        }
        async fn health_check(&self) -> Result<bool, ModelError> {
            Ok(true)
        }
    }

    #[tokio::test]
    async fn test_registry_empty() {
        let reg = ModelRegistry::new();
        assert!(reg.list_models().is_empty());
    }

    #[tokio::test]
    async fn test_registry_generate_no_profile() {
        let reg = ModelRegistry::new();
        let result = reg.generate(ModelRole::Executor, "hello", None, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No model for role"));
    }

    #[tokio::test]
    async fn test_registry_health_check_empty() {
        let reg = ModelRegistry::new();
        let health = reg.health_check().await;
        assert!(health.is_empty());
    }

    #[tokio::test]
    async fn test_registry_generate_with_registered_model() {
        let reg = ModelRegistry::new();
        let backend: Arc<dyn ModelBackend> = Arc::new(SpyBackend::new("executor-model", "exec-id", "mock output"));
        reg.models.write().insert("executor-model".into(), backend);
        reg.set_profile(ModelProfile {
            models: vec![make_executor_config("executor-model", "exec-id")],
            monthly_budget_usd: None,
        });

        let result = reg.generate(ModelRole::Executor, "hello", None, None).await;
        assert!(result.is_ok());
        let output = result.unwrap();
        assert_eq!(output.text, "mock output");
    }

    #[tokio::test]
    async fn test_registry_generate_role_mismatch() {
        let reg = ModelRegistry::new();
        let backend: Arc<dyn ModelBackend> = Arc::new(SpyBackend::new("planner-model", "plan-id", "plan output"));
        reg.models.write().insert("planner-model".into(), backend);
        reg.set_profile(ModelProfile {
            models: vec![make_executor_config("executor-model", "exec-id")],
            monthly_budget_usd: None,
        });

        let result = reg.generate(ModelRole::Planner, "plan this", None, None).await;
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("No model for role"));
    }

    #[tokio::test]
    async fn test_registry_health_check_with_model() {
        let reg = ModelRegistry::new();
        let backend: Arc<dyn ModelBackend> = Arc::new(SpyBackend::new("healthy-model", "h-id", ""));
        reg.models.write().insert("healthy-model".into(), backend);
        let health = reg.health_check().await;
        assert_eq!(health.len(), 1);
        assert_eq!(health[0]["name"], "healthy-model");
        assert_eq!(health[0]["healthy"], true);
    }

    #[tokio::test]
    async fn test_registry_register_model_unknown_backend() {
        let mut reg = ModelRegistry::new();
        // Manually clear builders to simulate missing backend
        reg.builders = HashMap::new();
        let cfg = ModelConfig {
            name: "bad-backend".into(),
            backend: BackendType::Ollama,
            model_id: "nope".into(),
            role: ModelRole::Executor,
            deployment: crate::models::config::DeploymentType::Local,
            base_url: None,
            max_tokens: None,
            temperature: None,
            cost_per_call: None,
        };
        let result = reg.register_model(cfg);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("Unknown backend"));
    }
}
