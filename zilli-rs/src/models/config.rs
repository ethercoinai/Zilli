use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};

#[derive(Debug, Clone, Serialize, Deserialize, Display, EnumString, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum ModelRole {
    Planner,
    Executor,
    Reviewer,
}

#[derive(Debug, Clone, Serialize, Deserialize, Display, EnumString, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum DeploymentType {
    Local,
    Cloud,
}

#[derive(Debug, Clone, Serialize, Deserialize, Display, EnumString, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum BackendType {
    Ollama,
    Vllm,
    LlamaCpp,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub name: String,
    pub backend: BackendType,
    pub model_id: String,
    pub role: ModelRole,
    pub deployment: DeploymentType,
    pub base_url: Option<String>,
    pub max_tokens: Option<i32>,
    pub temperature: Option<f64>,
    pub cost_per_call: Option<f64>,
}

impl ModelConfig {
    pub fn validate(&self) -> Result<(), Vec<String>> {
        let mut errors = Vec::new();
        if self.name.is_empty() {
            errors.push("name must not be empty".into());
        }
        if self.model_id.is_empty() {
            errors.push("model_id must not be empty".into());
        }
        if let Some(tokens) = self.max_tokens {
            if tokens <= 0 {
                errors.push(format!("max_tokens must be positive, got {}", tokens));
            }
        }
        if let Some(temp) = self.temperature {
            if temp < 0.0 || temp > 2.0 {
                errors.push(format!("temperature must be 0.0..2.0, got {}", temp));
            }
        }
        if let Some(cost) = self.cost_per_call {
            if cost < 0.0 {
                errors.push(format!("cost_per_call must be >= 0, got {}", cost));
            }
        }
        if errors.is_empty() { Ok(()) } else { Err(errors) }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelProfile {
    pub models: Vec<ModelConfig>,
    pub monthly_budget_usd: Option<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config_fields() {
        let cfg = ModelConfig {
            name: "test".into(),
            backend: BackendType::Ollama,
            model_id: "qwen3:4b".into(),
            role: ModelRole::Executor,
            deployment: DeploymentType::Local,
            base_url: None,
            max_tokens: None,
            temperature: None,
            cost_per_call: None,
        };
        assert_eq!(cfg.name, "test");
        assert_eq!(cfg.backend, BackendType::Ollama);
        assert_eq!(cfg.model_id, "qwen3:4b");
    }

    #[test]
    fn test_model_profile_empty() {
        let profile = ModelProfile { models: vec![], monthly_budget_usd: None };
        assert!(profile.models.is_empty());
        assert!(profile.monthly_budget_usd.is_none());
    }

    #[test]
    fn test_model_role_display() {
        assert_eq!(ModelRole::Planner.to_string(), "Planner");
        assert_eq!(ModelRole::Executor.to_string(), "Executor");
        assert_eq!(ModelRole::Reviewer.to_string(), "Reviewer");
    }

    #[test]
    fn test_deployment_type_display() {
        assert_eq!(DeploymentType::Local.to_string(), "Local");
        assert_eq!(DeploymentType::Cloud.to_string(), "Cloud");
    }

    #[test]
    fn test_model_config_validate_ok() {
        let cfg = ModelConfig {
            name: "test".into(),
            backend: BackendType::Ollama,
            model_id: "qwen3:4b".into(),
            role: ModelRole::Executor,
            deployment: DeploymentType::Local,
            base_url: None,
            max_tokens: Some(4096),
            temperature: Some(0.7),
            cost_per_call: Some(0.01),
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_model_config_validate_bad_temperature() {
        let cfg = ModelConfig {
            name: "test".into(),
            backend: BackendType::Ollama,
            model_id: "qwen3:4b".into(),
            role: ModelRole::Executor,
            deployment: DeploymentType::Local,
            base_url: None,
            max_tokens: Some(4096),
            temperature: Some(3.0),
            cost_per_call: None,
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.iter().any(|e| e.contains("temperature")));
    }

    #[test]
    fn test_model_config_validate_negative_tokens() {
        let cfg = ModelConfig {
            name: "test".into(),
            backend: BackendType::Ollama,
            model_id: "qwen3:4b".into(),
            role: ModelRole::Executor,
            deployment: DeploymentType::Local,
            base_url: None,
            max_tokens: Some(-1),
            temperature: None,
            cost_per_call: None,
        };
        let err = cfg.validate().unwrap_err();
        assert!(err.iter().any(|e| e.contains("max_tokens")));
    }
}
