use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::models::config::ModelProfile;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierRule {
    pub pattern: String,
    pub route: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassifierConfig {
    pub rules: Vec<ClassifierRule>,
    pub long_request_threshold: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingConfig {
    pub classifier: Option<ClassifierConfig>,
    pub planner: Option<String>,
    pub executor: Option<String>,
    pub reviewer: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIIConfig {
    pub enabled: bool,
    pub custom_patterns: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecurityConfig {
    pub pii: Option<PIIConfig>,
    pub max_input_length: Option<i32>,
    pub sanitize_by_default: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditConfig {
    pub enabled: bool,
    pub log_dir: Option<String>,
    pub sanitize: Option<bool>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfigSection {
    pub algorithm: String,
    pub batch_size: Option<i32>,
    pub learning_rate: Option<f64>,
    pub checkpoint_interval: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ZilliConfig {
    pub version: String,
    pub models: Option<Vec<ModelProfile>>,
    pub routing: Option<RoutingConfig>,
    pub security: Option<SecurityConfig>,
    pub audit: Option<AuditConfig>,
    pub training: Option<TrainingConfigSection>,
}

impl ZilliConfig {
    pub fn from_yaml(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        let content = std::fs::read_to_string(path)?;
        let config: ZilliConfig = serde_yaml::from_str(&content)?;
        Ok(config)
    }

    pub fn to_model_profile(&self) -> Option<ModelProfile> {
        self.models.as_ref().and_then(|profiles| profiles.first().cloned())
    }
}

impl Default for ZilliConfig {
    fn default() -> Self {
        Self {
            version: "0.3.0".into(),
            models: None,
            routing: None,
            security: None,
            audit: None,
            training: None,
        }
    }
}

pub fn load_config(path: Option<&str>) -> Result<ZilliConfig, String> {
    match path {
        Some(path) if Path::new(path).exists() => {
            ZilliConfig::from_yaml(path).map_err(|e| {
                tracing::error!("Failed to parse config file {}: {}", path, e);
                format!("Config parse error: {}", e)
            })
        }
        Some(path) => {
            tracing::warn!("Config file not found: {}, using defaults", path);
            Ok(ZilliConfig::default())
        }
        None => Ok(ZilliConfig::default()),
    }
}
