use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingConfig {
    pub algorithm: String,
    pub clip_range: f64,
    pub kl_penalty: f64,
    pub is_weight_cap: f64,
    pub gamma: f64,
    pub gae_lambda: f64,
    pub entropy_coef: f64,
    pub vf_coef: f64,
    pub batch_size: i32,
    pub learning_rate: f64,
}

impl TrainingConfig {
    pub fn new(algorithm: &str) -> Self {
        match algorithm {
            "cispo" => Self {
                algorithm: "cispo".into(),
                clip_range: 0.2,
                kl_penalty: 0.1,
                is_weight_cap: 2.0,
                gamma: 0.99,
                gae_lambda: 0.95,
                entropy_coef: 0.01,
                vf_coef: 0.5,
                batch_size: 64,
                learning_rate: 3e-5,
            },
            "grpo" => Self {
                algorithm: "grpo".into(),
                clip_range: 0.2,
                kl_penalty: 0.0,
                is_weight_cap: 2.0,
                gamma: 1.0,
                gae_lambda: 1.0,
                entropy_coef: 0.01,
                vf_coef: 0.0,
                batch_size: 128,
                learning_rate: 3e-5,
            },
            _ => Self::default(),
        }
    }

    pub fn to_training_kwargs(&self) -> serde_json::Value {
        serde_json::json!({
            "clip_range": self.clip_range,
            "kl_penalty": self.kl_penalty,
            "gamma": self.gamma,
            "gae_lambda": self.gae_lambda,
            "entropy_coef": self.entropy_coef,
            "vf_coef": self.vf_coef,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
        })
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self::new("cispo")
    }
}
