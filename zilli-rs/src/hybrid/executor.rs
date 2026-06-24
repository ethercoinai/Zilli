use serde::{Deserialize, Serialize};
use tracing::instrument;

use super::gatekeeper::{ExecutionTarget, PrivacyGatekeeper};
use crate::models::config::ModelRole;
use crate::models::registry::ModelRegistry;
use crate::privacy::classifier::DataClassifier;
use crate::privacy::consent::DataUse;
use crate::security::Sanitizer;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HybridResult {
    pub text: String,
    pub target: ExecutionTarget,
    pub verdict: String,
    pub model_name: String,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub error: Option<String>,
    pub warnings: Vec<String>,
}

const _CLOUD_MODEL_PREFIX: &str = "cloud-";

pub struct HybridExecutor {
    gatekeeper: PrivacyGatekeeper,
    model_registry: ModelRegistry,
    sanitizer: Sanitizer,
    classifier: DataClassifier,
}

impl HybridExecutor {
    pub fn new(
        gatekeeper: PrivacyGatekeeper,
        model_registry: ModelRegistry,
    ) -> Self {
        Self {
            gatekeeper,
            model_registry,
            sanitizer: Sanitizer::new(),
            classifier: DataClassifier::new(),
        }
    }

    #[instrument(skip(self, prompt))]
    pub async fn execute(
        &self,
        prompt: &str,
        model_role: ModelRole,
        data_use: &DataUse,
        tenant_id: &str,
    ) -> HybridResult {
        let classification = self.classifier.classify(prompt);

        let decision = self.gatekeeper.decide(
            prompt,
            classification.data_class,
            data_use,
            tenant_id,
        );

        match decision.target {
            ExecutionTarget::Local | ExecutionTarget::LocalWithCloudFallback => {
                let result = self
                    .model_registry
                    .generate(model_role.clone(), prompt, None, None)
                    .await;

                match result {
                    Ok(res) => HybridResult {
                        text: res.text,
                        target: decision.target,
                        verdict: decision.verdict.clone(),
                        model_name: res.model_name,
                        tokens_in: res.tokens_in,
                        tokens_out: res.tokens_out,
                        error: None,
                        warnings: decision.warnings.clone(),
                    },
                Err(e) => {
                        if decision.target == ExecutionTarget::LocalWithCloudFallback {
                            let sanitized = self.sanitizer.sanitize(prompt);
                            let fallback = self
                                .model_registry
                                .generate(model_role, &sanitized, None, None)
                                .await;

                            match fallback {
                                Ok(res) => HybridResult {
                                    text: res.text,
                                    target: ExecutionTarget::Cloud,
                                    verdict: "cloud_fallback".into(),
                                    model_name: res.model_name,
                                    tokens_in: res.tokens_in,
                                    tokens_out: res.tokens_out,
                                    error: None,
                                    warnings: decision.warnings,
                                },
                                Err(fb_e) => HybridResult {
                                    text: String::new(),
                                    target: ExecutionTarget::Rejected,
                                    verdict: "fallback_failed".into(),
                                    model_name: String::new(),
                                    tokens_in: 0,
                                    tokens_out: 0,
                                    error: Some(format!("Local failed: {}, Fallback failed: {}", e, fb_e)),
                                    warnings: decision.warnings,
                                },
                            }
                        } else {
                            HybridResult {
                                text: String::new(),
                                target: decision.target,
                                verdict: decision.verdict,
                                model_name: String::new(),
                                tokens_in: 0,
                                tokens_out: 0,
                                error: Some(e.to_string()),
                                warnings: decision.warnings,
                            }
                        }
                    }
                }
            }
            ExecutionTarget::Cloud => {
                let result = self
                    .model_registry
                    .generate(model_role, prompt, None, None)
                    .await;

                match result {
                    Ok(res) => HybridResult {
                        text: res.text,
                        target: ExecutionTarget::Cloud,
                        verdict: decision.verdict.clone(),
                        model_name: res.model_name,
                        tokens_in: res.tokens_in,
                        tokens_out: res.tokens_out,
                        error: None,
                        warnings: decision.warnings,
                    },
                    Err(e) => HybridResult {
                        text: String::new(),
                        target: ExecutionTarget::Rejected,
                        verdict: "execution_failed".into(),
                        model_name: String::new(),
                        tokens_in: 0,
                        tokens_out: 0,
                        error: Some(e.to_string()),
                        warnings: decision.warnings,
                    },
                }
            }
            ExecutionTarget::Rejected => HybridResult {
                text: String::new(),
                target: ExecutionTarget::Rejected,
                verdict: decision.verdict,
                model_name: String::new(),
                tokens_in: 0,
                tokens_out: 0,
                error: Some(decision.reason.clone()),
                warnings: decision.warnings,
            },
        }
    }
}
