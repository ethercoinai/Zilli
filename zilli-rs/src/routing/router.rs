use serde::{Deserialize, Serialize};
use std::time::Instant;
use tracing::instrument;

use super::classifier::{RouteClassifier, RouteType};
use crate::cache::engine::CacheEngine;
use crate::models::config::ModelRole;
use crate::models::registry::ModelRegistry;
use crate::privacy::consent::{ConsentManager, ConsentStatus, DataUse};
use crate::security::{OutputSanitizer, Sanitizer};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteResult {
    pub final_text: String,
    pub route_type: RouteType,
    pub decision: String,
    pub planner_result: Option<String>,
    pub executor_result: Option<String>,
    pub reviewer_result: Option<String>,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub total_duration_ms: u64,
    pub error: Option<String>,
}

pub struct LocalHybridRouter {
    model_registry: ModelRegistry,
    classifier: RouteClassifier,
    sanitizer: Sanitizer,
    output_sanitizer: OutputSanitizer,
    cache: Option<CacheEngine>,
    consent_manager: ConsentManager,
    default_model_name: String,
}

impl LocalHybridRouter {
    pub fn new(
        model_registry: ModelRegistry,
        classifier: RouteClassifier,
        cache: Option<CacheEngine>,
        consent_manager: ConsentManager,
    ) -> Self {
        let default_model_name = model_registry
            .list_models()
            .first()
            .and_then(|m| m.get("name").and_then(|n| n.as_str()))
            .unwrap_or("default")
            .to_string();
        Self {
            model_registry,
            classifier,
            sanitizer: Sanitizer::new(),
            output_sanitizer: OutputSanitizer::new(),
            cache,
            consent_manager,
            default_model_name,
        }
    }

    #[instrument(skip(self, request))]
    pub async fn run(
        &self,
        request: &str,
        industry: &str,
        force_full_route: bool,
        tenant_id: &str,
        data_use: &DataUse,
    ) -> RouteResult {
        let _start = Instant::now();

        let consent = self.consent_manager.check(tenant_id, data_use);
        if consent == ConsentStatus::Denied || consent == ConsentStatus::Revoked {
            return RouteResult {
                final_text: String::new(),
                route_type: RouteType::FastLane,
                decision: "consent_denied".into(),
                planner_result: None,
                executor_result: None,
                reviewer_result: None,
                tokens_in: 0,
                tokens_out: 0,
                total_duration_ms: 0,
                error: Some(format!("Consent {} for user {} on data use {:?}", consent, tenant_id, data_use)),
            };
        }

        let sanitized = self.sanitizer.sanitize(request);
        tracing::debug!(original_len = request.len(), sanitized_len = sanitized.len(), "request sanitized");

        let decision = self.classifier.classify(&sanitized, industry);

        if let Some(ref cache) = self.cache {
            if let Some(entry) = cache.get(&sanitized, &self.default_model_name, 0.0, None) {
                let output_check = self.output_sanitizer.check(&entry.response_text);
                tracing::warn!(warnings = ?output_check.warnings, "output sanitizer triggered on cached response");
                let final_text = output_check.sanitized.unwrap_or(entry.response_text.clone());
                return RouteResult {
                    final_text,
                    route_type: RouteType::FastLane,
                    decision: "cached".into(),
                    planner_result: None,
                    executor_result: None,
                    reviewer_result: None,
                    tokens_in: entry.tokens_in,
                    tokens_out: entry.tokens_out,
                    total_duration_ms: 0,
                    error: None,
                };
            }
        }

        let route = if force_full_route {
            RouteType::FullRoute
        } else {
            decision.route.clone()
        };

        let result = match route {
            RouteType::FastLane => self.run_fast_lane(&sanitized).await,
            RouteType::FullRoute => {
                self.run_full_route(&sanitized, &decision.reason).await
            }
        };

        if let Some(ref cache) = self.cache {
            if result.error.is_none() {
                if let Some(ref text) = result.executor_result {
                    let output_check = self.output_sanitizer.check(text);
                    if output_check.warnings.is_empty() {
                        cache.set(&sanitized, &self.default_model_name, text, result.tokens_in, result.tokens_out, 0.0, None);
                    }
                }
            }
        }

        result
    }

    async fn run_fast_lane(&self, request: &str) -> RouteResult {
        let start = Instant::now();

        let result = self
            .model_registry
            .generate(ModelRole::Executor, request, None, None)
            .await;

        match result {
            Ok(res) => {
                let elapsed = start.elapsed().as_millis() as u64;
                let output_check = self.output_sanitizer.check(&res.text);
                let final_text = output_check.sanitized.unwrap_or_else(|| res.text.clone());
                if !output_check.warnings.is_empty() {
                    tracing::warn!(warnings = ?output_check.warnings, "output sanitizer triggered");
                }
                RouteResult {
                    final_text,
                    route_type: RouteType::FastLane,
                    decision: "executor_only".into(),
                    planner_result: None,
                    executor_result: Some(res.text),
                    reviewer_result: None,
                    tokens_in: res.tokens_in,
                    tokens_out: res.tokens_out,
                    total_duration_ms: elapsed,
                    error: None,
                }
            }
            Err(e) => RouteResult {
                final_text: String::new(),
                route_type: RouteType::FastLane,
                decision: "executor_failed".into(),
                planner_result: None,
                executor_result: None,
                reviewer_result: None,
                tokens_in: 0,
                tokens_out: 0,
                total_duration_ms: start.elapsed().as_millis() as u64,
                error: Some(e.to_string()),
            },
        }
    }

    async fn run_full_route(&self, request: &str, reason: &str) -> RouteResult {
        let start = Instant::now();

        let planner = self
            .model_registry
            .generate(ModelRole::Planner, request, None, None)
            .await;

        let (planner_text, planner_error) = match planner {
            Ok(res) => (Some(res.text.clone()), None),
            Err(e) => (None, Some(e.to_string())),
        };

        let executor_request = planner_text
            .as_deref()
            .unwrap_or(request);

        let executor = self
            .model_registry
            .generate(ModelRole::Executor, executor_request, None, None)
            .await;

        let (executor_text, executor_error) = match executor {
            Ok(res) => (Some(res.text.clone()), None),
            Err(e) => (None, Some(e.to_string())),
        };

        let raw_text = executor_text.clone().unwrap_or_default();
        let output_check = self.output_sanitizer.check(&raw_text);
        let final_text = output_check.sanitized.unwrap_or(raw_text);
        if !output_check.warnings.is_empty() {
            tracing::warn!(warnings = ?output_check.warnings, "output sanitizer triggered on full route");
        }
        let error = planner_error.or(executor_error);

        let duration = start.elapsed().as_millis() as u64;

        RouteResult {
            final_text,
            route_type: RouteType::FullRoute,
            decision: reason.to_string(),
            planner_result: planner_text,
            executor_result: executor_text,
            reviewer_result: None,
            tokens_in: 0,
            tokens_out: 0,
            total_duration_ms: duration,
            error,
        }
    }
}
