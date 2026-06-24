use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use strum::{Display, EnumString};

use super::classifier::{ClassificationResult, DataClassifier, DataClass};
use super::consent::{ConsentManager, ConsentStatus, DataUse};
use super::policy::PolicyStore;
use super::reid::{ReIDAssessor, ReIDRisk};
use crate::security::PIIDetector;
use crate::security::Sanitizer;

#[derive(Debug, Clone, Serialize, Deserialize, Display, EnumString, PartialEq)]
pub enum SanitizationMode {
    None,
    Auto,
    Force,
    Strict,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyVerdict {
    pub passed: bool,
    pub data_class: DataClass,
    pub risk_score: f64,
    pub sanitized_text: Option<String>,
    pub classification: Option<ClassificationResult>,
}

pub struct PrivacyEngine {
    classifier: DataClassifier,
    detector: PIIDetector,
    sanitizer: Sanitizer,
    consent_manager: ConsentManager,
    reid_assessor: ReIDAssessor,
    _policy_store: PolicyStore,
}

impl PrivacyEngine {
    pub fn new(
        classifier: DataClassifier,
        detector: PIIDetector,
        sanitizer: Sanitizer,
        consent_manager: ConsentManager,
        policy_store: PolicyStore,
    ) -> Self {
        Self {
            classifier,
            detector,
            sanitizer,
            consent_manager,
            reid_assessor: ReIDAssessor,
            _policy_store: policy_store,
        }
    }

    pub fn evaluate(
        &self,
        text: &str,
        data_use: &DataUse,
        user_id: &str,
        sanitize: SanitizationMode,
    ) -> PrivacyVerdict {
        let classification = self.classifier.classify(text);

        let consent = self.consent_manager.check(user_id, data_use);
        if consent == ConsentStatus::Denied || consent == ConsentStatus::Revoked {
            return PrivacyVerdict {
                passed: false,
                data_class: classification.data_class,
                risk_score: 1.0,
                sanitized_text: None,
                classification: Some(classification),
            };
        }

        let pii_findings = self.detector.detect(text, true);
        let mut category_counts: HashMap<String, usize> = HashMap::new();
        for f in &pii_findings {
            let cat = format!("{:?}", f.category);
            *category_counts.entry(cat).or_insert(0) += 1;
        }
        let fields: Vec<(&str, usize)> = category_counts
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect();
        let reid = self.reid_assessor.assess(&fields, 100);
        let reid_score = if reid.risk >= ReIDRisk::High { 0.4 } else { 0.0 };

        let pii_risk = if pii_findings.is_empty() {
            0.0
        } else {
            (pii_findings.len() as f64 * 0.15).min(1.0)
        };
        let risk_score = (pii_risk + reid_score).min(1.0);

        let sanitized_text = if classification.requires_sanitization
            || sanitize == SanitizationMode::Force
            || sanitize == SanitizationMode::Strict
        {
            Some(self.sanitizer.sanitize(text))
        } else {
            None
        };

        let passed = match classification.data_class {
            DataClass::Regulated => sanitize == SanitizationMode::Strict,
            DataClass::Restricted => sanitize == SanitizationMode::Force || sanitize == SanitizationMode::Strict,
            DataClass::Confidential => true,
            DataClass::Internal | DataClass::Public => true,
        };

        PrivacyVerdict {
            passed,
            data_class: classification.data_class,
            risk_score,
            sanitized_text,
            classification: Some(classification),
        }
    }
}
