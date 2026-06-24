use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use strum::{Display, EnumString};

use crate::security::patterns;

#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize, Display, EnumString,
)]
pub enum DataClass {
    Public,
    Internal,
    Confidential,
    Restricted,
    Regulated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub data_class: DataClass,
    pub requires_sanitization: bool,
    pub can_use_cloud: bool,
    pub findings: Vec<String>,
    pub confidence: f64,
}

pub struct DataClassifier {
    pii_patterns: Vec<(String, Regex, DataClass)>,
    keyword_classifiers: HashMap<DataClass, Vec<String>>,
}

impl DataClassifier {
    pub fn new() -> Self {
        let mut patterns_vec = Vec::new();
        patterns_vec.push((
            "email".into(),
            Regex::new(patterns::EMAIL).unwrap(),
            DataClass::Confidential,
        ));
        patterns_vec.push((
            "phone".into(),
            Regex::new(patterns::PHONE).unwrap(),
            DataClass::Confidential,
        ));
        patterns_vec.push((
            "ssn".into(),
            Regex::new(patterns::SSN).unwrap(),
            DataClass::Restricted,
        ));
        patterns_vec.push((
            "credit_card".into(),
            Regex::new(patterns::CREDIT_CARD).unwrap(),
            DataClass::Restricted,
        ));
        patterns_vec.push((
            "chinese_id".into(),
            Regex::new(patterns::CHINESE_ID).unwrap(),
            DataClass::Restricted,
        ));
        patterns_vec.push((
            "api_key".into(),
            Regex::new(patterns::API_KEY).unwrap(),
            DataClass::Confidential,
        ));
        patterns_vec.push((
            "medical_record".into(),
            Regex::new(patterns::MEDICAL_RECORD).unwrap(),
            DataClass::Regulated,
        ));
        patterns_vec.push((
            "address".into(),
            Regex::new(patterns::ADDRESS).unwrap(),
            DataClass::Confidential,
        ));

        let mut keywords = HashMap::new();
        keywords.insert(
            DataClass::Regulated,
            vec![
                "hipaa".into(),
                "phi".into(),
                "gdpr".into(),
                "pii".into(),
                "medical".into(),
                "health".into(),
                "diagnosis".into(),
            ],
        );
        keywords.insert(
            DataClass::Restricted,
            vec![
                "confidential".into(),
                "restricted".into(),
                "classified".into(),
                "secret".into(),
                "salary".into(),
            ],
        );
        keywords.insert(
            DataClass::Confidential,
            vec![
                "internal".into(),
                "private".into(),
                "personal".into(),
                "employee".into(),
            ],
        );

        Self {
            pii_patterns: patterns_vec,
            keyword_classifiers: keywords,
        }
    }

    pub fn classify(&self, text: &str) -> ClassificationResult {
        let mut max_class = DataClass::Public;
        let mut findings = Vec::new();
        let mut pii_count = 0;

        for (label, pattern, data_class) in &self.pii_patterns {
            for _match in pattern.find_iter(text) {
                pii_count += 1;
                let masked = patterns::mask_pii(_match.as_str());
                findings.push(format!("{}: {}", label, masked));
                if *data_class > max_class {
                    max_class = *data_class;
                }
            }
        }

        for (data_class, keywords) in &self.keyword_classifiers {
            for kw in keywords {
                if text.to_lowercase().contains(kw) {
                    findings.push(format!("keyword: {}", kw));
                    if *data_class > max_class {
                        max_class = *data_class;
                    }
                }
            }
        }

        let confidence = if pii_count > 0 {
            0.95_f64.min(0.7 + pii_count as f64 * 0.05)
        } else if !findings.is_empty() {
            0.8
        } else {
            0.3
        };

        let (requires_sanitization, can_use_cloud) = match max_class {
            DataClass::Public | DataClass::Internal => (false, true),
            DataClass::Confidential => (true, true),
            DataClass::Restricted | DataClass::Regulated => (true, false),
        };

        ClassificationResult {
            data_class: max_class,
            requires_sanitization,
            can_use_cloud,
            findings,
            confidence,
        }
    }
}

impl Default for DataClassifier {
    fn default() -> Self {
        Self::new()
    }
}
