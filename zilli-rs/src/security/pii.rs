use regex::Regex;
use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};

use super::patterns;

#[derive(Debug, Clone, Serialize, Deserialize, Display, EnumString, PartialEq, Hash, Eq)]
pub enum PIICategory {
    Name,
    IdNumber,
    Phone,
    Email,
    Address,
    Ssn,
    CreditCard,
    Passport,
    Dob,
    MedicalRecord,
    BankAccount,
    IpAddress,
    ApiKey,
    ChineseId,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PIEFinding {
    pub category: PIICategory,
    pub text: String,
    pub start: usize,
    pub end: usize,
    pub confidence: f64,
}

#[derive(Clone)]
pub struct PIIDetector {
    patterns: Vec<(PIICategory, Regex)>,
}

impl PIIDetector {
    pub fn new() -> Self {
        let mut pattern_vec = Vec::new();

        pattern_vec.push((
            PIICategory::Email,
            Regex::new(patterns::EMAIL).unwrap(),
        ));
        pattern_vec.push((
            PIICategory::Phone,
            Regex::new(patterns::PHONE).unwrap(),
        ));
        pattern_vec.push((
            PIICategory::Ssn,
            Regex::new(patterns::SSN).unwrap(),
        ));
        pattern_vec.push((
            PIICategory::CreditCard,
            Regex::new(patterns::CREDIT_CARD).unwrap(),
        ));
        pattern_vec.push((
            PIICategory::ChineseId,
            Regex::new(patterns::CHINESE_ID).unwrap(),
        ));
        pattern_vec.push((
            PIICategory::ApiKey,
            Regex::new(patterns::API_KEY).unwrap(),
        ));
        pattern_vec.push((
            PIICategory::IpAddress,
            Regex::new(patterns::IP).unwrap(),
        ));
        pattern_vec.push((
            PIICategory::Address,
            Regex::new(patterns::ADDRESS).unwrap(),
        ));
        pattern_vec.push((
            PIICategory::MedicalRecord,
            Regex::new(patterns::MEDICAL_RECORD).unwrap(),
        ));

        Self { patterns: pattern_vec }
    }

    pub fn detect(&self, text: &str, mask: bool) -> Vec<PIEFinding> {
        let mut findings = Vec::new();
        for (category, pattern) in &self.patterns {
            for m in pattern.find_iter(text) {
                let raw = m.as_str();
                let display_text = if mask { patterns::mask_pii(raw) } else { raw.to_string() };
                findings.push(PIEFinding {
                    category: category.clone(),
                    text: display_text,
                    start: m.start(),
                    end: m.end(),
                    confidence: 0.95,
                });
            }
        }
        findings
    }
}

impl Default for PIIDetector {
    fn default() -> Self {
        Self::new()
    }
}
