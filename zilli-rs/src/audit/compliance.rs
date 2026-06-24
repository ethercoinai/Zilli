use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use strum::{Display, EnumString};

use super::logger::AuditEvent;

#[derive(Debug, Clone, Serialize, Deserialize, Display, EnumString, PartialEq, Eq, Hash)]
pub enum ComplianceFramework {
    GDPR,
    HIPAA,
    Soc2,
    PciDss,
    Ferpa,
    Ccpa,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceReport {
    pub framework: ComplianceFramework,
    pub tenant_id: String,
    pub generated_at: DateTime<Utc>,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub total_requests: u64,
    pub cloud_requests: u64,
    pub violations: Vec<ComplianceViolation>,
    pub remediation: Vec<String>,
    pub passed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplianceViolation {
    pub rule: String,
    pub severity: String,
    pub description: String,
    pub event_id: Option<String>,
}

pub struct ComplianceReporter {
    framework_rules: HashMap<ComplianceFramework, Vec<&'static str>>,
}

impl ComplianceReporter {
    pub fn new() -> Self {
        let mut rules = HashMap::new();

        rules.insert(
            ComplianceFramework::GDPR,
            vec![
                "Right to erasure audit trail required",
                "Consent records must be retained",
                "Data processing activities must be logged",
                "Cross-border transfer documentation required",
                "Data breach notification within 72 hours",
            ],
        );

        rules.insert(
            ComplianceFramework::HIPAA,
            vec![
                "PHI access must be logged and monitored",
                "Minimum necessary use verification required",
                "Breach notification readiness assessment required",
                "Encryption at rest and in transit required",
                "Business associate agreements required",
            ],
        );

        rules.insert(
            ComplianceFramework::Soc2,
            vec![
                "Security monitoring evidence must be retained",
                "Access control logs required",
                "Change management records required",
                "Risk assessment summaries required",
                "Vendor management documentation required",
            ],
        );

        Self { framework_rules: rules }
    }

    pub fn generate_report(
        &self,
        framework: &ComplianceFramework,
        tenant_id: &str,
        events: &[AuditEvent],
        period_start: DateTime<Utc>,
        period_end: DateTime<Utc>,
    ) -> ComplianceReport {
        let total_requests = events.len() as u64;
        let cloud_requests = events
            .iter()
            .filter(|e| e.event_type.contains("cloud"))
            .count() as u64;

        let violations: Vec<ComplianceViolation> = self
            .framework_rules
            .get(framework)
            .map(|rules| self.check_violations(rules, events))
            .unwrap_or_default();

        let passed = violations.is_empty();

        let remediation = if passed {
            vec!["All checks passed: no remediation required".into()]
        } else {
            violations
                .iter()
                .map(|v| format!("Address {}: {} (event: {})", v.severity, v.rule, v.event_id.as_deref().unwrap_or("unknown")))
                .collect()
        };

        ComplianceReport {
            framework: framework.clone(),
            tenant_id: tenant_id.to_string(),
            generated_at: Utc::now(),
            period_start,
            period_end,
            total_requests,
            cloud_requests,
            violations,
            remediation,
            passed,
        }
    }

    fn check_violations(&self, rules: &[&str], events: &[AuditEvent]) -> Vec<ComplianceViolation> {
        let mut violations = Vec::new();

        for rule in rules {
            let severity = self.assess_risk(rule);
            if severity == "passed" {
                continue;
            }

            let matching_events: Vec<&AuditEvent> = events
                .iter()
                .filter(|e| {
                    let combined = format!("{} {} {:?} {:?}", e.event_type, e.message, e.data_class, e.level);
                    let combined_lower = combined.to_lowercase();
                    let keywords = self.rule_keywords(rule);
                    keywords.split('|').any(|kw| combined_lower.contains(kw.trim()))
                })
                .collect();

            if matching_events.is_empty() {
                violations.push(ComplianceViolation {
                    rule: rule.to_string(),
                    severity,
                    description: format!("No audit events match rule: {} — possible coverage gap", rule),
                    event_id: None,
                });
            } else {
                for event in matching_events {
                    let level = format!("{:?}", event.level);
                    violations.push(ComplianceViolation {
                        rule: rule.to_string(),
                        severity: if level == "Critical" || level == "Error" { "high".into() } else { severity.clone() },
                        description: format!("Event '{}' triggered rule: {}", event.message, rule),
                        event_id: Some(event.event_type.clone()),
                    });
                }
            }
        }

        violations
    }

    fn rule_keywords(&self, rule: &str) -> String {
        let r = rule.to_lowercase();
        if r.contains("erasure") { "delet|remove|erasure" }
        else if r.contains("consent") { "consent|opt" }
        else if r.contains("breach") { "breach|leak|expos" }
        else if r.contains("encryption") { "encrypt|tls|https" }
        else if r.contains("phi") || r.contains("ph") { "phi|health|patient" }
        else if r.contains("access") { "access|auth|login" }
        else if r.contains("monitor") || r.contains("log") { "audit|log|monitor" }
        else if r.contains("retained") || r.contains("retention") { "retain|retention|storage" }
        else { &r[..r.len().min(20)] }
        .to_string()
    }

    fn assess_risk(&self, rule: &str) -> String {
        let high_risk = ["breach", "erasure", "encryption", "phi"];
        let medium_risk = ["access", "monitoring", "retained", "logs"];

        if high_risk.iter().any(|k| rule.to_lowercase().contains(k)) {
            "high"
        } else if medium_risk.iter().any(|k| rule.to_lowercase().contains(k)) {
            "medium"
        } else {
            "passed"
        }
        .to_string()
    }
}

impl Default for ComplianceReporter {
    fn default() -> Self {
        Self::new()
    }
}
