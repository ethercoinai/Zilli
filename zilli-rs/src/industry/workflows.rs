use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use strum::{Display, EnumString};

use crate::privacy::classifier::DataClass;
use crate::privacy::policy::CloudProvider;
use crate::models::registry::ModelRegistry;

#[derive(Debug, Clone, Serialize, Deserialize, Display, EnumString, PartialEq, Hash, Eq)]
pub enum IndustryType {
    Legal,
    Medical,
    Financial,
    Education,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndustryWorkflow {
    pub industry: IndustryType,
    pub compliance_rules: Vec<String>,
    pub access_level: DataClass,
    pub retention_days: i32,
    pub permitted_cloud: Vec<CloudProvider>,
}

pub struct WorkflowRegistry {
    workflows: HashMap<IndustryType, IndustryWorkflow>,
    _model_registry: Option<ModelRegistry>,
}

impl WorkflowRegistry {
    pub fn new() -> Self {
        let mut workflows = HashMap::new();

        workflows.insert(
            IndustryType::Legal,
            IndustryWorkflow {
                industry: IndustryType::Legal,
                compliance_rules: vec![
                    "attorney-client_privacy".into(),
                    "document_retention_7yr".into(),
                ],
                access_level: DataClass::Confidential,
                retention_days: 2555,
                permitted_cloud: vec![],
            },
        );

        workflows.insert(
            IndustryType::Medical,
            IndustryWorkflow {
                industry: IndustryType::Medical,
                compliance_rules: vec![
                    "hipaa_compliant".into(),
                    "phi_protection".into(),
                ],
                access_level: DataClass::Regulated,
                retention_days: 2555,
                permitted_cloud: vec![],
            },
        );

        workflows.insert(
            IndustryType::Financial,
            IndustryWorkflow {
                industry: IndustryType::Financial,
                compliance_rules: vec![
                    "sox_compliant".into(),
                    "pci_dss".into(),
                    "audit_trail_required".into(),
                ],
                access_level: DataClass::Restricted,
                retention_days: 1825,
                permitted_cloud: vec![],
            },
        );

        workflows.insert(
            IndustryType::Education,
            IndustryWorkflow {
                industry: IndustryType::Education,
                compliance_rules: vec![
                    "ferpa_compliant".into(),
                    "student_data_privacy".into(),
                ],
                access_level: DataClass::Confidential,
                retention_days: 365,
                permitted_cloud: vec![],
            },
        );

        Self {
            workflows,
            _model_registry: None,
        }
    }

    pub fn list_industries(&self) -> Vec<serde_json::Value> {
        self.workflows
            .iter()
            .map(|(_, w)| {
                serde_json::json!({
                    "industry": w.industry.to_string(),
                    "compliance_rules": w.compliance_rules,
                    "access_level": w.access_level.to_string(),
                    "retention_days": w.retention_days,
                })
            })
            .collect()
    }
}

impl Default for WorkflowRegistry {
    fn default() -> Self {
        Self::new()
    }
}
