use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use parking_lot::RwLock;

use super::classifier::DataClass;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CloudProvider {
    None,
    OpenAI,
    Anthropic,
    Google,
    Sakana,
    Custom(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SanitizationRule {
    pub mask_char: Option<char>,
    pub mask_length: Option<usize>,
    pub preserve_prefix: Option<usize>,
    pub hash_salt: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataGovernancePolicy {
    pub tenant_id: String,
    pub max_data_class: DataClass,
    pub allowed_cloud: Vec<CloudProvider>,
    pub sanitization: HashMap<String, SanitizationRule>,
    pub retention_days: i32,
    pub audit_required: bool,
}

pub struct PolicyStore {
    policies: RwLock<HashMap<String, DataGovernancePolicy>>,
}

impl PolicyStore {
    pub fn new() -> Self {
        Self {
            policies: RwLock::new(HashMap::new()),
        }
    }

    pub fn get_policy(&self, tenant_id: &str) -> Option<DataGovernancePolicy> {
        self.policies.read().get(tenant_id).cloned()
    }

    pub fn set_policy(&self, policy: DataGovernancePolicy) {
        self.policies
            .write()
            .insert(policy.tenant_id.clone(), policy);
    }

    pub fn delete_policy(&self, tenant_id: &str) -> bool {
        self.policies.write().remove(tenant_id).is_some()
    }

    pub fn list_tenants(&self) -> Vec<String> {
        self.policies.read().keys().cloned().collect()
    }

    pub fn is_action_allowed(
        &self,
        tenant_id: &str,
        data_class: DataClass,
        use_cloud: bool,
    ) -> Result<bool, String> {
        let policy = self
            .get_policy(tenant_id)
            .ok_or_else(|| format!("No policy for tenant: {}", tenant_id))?;

        if data_class > policy.max_data_class {
            return Ok(false);
        }

        if use_cloud && policy.allowed_cloud.is_empty() {
            return Ok(false);
        }

        Ok(true)
    }
}

impl Default for PolicyStore {
    fn default() -> Self {
        Self::new()
    }
}
