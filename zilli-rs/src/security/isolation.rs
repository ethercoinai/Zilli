use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use strum::{Display, EnumString};

#[derive(Debug, Clone, Serialize, Deserialize, Display, EnumString, PartialEq, PartialOrd)]
pub enum AccessLevel {
    Public,
    Internal,
    Confidential,
    Restricted,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationPolicy {
    pub tenant_id: String,
    pub access_level: AccessLevel,
    pub allowed_roles: Vec<String>,
    pub max_input_length: i32,
    pub require_sanitization: bool,
    pub audit_required: bool,
    pub retention_days: i32,
}

pub struct DataIsolation {
    policies: RwLock<HashMap<String, IsolationPolicy>>,
}

impl DataIsolation {
    pub fn new() -> Self {
        Self {
            policies: RwLock::new(HashMap::new()),
        }
    }

    pub fn get_policy(&self, tenant_id: &str) -> Option<IsolationPolicy> {
        self.policies.read().get(tenant_id).cloned()
    }

    pub fn set_policy(&self, policy: IsolationPolicy) {
        self.policies
            .write()
            .insert(policy.tenant_id.clone(), policy);
    }

    pub fn check_access(&self, tenant_id: &str, role: &str) -> bool {
        self.policies
            .read()
            .get(tenant_id)
            .map_or(false, |p| p.allowed_roles.contains(&role.to_string()))
    }

    pub fn check_input_length(&self, tenant_id: &str, input_len: usize) -> bool {
        self.policies.read().get(tenant_id).map_or(false, |p| {
            input_len <= p.max_input_length as usize
        })
    }

    pub fn needs_sanitization(&self, tenant_id: &str) -> bool {
        self.policies
            .read()
            .get(tenant_id)
            .map_or(false, |p| p.require_sanitization)
    }
}

impl Default for DataIsolation {
    fn default() -> Self {
        Self::new()
    }
}
