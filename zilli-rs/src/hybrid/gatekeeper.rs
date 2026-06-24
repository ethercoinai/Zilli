use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};

use crate::privacy::classifier::DataClass;
use crate::privacy::consent::{ConsentManager, DataUse};
use crate::privacy::engine::PrivacyEngine;
use crate::privacy::policy::PolicyStore;

#[derive(Debug, Clone, Serialize, Deserialize, Display, EnumString, PartialEq)]
pub enum ExecutionTarget {
    Local,
    Cloud,
    Rejected,
    LocalWithCloudFallback,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GatekeeperDecision {
    pub target: ExecutionTarget,
    pub verdict: String,
    pub cloud_provider: Option<String>,
    pub reason: String,
    pub warnings: Vec<String>,
}

pub struct PrivacyGatekeeper {
    _privacy_engine: PrivacyEngine,
    policy_store: PolicyStore,
    consent_manager: ConsentManager,
}

impl PrivacyGatekeeper {
    pub fn new(
        privacy_engine: PrivacyEngine,
        policy_store: PolicyStore,
        consent_manager: ConsentManager,
    ) -> Self {
        Self {
            _privacy_engine: privacy_engine,
            policy_store,
            consent_manager,
        }
    }

    pub fn decide(
        &self,
        _text: &str,
        data_class: DataClass,
        data_use: &DataUse,
        tenant_id: &str,
    ) -> GatekeeperDecision {
        let mut warnings = Vec::new();

        let policy_allowed = self
            .policy_store
            .is_action_allowed(tenant_id, data_class, false);
        match policy_allowed {
            Ok(false) => {
                return GatekeeperDecision {
                    target: ExecutionTarget::Rejected,
                    verdict: "deny".into(),
                    cloud_provider: None,
                    reason: format!("Policy denies data class {:?} for tenant {}", data_class, tenant_id),
                    warnings,
                };
            }
            Err(e) => {
                warnings.push(format!("Policy check error: {}", e));
            }
            _ => {}
        }

        match data_class {
            DataClass::Restricted | DataClass::Regulated => GatekeeperDecision {
                target: ExecutionTarget::Local,
                verdict: "local_only".into(),
                cloud_provider: None,
                reason: format!(
                    "Data class {:?} requires local execution only",
                    data_class
                ),
                warnings,
            },
            DataClass::Confidential => {
                let consent = self.consent_manager.check(tenant_id, data_use);
                if consent != crate::privacy::consent::ConsentStatus::Granted {
                    return GatekeeperDecision {
                        target: ExecutionTarget::Local,
                        verdict: "no_consent_for_cloud".into(),
                        cloud_provider: None,
                        reason: "Confidential data: no consent for cloud fallback".into(),
                        warnings,
                    };
                }
                warnings.push("Confidential data: cloud execution requires sanitization".into());
                GatekeeperDecision {
                    target: ExecutionTarget::LocalWithCloudFallback,
                    verdict: "local_preferred_with_cloud_fallback".into(),
                    cloud_provider: None,
                    reason: "Confidential data: prefer local, cloud after sanitization".into(),
                    warnings,
                }
            }
            DataClass::Internal | DataClass::Public => {
                let consent = self.consent_manager.check(tenant_id, data_use);
                if consent == crate::privacy::consent::ConsentStatus::Granted {
                    GatekeeperDecision {
                        target: ExecutionTarget::Cloud,
                        verdict: "cloud_allowed".into(),
                        cloud_provider: Some("openai".into()),
                        reason: "Internal/Public data with consent: cloud execution allowed".into(),
                        warnings,
                    }
                } else {
                    GatekeeperDecision {
                        target: ExecutionTarget::Local,
                        verdict: "no_consent".into(),
                        cloud_provider: None,
                        reason: "No consent for cloud execution".into(),
                        warnings,
                    }
                }
            }
        }
    }
}
