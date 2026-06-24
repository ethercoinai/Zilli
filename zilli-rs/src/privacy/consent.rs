use std::sync::Arc;
use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};

#[derive(Debug, Clone, Serialize, Deserialize, Display, EnumString, PartialEq)]
pub enum ConsentStatus {
    Granted,
    Denied,
    Expired,
    Revoked,
    Pending,
}

#[derive(Debug, Clone, Serialize, Deserialize, Display, EnumString, PartialEq, Hash, Eq)]
pub enum DataUse {
    LocalInference,
    CloudInference,
    Training,
    Distillation,
    Audit,
    AnonymizedAnalytics,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsentRecord {
    pub user_id: String,
    pub data_use: DataUse,
    pub status: ConsentStatus,
    pub granted_at: DateTime<Utc>,
    pub expires_at: Option<DateTime<Utc>>,
    pub purpose: Option<String>,
    pub revoked_at: Option<DateTime<Utc>>,
}

pub struct ConsentManager {
    records: Arc<RwLock<Vec<ConsentRecord>>>,
}

impl ConsentManager {
    pub fn new() -> Self {
        Self {
            records: Arc::new(RwLock::new(Vec::new())),
        }
    }
}

impl Clone for ConsentManager {
    fn clone(&self) -> Self {
        Self {
            records: Arc::clone(&self.records),
        }
    }
}

impl ConsentManager {

    pub fn grant(
        &self,
        user_id: &str,
        data_use: DataUse,
        purpose: Option<&str>,
        ttl_days: Option<i64>,
    ) -> ConsentRecord {
        let record = ConsentRecord {
            user_id: user_id.to_string(),
            data_use,
            status: ConsentStatus::Granted,
            granted_at: Utc::now(),
            expires_at: ttl_days.map(|d| Utc::now() + chrono::Duration::days(d)),
            purpose: purpose.map(|s| s.to_string()),
            revoked_at: None,
        };
        self.records.write().push(record.clone());
        record
    }

    pub fn revoke(&self, user_id: &str, data_use: &DataUse) -> bool {
        let mut records = self.records.write();
        for record in records.iter_mut().rev() {
            if record.user_id == user_id && record.data_use == *data_use {
                record.status = ConsentStatus::Revoked;
                record.revoked_at = Some(Utc::now());
                return true;
            }
        }
        false
    }

    pub fn check(&self, user_id: &str, data_use: &DataUse) -> ConsentStatus {
        let records = self.records.read();
        for record in records.iter().rev() {
            if record.user_id == user_id && record.data_use == *data_use {
                match &record.status {
                    ConsentStatus::Granted => {
                        if let Some(expires) = record.expires_at {
                            if Utc::now() > expires {
                                return ConsentStatus::Expired;
                            }
                        }
                        return ConsentStatus::Granted;
                    }
                    other => return other.clone(),
                }
            }
        }
        ConsentStatus::Pending
    }

    pub fn get_user_records(&self, user_id: &str) -> Vec<ConsentRecord> {
        self.records
            .read()
            .iter()
            .filter(|r| r.user_id == user_id)
            .cloned()
            .collect()
    }

    pub fn get_active_consents(&self, data_use: &DataUse) -> Vec<ConsentRecord> {
        let now = Utc::now();
        self.records
            .read()
            .iter()
            .filter(|r| {
                r.data_use == *data_use
                    && r.status == ConsentStatus::Granted
                    && r.expires_at.map_or(true, |e| e > now)
            })
            .cloned()
            .collect()
    }
}

impl Default for ConsentManager {
    fn default() -> Self {
        Self::new()
    }
}
