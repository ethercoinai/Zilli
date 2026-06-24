use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;
use std::fs::{self, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::PathBuf;
use strum::{Display, EnumString};

#[derive(Debug, Clone, Serialize, Deserialize, Display, EnumString, PartialEq)]
pub enum AuditLevel {
    Debug,
    Info,
    Warning,
    Error,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEvent {
    pub event_type: String,
    pub level: AuditLevel,
    pub message: String,
    pub timestamp: DateTime<Utc>,
    pub tenant_id: String,
    pub user_id: Option<String>,
    pub data_class: Option<String>,
    pub sanitized: bool,
}

pub struct AuditLogger {
    log_dir: PathBuf,
    events: RwLock<VecDeque<AuditEvent>>,
    _sanitize: bool,
}

impl AuditLogger {
    pub fn new(log_dir: &str, sanitize: bool) -> std::io::Result<Self> {
        let path = PathBuf::from(log_dir);
        fs::create_dir_all(&path)?;

        Ok(Self {
            log_dir: path,
            events: RwLock::new(VecDeque::with_capacity(10000)),
            _sanitize: sanitize,
        })
    }

    pub fn log(&self, event: AuditEvent) -> std::io::Result<()> {
        let event_json = serde_json::to_string(&event)?;

        {
            let mut events = self.events.write();
            events.push_back(event.clone());
            if events.len() > 10000 {
                events.pop_front();
            }
        }

        let date = event.timestamp.format("%Y-%m-%d");
        let filename = self.log_dir.join(format!("audit-{}.jsonl", date));

        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&filename)?;

        let mut writer = BufWriter::new(file);
        writeln!(writer, "{}", event_json)?;

        Ok(())
    }

    pub fn query(
        &self,
        tenant_id: &str,
        start: Option<DateTime<Utc>>,
        end: Option<DateTime<Utc>>,
    ) -> Vec<AuditEvent> {
        let events = self.events.read();
        events
            .iter()
            .filter(|e| {
                if e.tenant_id != tenant_id {
                    return false;
                }
                if let Some(start) = start {
                    if e.timestamp < start {
                        return false;
                    }
                }
                if let Some(end) = end {
                    if e.timestamp > end {
                        return false;
                    }
                }
                true
            })
            .cloned()
            .collect()
    }

    pub fn recent(&self, n: usize) -> Vec<AuditEvent> {
        let events = self.events.read();
        events.iter().rev().take(n).cloned().collect()
    }
}
