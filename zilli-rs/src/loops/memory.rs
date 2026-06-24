use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

use super::runner::LoopCycle;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    pub cycle_id: String,
    pub timestamp: DateTime<Utc>,
    pub input_data: String,
    pub output: Option<String>,
    pub passed: bool,
    pub evidence: Option<String>,
    pub duration_ms: u64,
    pub metadata: Option<serde_json::Value>,
}

pub struct CycleMemory {
    entries: RwLock<VecDeque<MemoryEntry>>,
    max_entries: usize,
    _persist_path: Option<String>,
}

impl CycleMemory {
    pub fn new(max_entries: usize) -> Self {
        Self {
            entries: RwLock::new(VecDeque::with_capacity(max_entries)),
            max_entries,
            _persist_path: None,
        }
    }

    pub fn store(&self, cycle: LoopCycle) {
        let entry = MemoryEntry {
            cycle_id: cycle.id,
            timestamp: Utc::now(),
            input_data: cycle.input_data,
            output: cycle.output,
            passed: cycle.verification.as_ref().map_or(false, |v| v.passed),
            evidence: cycle.verification.map(|v| v.evidence),
            duration_ms: cycle.duration_ms,
            metadata: cycle.metadata,
        };

        let mut entries = self.entries.write();
        entries.push_back(entry);
        if entries.len() > self.max_entries {
            entries.pop_front();
        }
    }

    pub fn recent(&self, n: usize) -> Vec<MemoryEntry> {
        let entries = self.entries.read();
        entries.iter().rev().take(n).cloned().collect()
    }

    pub fn stats(&self) -> serde_json::Value {
        let entries = self.entries.read();
        let passed = entries.iter().filter(|e| e.passed).count();
        serde_json::json!({
            "total": entries.len(),
            "passed": passed,
            "failed": entries.len() - passed,
            "max_entries": self.max_entries,
        })
    }
}
