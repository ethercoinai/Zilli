use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryEntry {
    pub trajectory: Vec<serde_json::Value>,
    pub final_reward: f64,
    pub task_type: Option<String>,
    pub timestamp: i64,
}

pub struct TrajectoryStore {
    golden: RwLock<VecDeque<TrajectoryEntry>>,
    failures: RwLock<VecDeque<TrajectoryEntry>>,
    max_golden: usize,
    max_failure: usize,
    rollout_buffer: RwLock<VecDeque<TrajectoryEntry>>,
}

impl TrajectoryStore {
    pub fn new() -> Self {
        Self {
            golden: RwLock::new(VecDeque::with_capacity(5000)),
            failures: RwLock::new(VecDeque::with_capacity(2000)),
            max_golden: 5000,
            max_failure: 2000,
            rollout_buffer: RwLock::new(VecDeque::with_capacity(1000)),
        }
    }

    pub fn add_trajectory(&self, trajectory: Vec<serde_json::Value>, final_reward: f64) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let entry = TrajectoryEntry {
            trajectory,
            final_reward,
            task_type: None,
            timestamp: now,
        };

        if final_reward >= 0.8 {
            let mut golden = self.golden.write();
            golden.push_back(entry.clone());
            if golden.len() > self.max_golden {
                golden.pop_front();
            }
        } else if final_reward <= 0.3 {
            let mut failures = self.failures.write();
            failures.push_back(entry.clone());
            if failures.len() > self.max_failure {
                failures.pop_front();
            }
        }

        let mut buffer = self.rollout_buffer.write();
        buffer.push_back(entry);
        if buffer.len() > 1000 {
            buffer.pop_front();
        }
    }

    pub fn sample_batch(&self, batch_size: usize, golden_ratio: f64) -> Vec<TrajectoryEntry> {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let mut batch = Vec::new();

        let golden_count = (batch_size as f64 * golden_ratio) as usize;
        let failure_count = batch_size - golden_count;

        {
            let golden = self.golden.read();
            for _ in 0..golden_count.min(golden.len()) {
                let idx = rng.gen_range(0..golden.len());
                if let Some(entry) = golden.get(idx) {
                    batch.push(entry.clone());
                }
            }
        }

        {
            let failures = self.failures.read();
            for _ in 0..failure_count.min(failures.len()) {
                let idx = rng.gen_range(0..failures.len());
                if let Some(entry) = failures.get(idx) {
                    batch.push(entry.clone());
                }
            }
        }

        batch
    }

    pub fn purify(&self) -> usize {
        let mut golden = self.golden.write();
        let golden_before = golden.len();
        while golden.len() > self.max_golden {
            golden.pop_front();
        }
        let golden_purged = golden_before - golden.len();

        let mut failures = self.failures.write();
        let failures_before = failures.len();
        while failures.len() > self.max_failure {
            failures.pop_front();
        }
        let failures_purged = failures_before - failures.len();

        golden_purged + failures_purged
    }

    pub fn stats(&self) -> serde_json::Value {
        serde_json::json!({
            "golden_count": self.golden.read().len(),
            "failure_count": self.failures.read().len(),
            "rollout_buffer": self.rollout_buffer.read().len(),
        })
    }
}

impl Default for TrajectoryStore {
    fn default() -> Self {
        Self::new()
    }
}
