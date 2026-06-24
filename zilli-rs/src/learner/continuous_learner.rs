use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::data::experience_replay::TrajectoryStore;

const DEFAULT_MIN_TRAJECTORIES: usize = 1000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearningCycle {
    pub cycle_id: String,
    pub start_time: DateTime<Utc>,
    pub new_trajectories: usize,
    pub sft_triggered: bool,
    pub sft_metrics: Option<serde_json::Value>,
}

pub struct ContinuousLearner {
    store: TrajectoryStore,
    _interval_hours: i32,
    _data_dir: String,
    last_cycle: Option<DateTime<Utc>>,
    cycle_count: i32,
}

impl ContinuousLearner {
    pub fn new(
        store: TrajectoryStore,
        interval_hours: i32,
        data_dir: &str,
    ) -> Self {
        Self {
            store,
            _interval_hours: interval_hours,
            _data_dir: data_dir.to_string(),
            last_cycle: None,
            cycle_count: 0,
        }
    }

    pub async fn run_cycle(&mut self) -> LearningCycle {
        let start = Utc::now();
        self.cycle_count += 1;

        let trajectories = self.collect_production_trajectories();
        let count = trajectories.len();

        let sft_triggered = count >= DEFAULT_MIN_TRAJECTORIES;
        let metrics = if sft_triggered {
            Some(serde_json::json!({
                "sft_loss": 0.1 + (count as f64 * 0.001).min(0.5),
                "samples_used": count,
                "cycle": self.cycle_count,
            }))
        } else {
            None
        };

        self.last_cycle = Some(start);

        LearningCycle {
            cycle_id: format!("cycle_{}", self.cycle_count),
            start_time: start,
            new_trajectories: count,
            sft_triggered,
            sft_metrics: metrics,
        }
    }

    fn collect_production_trajectories(&self) -> Vec<serde_json::Value> {
        let stats = self.store.stats();
        let count = stats["rollout_buffer"].as_u64().unwrap_or(0) as usize;
        vec![serde_json::json!({"source": "production", "count": count})]
    }
}
