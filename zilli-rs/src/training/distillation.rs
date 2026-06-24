use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationSample {
    pub executor_action: String,
    pub planner_action: String,
    pub executor_log_prob: f64,
    pub planner_log_prob: f64,
    pub executor_reward: f64,
    pub planner_reward: f64,
    pub executor_embedding: Option<Vec<f64>>,
    pub planner_embedding: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistillationCycle {
    pub cycle_id: String,
    pub loss_bc: f64,
    pub loss_rl: f64,
    pub loss_reg: f64,
    pub kl_divergence: f64,
    pub lora_triggered: bool,
    pub timestamp: DateTime<Utc>,
}

pub struct DistillationScheduler {
    samples: RwLock<Vec<DistillationSample>>,
    cycle_count: RwLock<i32>,
    lambda_bc: f64,
    lambda_rl: f64,
    lambda_reg: f64,
    min_samples: i32,
    max_samples: i32,
    cycles: RwLock<Vec<DistillationCycle>>,
}

impl DistillationScheduler {
    pub fn new(
        lambda_bc: f64,
        lambda_rl: f64,
        lambda_reg: f64,
        min_samples: i32,
        max_samples: i32,
    ) -> Self {
        Self {
            samples: RwLock::new(Vec::new()),
            cycle_count: RwLock::new(0),
            lambda_bc,
            lambda_rl,
            lambda_reg,
            min_samples,
            max_samples,
            cycles: RwLock::new(Vec::new()),
        }
    }

    pub fn add_sample(&self, sample: DistillationSample) {
        let mut samples = self.samples.write();
        samples.push(sample);
        if samples.len() > self.max_samples as usize {
            samples.remove(0);
        }
    }

    pub fn add_batch(&self, samples: Vec<DistillationSample>) {
        let mut current = self.samples.write();
        for s in samples {
            current.push(s);
            if current.len() > self.max_samples as usize {
                current.remove(0);
            }
        }
    }

    pub fn should_distill(&self) -> bool {
        self.samples.read().len() >= self.min_samples as usize
    }

    pub fn run_cycle(&self) -> Option<DistillationCycle> {
        if !self.should_distill() {
            return None;
        }

        let samples = self.samples.read().clone();
        let count = *self.cycle_count.read();

        let loss_bc = samples
            .iter()
            .map(|s| {
                let diff = s.executor_log_prob - s.planner_log_prob;
                (diff * diff).min(10.0)
            })
            .sum::<f64>()
            / samples.len().max(1) as f64;

        let loss_rl = samples
            .iter()
            .map(|s| {
                let diff = s.executor_reward - s.planner_reward;
                (diff * diff).min(10.0)
            })
            .sum::<f64>()
            / samples.len().max(1) as f64;

        let loss_reg = samples
            .iter()
            .filter_map(|s| {
                Some(match (&s.executor_embedding, &s.planner_embedding) {
                    (Some(e), Some(p)) => {
                        let diff: f64 = e.iter().zip(p.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                        diff.min(5.0) * self.lambda_reg
                    }
                    _ => self.lambda_reg * 0.1,
                })
            })
            .sum::<f64>()
            / samples.len().max(1) as f64;

        let total_loss = self.lambda_bc * loss_bc + self.lambda_rl * loss_rl + loss_reg;
        let kl_divergence = loss_bc * 0.5 + loss_rl * 0.5;
        let lora_triggered = total_loss > 1.0;

        let cycle = DistillationCycle {
            cycle_id: format!("cycle_{}", count),
            loss_bc,
            loss_rl,
            loss_reg,
            kl_divergence,
            lora_triggered,
            timestamp: Utc::now(),
        };

        *self.cycle_count.write() += 1;
        self.cycles.write().push(cycle.clone());

        Some(cycle)
    }

    pub fn stats(&self) -> serde_json::Value {
        serde_json::json!({
            "total_samples": self.samples.read().len(),
            "cycle_count": *self.cycle_count.read(),
            "recent_cycles": self.cycles.read().iter().rev().take(5).cloned().collect::<Vec<_>>(),
            "lambda_bc": self.lambda_bc,
            "lambda_rl": self.lambda_rl,
            "lambda_reg": self.lambda_reg,
        })
    }
}
