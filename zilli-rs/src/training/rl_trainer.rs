use super::cispo::CispoTrainer;
use super::config::TrainingConfig;
use super::grpo::GrpoTrainer;

pub enum SelectedTrainer {
    CISPO(CispoTrainer),
    GRPO(GrpoTrainer),
}

pub struct RLTrainer {
    trainer: SelectedTrainer,
}

impl RLTrainer {
    pub fn new(config: Option<TrainingConfig>) -> Self {
        let config = config.unwrap_or_default();
        let trainer = match config.algorithm.as_str() {
            "grpo" => SelectedTrainer::GRPO(GrpoTrainer::new(config)),
            _ => SelectedTrainer::CISPO(CispoTrainer::new(config)),
        };
        Self { trainer }
    }

    pub fn update(&self, batch: &[f64]) -> serde_json::Value {
        let dones = vec![false; batch.len()];
        let group_batch = vec![batch.to_vec()];

        let advantages = match &self.trainer {
            SelectedTrainer::CISPO(t) => t.compute_advantages(batch, &dones),
            SelectedTrainer::GRPO(t) => t.compute_advantages(&group_batch),
        };

        match &self.trainer {
            SelectedTrainer::CISPO(t) => t.compute_loss(&advantages),
            SelectedTrainer::GRPO(t) => t.compute_loss(batch, &advantages),
        }
    }
}
