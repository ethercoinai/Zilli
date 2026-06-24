use super::config::TrainingConfig;

pub struct GrpoTrainer {
    config: TrainingConfig,
}

impl GrpoTrainer {
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }

    pub fn compute_advantages(&self, group_trajectories: &[Vec<f64>]) -> Vec<f64> {
        let mut advantages = Vec::new();

        for group in group_trajectories {
            let mean = group.iter().sum::<f64>() / group.len() as f64;
            let std = (group.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / group.len() as f64)
                .sqrt()
                .max(1e-8);

            for reward in group {
                let adv = (reward - mean) / std;
                advantages.push(adv);
            }
        }

        advantages
    }

    pub fn compute_loss(&self, _trajectories: &[f64], advantages: &[f64]) -> serde_json::Value {
        let policy_loss = advantages
            .iter()
            .map(|adv| {
                let ratio = 1.0 + 0.1 * adv;
                let clipped = ratio.clamp(1.0 - self.config.clip_range, 1.0 + self.config.clip_range);
                -((ratio * adv).min(clipped * adv))
            })
            .sum::<f64>()
            / advantages.len().max(1) as f64;

        let entropy = self
            .config
            .entropy_coef
            * advantages
                .iter()
                .filter_map(|a| {
                    let p = (1.0 + a.exp()).ln();
                    if p.is_finite() { Some(-p) } else { None }
                })
                .sum::<f64>()
            / advantages.len().max(1) as f64;

        serde_json::json!({
            "policy_loss": policy_loss,
            "entropy": entropy,
            "total_loss": policy_loss - entropy,
        })
    }
}
