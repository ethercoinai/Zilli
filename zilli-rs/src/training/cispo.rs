use super::config::TrainingConfig;

pub struct CispoTrainer {
    config: TrainingConfig,
}

impl CispoTrainer {
    pub fn new(config: TrainingConfig) -> Self {
        Self { config }
    }

    pub fn compute_advantages(&self, rewards: &[f64], dones: &[bool]) -> Vec<f64> {
        let mut returns = Vec::with_capacity(rewards.len());
        let mut discounted_return = 0.0;

        for i in (0..rewards.len()).rev() {
            discounted_return = rewards[i]
                + self.config.gamma * discounted_return * (1.0 - dones[i] as i32 as f64);
            returns.push(discounted_return);
        }
        returns.reverse();

        let mut advantages = Vec::with_capacity(rewards.len());
        let mut gae = 0.0;

        for i in (0..rewards.len()).rev() {
            let next_return = if i + 1 < rewards.len() { returns[i + 1] } else { 0.0 };
            let delta = rewards[i]
                + self.config.gamma * next_return * (1.0 - dones[i] as i32 as f64)
                - returns[i];
            gae = delta
                + self.config.gamma * self.config.gae_lambda * gae * (1.0 - dones[i] as i32 as f64);
            advantages.push(gae);
        }

        advantages.reverse();
        advantages
    }

    pub fn compute_loss(&self, advantages: &[f64]) -> serde_json::Value {
        let policy_loss = advantages
            .iter()
            .map(|adv| {
                if *adv >= 0.0 {
                    let ratio = 1.0 + 0.1 * adv;
                    ratio.min(1.0 + self.config.clip_range) * adv
                } else {
                    let ratio = 1.0 + 0.1 * adv;
                    ratio.max(1.0 - self.config.clip_range) * adv
                }
            })
            .sum::<f64>()
            / advantages.len().max(1) as f64;

        let entropy = self
            .config
            .entropy_coef
            * advantages
                .iter()
                .map(|a| a.abs() * f64::ln(1.0 + a.abs()))
                .sum::<f64>()
            / advantages.len().max(1) as f64;

        let total_loss = -policy_loss - entropy;

        serde_json::json!({
            "policy_loss": -policy_loss,
            "value_loss": 0.0,
            "entropy": entropy,
            "total_loss": total_loss,
        })
    }
}
