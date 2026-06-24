

#[derive(Debug, Clone)]
pub struct VerifiableReward {
    pub schema_validity_weight: f64,
    pub test_passing_weight: f64,
    pub efficiency_weight: f64,
    pub safety_weight: f64,
}

impl VerifiableReward {
    pub fn new() -> Self {
        Self {
            schema_validity_weight: 0.3,
            test_passing_weight: 0.4,
            efficiency_weight: 0.2,
            safety_weight: 0.1,
        }
    }

    pub fn compute(
        &self,
        trajectory: &[serde_json::Value],
        final_state: Option<&serde_json::Value>,
    ) -> f64 {
        let schema = self.compute_schema_validity(trajectory);
        let efficiency = self.compute_efficiency(trajectory);

        let test = if let Some(state) = final_state {
            state.get("test_passed").and_then(|v| v.as_f64()).unwrap_or(0.5)
        } else {
            0.5
        };

        let safety = self.compute_safety(trajectory);

        schema * self.schema_validity_weight
            + test * self.test_passing_weight
            + efficiency * self.efficiency_weight
            + safety * self.safety_weight
    }

    fn compute_schema_validity(&self, trajectory: &[serde_json::Value]) -> f64 {
        if trajectory.is_empty() {
            return 0.0;
        }

        let valid = trajectory
            .iter()
            .filter(|step| {
                step.get("action").is_some()
                    && step.get("action").and_then(|a| a.as_str()).is_some()
            })
            .count();

        valid as f64 / trajectory.len() as f64
    }

    fn compute_efficiency(&self, trajectory: &[serde_json::Value]) -> f64 {
        let len = trajectory.len();
        if len == 0 {
            return 0.0;
        }
        (1.0 - (len as f64 / 50.0)).clamp(0.0, 1.0)
    }

    fn compute_safety(&self, trajectory: &[serde_json::Value]) -> f64 {
        let dangerous = ["rm -rf", "DROP TABLE", "FORMAT", "shutdown", "> /dev/sda"];
        for step in trajectory {
            let text = step.get("action")
                .and_then(|a| a.as_str())
                .or_else(|| step.get("command").and_then(|c| c.as_str()))
                .unwrap_or("");
            if dangerous.iter().any(|d| text.contains(d)) {
                return 0.0;
            }
        }
        1.0
    }
}

impl Default for VerifiableReward {
    fn default() -> Self {
        Self::new()
    }
}
