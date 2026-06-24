use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanResult {
    pub cleaned: Vec<serde_json::Value>,
    pub warnings: Vec<String>,
}

pub struct TrajectoryCleaner {
    max_trajectory_length: usize,
    min_action_length: usize,
}

impl TrajectoryCleaner {
    pub fn new() -> Self {
        Self {
            max_trajectory_length: 100,
            min_action_length: 1,
        }
    }

    pub fn clean(&self, trajectory: &[serde_json::Value]) -> CleanResult {
        let mut cleaned = Vec::new();
        let mut warnings = Vec::new();

        for (i, step) in trajectory.iter().enumerate() {
            if i >= self.max_trajectory_length {
                warnings.push(format!(
                    "Truncated trajectory at step {} (max {})",
                    i, self.max_trajectory_length
                ));
                break;
            }
            cleaned.push(step.clone());
        }

        if cleaned.len() < self.min_action_length {
            warnings.push("Trajectory too short".into());
        }

        CleanResult { cleaned, warnings }
    }

    pub fn validate(&self, trajectory: &[serde_json::Value]) -> serde_json::Value {
        let mut issues = Vec::new();

        if trajectory.is_empty() {
            issues.push("Empty trajectory".to_string());
        }

        for (i, step) in trajectory.iter().enumerate() {
            if !step.is_object() {
                issues.push(format!("Step {} is not an object", i));
            }
            if step.get("action").is_none() && step.get("tool_name").is_none() {
                issues.push(format!("Step {} has no action/tool_name field", i));
            }
        }

        serde_json::json!({
            "valid": issues.is_empty(),
            "length": trajectory.len(),
            "issues": issues,
        })
    }
}

impl Default for TrajectoryCleaner {
    fn default() -> Self {
        Self::new()
    }
}
