use super::distillation::DistillationSample;

pub fn make_dummy_golden(count: i32, base_reward: f64) -> Vec<serde_json::Value> {
    let mut samples = Vec::new();
    for i in 0..count {
        samples.push(serde_json::json!({
            "trajectory": [
                {"action": "think", "content": format!("Step {} planning", i)},
                {"action": "bash_run", "command": format!("echo step_{}", i)},
                {"action": "finish", "summary": format!("Task {} complete", i)},
            ],
            "final_reward": base_reward + (i as f64 * 0.01).min(0.1),
            "task_type": "golden",
        }));
    }
    samples
}

pub fn make_dummy_failure(count: i32) -> Vec<serde_json::Value> {
    let mut samples = Vec::new();
    for i in 0..count {
        samples.push(serde_json::json!({
            "trajectory": [
                {"action": "think", "content": format!("Step {} plan", i)},
                {"action": "bash_run", "command": "invalid_command"},
                {"action": "bash_run", "command": "retry"},
            ],
            "final_reward": 0.15 + (i as f64 * 0.02).min(0.15),
            "error_summary": format!("Bash command failed in iteration {}", i),
        }));
    }
    samples
}

pub fn make_dummy_distillation_samples(count: i32) -> Vec<DistillationSample> {
    let mut samples = Vec::new();
    for i in 0..count {
        samples.push(DistillationSample {
            executor_action: format!("exec_action_{}", i),
            planner_action: format!("plan_action_{}", i),
            executor_log_prob: -0.5 - (i as f64 * 0.1),
            planner_log_prob: -0.3 - (i as f64 * 0.05),
            executor_reward: 0.7 + (i as f64 * 0.02).min(0.2),
            planner_reward: 0.9,
            executor_embedding: None,
            planner_embedding: None,
        });
    }
    samples
}
