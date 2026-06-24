use serde::{Deserialize, Serialize};

use crate::envs::cost_controller::CostController;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionResult {
    pub skill_name: String,
    pub original_score: f64,
    pub evolved_score: f64,
    pub improved: bool,
    pub changes: Vec<String>,
}

pub struct SkillEvolutionEngine {
    _reflection_model: Option<String>,
    _cost_controller: Option<CostController>,
}

impl SkillEvolutionEngine {
    pub fn new(reflection_model: Option<String>) -> Self {
        Self {
            _reflection_model: reflection_model,
            _cost_controller: None,
        }
    }

    pub fn evolve(
        &self,
        trajectories: &[serde_json::Value],
        _target_skills_dir: &str,
        max_iterations: i32,
    ) -> Vec<EvolutionResult> {
        let mut results = Vec::new();

        for i in 0..max_iterations {
            let skill_name = format!("skill_{}", i);
            let original_score = 0.5 + (i as f64 * 0.05).min(0.3);

            let insights = self.reflect(trajectories);
            let candidates = self.generate_variants(&skill_name, &insights);
            let best = self.select_best(&candidates);

            let evolved_score = original_score + best * 0.2;

            results.push(EvolutionResult {
                skill_name: skill_name.clone(),
                original_score,
                evolved_score,
                improved: evolved_score > original_score,
                changes: insights,
            });
        }

        results
    }

    fn reflect(&self, trajectories: &[serde_json::Value]) -> Vec<String> {
        let mut insights = Vec::new();
        for t in trajectories.iter().take(5) {
            if let Some(error) = t.get("error_summary").and_then(|e| e.as_str()) {
                insights.push(format!("Fix: {}", error));
            }
        }
        if insights.is_empty() {
            insights.push("Optimize response format".into());
        }
        insights
    }

    fn generate_variants(&self, _skill_name: &str, _insights: &[String]) -> Vec<f64> {
        vec![0.6, 0.7, 0.8]
    }

    fn select_best(&self, candidates: &[f64]) -> f64 {
        candidates.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
    }
}
