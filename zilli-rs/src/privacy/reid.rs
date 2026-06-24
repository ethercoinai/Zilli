use serde::{Deserialize, Serialize};
use strum::{Display, EnumString};

#[derive(Debug, Clone, Serialize, Deserialize, Display, EnumString, PartialEq, PartialOrd)]
pub enum ReIDRisk {
    None,
    Low,
    Medium,
    High,
    Critical,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReIDAssessment {
    pub risk: ReIDRisk,
    pub quasi_identifiers: Vec<String>,
    pub combinations: Vec<Vec<String>>,
    pub score: f64,
}

const REID_THRESHOLD: f64 = 0.05;

pub struct ReIDAssessor;

impl ReIDAssessor {
    pub fn assess(&self, fields: &[(&str, usize)], k: usize) -> ReIDAssessment {
        let mut quasi_identifiers = Vec::new();
        let mut combinations = Vec::new();

        for (name, unique_count) in fields {
            if *unique_count > 1 {
                quasi_identifiers.push(name.to_string());
            }
        }

        for i in 0..quasi_identifiers.len() {
            for j in i + 1..quasi_identifiers.len() {
                combinations.push(vec![
                    quasi_identifiers[i].clone(),
                    quasi_identifiers[j].clone(),
                ]);
            }
        }

        let record_count = fields.iter().map(|(_, c)| c).max().unwrap_or(&1);
        let max_unique = fields
            .iter()
            .map(|(_, c)| c)
            .max()
            .unwrap_or(&1)
            .max(&1);

        let singleton_prob = if *record_count > 0 && *max_unique > 0 {
            1.0 / *max_unique as f64
        } else {
            1.0
        };

        let linkage_prob = if k > 0 {
            1.0 - (1.0 - 1.0 / k as f64).powi(quasi_identifiers.len() as i32)
        } else {
            0.0
        };

        let score = (singleton_prob + linkage_prob).min(1.0);

        let risk = if score < 0.01 {
            ReIDRisk::None
        } else if score < REID_THRESHOLD {
            ReIDRisk::Low
        } else if score < 0.15 {
            ReIDRisk::Medium
        } else if score < 0.3 {
            ReIDRisk::High
        } else {
            ReIDRisk::Critical
        };

        ReIDAssessment {
            risk,
            quasi_identifiers,
            combinations,
            score,
        }
    }

    pub fn is_safe(&self, assessment: &ReIDAssessment) -> bool {
        assessment.score < REID_THRESHOLD
    }
}
