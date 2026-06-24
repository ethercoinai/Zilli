use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use strum::{Display, EnumString};

#[derive(Debug, Clone, Serialize, Deserialize, Display, EnumString, PartialEq)]
pub enum ArenaStatus {
    Champion,
    Challenger,
    Contender,
    Retired,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArenaModel {
    pub name: String,
    pub version: String,
    pub status: ArenaStatus,
    pub deployed_at: DateTime<Utc>,
    pub metrics: HashMap<String, f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArenaMatch {
    pub champion: String,
    pub challenger: String,
    pub champion_score: f64,
    pub challenger_score: f64,
    pub winner: Option<String>,
    pub significant: bool,
}

pub struct ChampionChallenger {
    models: RwLock<HashMap<String, ArenaModel>>,
    matches: RwLock<Vec<ArenaMatch>>,
    min_win_gap: f64,
}

impl ChampionChallenger {
    pub fn new(min_win_gap: f64) -> Self {
        Self {
            models: RwLock::new(HashMap::new()),
            matches: RwLock::new(Vec::new()),
            min_win_gap,
        }
    }

    pub fn register_model(&self, name: &str, version: &str, status: ArenaStatus) {
        let model = ArenaModel {
            name: name.to_string(),
            version: version.to_string(),
            status,
            deployed_at: Utc::now(),
            metrics: HashMap::new(),
        };
        self.models.write().insert(name.to_string(), model);
    }

    pub fn run_match(
        &self,
        model_name: &str,
        champion_score: f64,
        challenger_score: f64,
    ) -> Option<ArenaMatch> {
        let models = self.models.read();
        let champion = models.values().find(|m| m.status == ArenaStatus::Champion)?;
        let challenger = models.get(model_name)?;

        let gap = (champion_score - challenger_score).abs();
        let significant = gap > self.min_win_gap;
        let winner = if challenger_score > champion_score + self.min_win_gap {
            Some(model_name.to_string())
        } else if champion_score > challenger_score + self.min_win_gap {
            Some(champion.name.clone())
        } else {
            None
        };

        let match_result = ArenaMatch {
            champion: champion.name.clone(),
            challenger: challenger.name.clone(),
            champion_score,
            challenger_score,
            winner,
            significant,
        };

        self.matches.write().push(match_result.clone());
        Some(match_result)
    }

    pub fn leaderboard(&self) -> Vec<ArenaModel> {
        let mut models: Vec<ArenaModel> = self.models.read().values().cloned().collect();
        models.sort_by(|a, b| {
            let a_score = a.metrics.get("elo").copied().unwrap_or(1000.0);
            let b_score = b.metrics.get("elo").copied().unwrap_or(1000.0);
            b_score.partial_cmp(&a_score).unwrap_or(std::cmp::Ordering::Equal)
        });
        models
    }

    pub fn stats(&self) -> serde_json::Value {
        serde_json::json!({
            "total_models": self.models.read().len(),
            "total_matches": self.matches.read().len(),
            "champion": self.models.read().values().find(|m| m.status == ArenaStatus::Champion).map(|m| m.name.clone()),
        })
    }
}
