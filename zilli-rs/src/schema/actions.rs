use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, strum::Display)]
#[serde(rename_all = "snake_case")]
#[strum(serialize_all = "snake_case")]
pub enum ActionType {
    MemoryWrite,
    MemoryRead,
    SkillCreate,
    SkillUpdate,
    BashRun,
    FileRead,
    FileWrite,
    Finish,
    Think,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaseAction {
    pub action_id: String,
    pub reasoning: Option<String>,
    pub tool_name: ActionType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryWriteAction {
    pub action_id: String,
    pub reasoning: Option<String>,
    pub key: String,
    pub value: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryReadAction {
    pub action_id: String,
    pub reasoning: Option<String>,
    pub key: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillCreateAction {
    pub action_id: String,
    pub reasoning: Option<String>,
    pub name: String,
    pub code: String,
    pub boundary: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SkillUpdateAction {
    pub action_id: String,
    pub reasoning: Option<String>,
    pub name: String,
    pub code: String,
    pub boundary: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BashRunAction {
    pub action_id: String,
    pub reasoning: Option<String>,
    pub command: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileReadAction {
    pub action_id: String,
    pub reasoning: Option<String>,
    pub path: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileWriteAction {
    pub action_id: String,
    pub reasoning: Option<String>,
    pub path: String,
    pub content: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FinishAction {
    pub action_id: String,
    pub reasoning: Option<String>,
    pub summary: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskConfig {
    pub id: String,
    pub name: String,
    pub description: String,
    pub category: String,
    pub max_steps: i32,
    pub initial_context: Option<String>,
    pub verification: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RewardRule {
    #[serde(rename = "type")]
    pub rule_type: String,
    pub weight: f64,
    pub params: Option<HashMap<String, serde_json::Value>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryTemplateStep {
    pub step_number: i32,
    pub action_type: String,
    pub required_fields: Vec<String>,
    pub expected_observation: Option<String>,
}
