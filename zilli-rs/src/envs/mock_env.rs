use serde::{Deserialize, Serialize};
use std::sync::Arc;
use parking_lot::RwLock;
use std::collections::HashMap;

use crate::schema::actions::BaseAction;

pub type ToolFn = Arc<dyn Fn(&str) -> Result<String, String> + Send + Sync>;

pub struct HermesSandbox {
    _scenario: Option<String>,
    trajectory: RwLock<Vec<serde_json::Value>>,
    tools: RwLock<HashMap<String, ToolFn>>,
    memory: RwLock<HashMap<String, String>>,
    step_count: RwLock<i32>,
    max_steps: i32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepResult {
    pub success: bool,
    pub observation: String,
    pub done: bool,
    pub reward: Option<f64>,
}

impl HermesSandbox {
    pub fn new(max_steps: i32) -> Self {
        Self {
            _scenario: None,
            trajectory: RwLock::new(Vec::new()),
            tools: RwLock::new(HashMap::new()),
            memory: RwLock::new(HashMap::new()),
            step_count: RwLock::new(0),
            max_steps,
        }
    }

    pub fn register_tool<F>(&self, name: &str, tool: F)
    where
        F: Fn(&str) -> Result<String, String> + Send + Sync + 'static,
    {
        self.tools.write().insert(name.to_string(), Arc::new(tool));
    }

    pub async fn step(&self, action: BaseAction) -> StepResult {
        let mut step_count = self.step_count.write();
        *step_count += 1;
        let current_step = *step_count;

        if current_step > self.max_steps {
            return StepResult {
                success: false,
                observation: "Max steps exceeded".into(),
                done: true,
                reward: Some(0.0),
            };
        }

        let result = match action.tool_name {
            crate::schema::actions::ActionType::BashRun => {
                "Command executed (simulated)".to_string()
            }
            crate::schema::actions::ActionType::FileRead => {
                self.memory.read().get("file_content").cloned().unwrap_or_default()
            }
            crate::schema::actions::ActionType::FileWrite => {
                self.memory.write()
                    .insert("file_content".into(), "written".into());
                "File written (simulated)".to_string()
            }
            crate::schema::actions::ActionType::MemoryWrite => {
                "Memory written".to_string()
            }
            crate::schema::actions::ActionType::MemoryRead => {
                self.memory.read().get("memory_value").cloned().unwrap_or_default()
            }
            crate::schema::actions::ActionType::Finish => {
                return StepResult {
                    success: true,
                    observation: "Task completed".into(),
                    done: true,
                    reward: Some(1.0),
                };
            }
            _ => "Unknown action".to_string(),
        };

        self.trajectory.write().push(serde_json::json!({
            "step": current_step,
            "action": action.tool_name.to_string(),
            "result": result,
        }));

        StepResult {
            success: true,
            observation: result,
            done: false,
            reward: None,
        }
    }

    pub fn get_trajectory(&self) -> Vec<serde_json::Value> {
        self.trajectory.read().clone()
    }

    pub fn reset(&self) {
        *self.step_count.write() = 0;
        self.trajectory.write().clear();
        self.memory.write().clear();
    }
}
