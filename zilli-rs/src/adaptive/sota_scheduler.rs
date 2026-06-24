use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use crate::models::registry::ModelRegistry;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerStats {
    pub total_calls: u64,
    pub planner_calls: u64,
    pub executor_calls: u64,
    pub total_cost: f64,
    pub remaining_budget: f64,
    pub emergency_mode: bool,
}

pub struct DynamicSOTAScheduler {
    monthly_budget_usd: f64,
    cost_per_call: f64,
    _model_registry: Option<ModelRegistry>,
    task_stats: RwLock<HashMap<String, TaskStats>>,
    hourly_calls: RwLock<u64>,
    total_planner_calls: RwLock<u64>,
    total_executor_calls: RwLock<u64>,
    total_cost: RwLock<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TaskStats {
    failure_rate: f64,
    total_tasks: u64,
    planner_calls: u64,
    success_with_planner: f64,
    success_without_planner: f64,
}

impl DynamicSOTAScheduler {
    pub fn new(monthly_budget_usd: f64, cost_per_call: f64) -> Self {
        Self {
            monthly_budget_usd,
            cost_per_call,
            _model_registry: None,
            task_stats: RwLock::new(HashMap::new()),
            hourly_calls: RwLock::new(0),
            total_planner_calls: RwLock::new(0),
            total_executor_calls: RwLock::new(0),
            total_cost: RwLock::new(0.0),
        }
    }

    pub fn should_call_planner(
        &self,
        task_type: &str,
        executor_confidence: f64,
    ) -> bool {
        let remaining = self.monthly_budget_usd - *self.total_cost.read();
        if remaining < self.monthly_budget_usd * 0.1 {
            let stats = self.task_stats.read();
            if let Some(s) = stats.get(task_type) {
                return s.failure_rate > 0.3;
            }
            return false;
        }

        let stats = self.task_stats.read();
        if let Some(s) = stats.get(task_type) {
            let gap = s.success_with_planner - s.success_without_planner;
            if gap > 0.2 && executor_confidence < 0.8 {
                return true;
            }
            if gap < 0.05 && executor_confidence > 0.9 {
                return false;
            }
        }

        executor_confidence < 0.7
    }

    pub fn record_call(&self, model_name: &str, task_type: &str, success: bool) {
        *self.total_cost.write() += self.cost_per_call;
        *self.hourly_calls.write() += 1;

        if model_name.contains("planner") || model_name.contains("sota") {
            *self.total_planner_calls.write() += 1;
        } else {
            *self.total_executor_calls.write() += 1;
        }

        let mut stats = self.task_stats.write();
        let entry = stats.entry(task_type.to_string()).or_insert(TaskStats {
            failure_rate: 0.3,
            total_tasks: 0,
            planner_calls: 0,
            success_with_planner: 0.8,
            success_without_planner: 0.6,
        });

        entry.total_tasks += 1;
        if model_name.contains("planner") || model_name.contains("sota") {
            entry.planner_calls += 1;
        }

        entry.failure_rate = if entry.total_tasks > 0 {
            0.3 * entry.failure_rate + 0.7 * if success { 0.0 } else { 1.0 }
        } else {
            0.3
        };
    }

    pub fn record_without_planner(&self, task_type: &str, success: bool) {
        let mut stats = self.task_stats.write();
        let entry = stats.entry(task_type.to_string()).or_insert(TaskStats {
            failure_rate: 0.3,
            total_tasks: 0,
            planner_calls: 0,
            success_with_planner: 0.8,
            success_without_planner: 0.6,
        });

        entry.total_tasks += 1;

        // Update success rate with EMA
        let rate = if success { 1.0 } else { 0.0 };
        entry.success_without_planner =
            0.9 * entry.success_without_planner + 0.1 * rate;
    }

    pub fn stats(&self) -> SchedulerStats {
        let total = *self.total_planner_calls.read() + *self.total_executor_calls.read();
        let remaining = self.monthly_budget_usd - *self.total_cost.read();
        let emergency = remaining < self.monthly_budget_usd * 0.1;

        SchedulerStats {
            total_calls: total,
            planner_calls: *self.total_planner_calls.read(),
            executor_calls: *self.total_executor_calls.read(),
            total_cost: *self.total_cost.read(),
            remaining_budget: remaining.max(0.0),
            emergency_mode: emergency,
        }
    }

    pub fn reset_hourly_counter(&self) {
        *self.hourly_calls.write() = 0;
    }
}
