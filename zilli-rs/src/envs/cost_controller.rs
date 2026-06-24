use chrono::{DateTime, Utc};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BudgetSnapshot {
    pub remaining_budget: f64,
    pub total_calls: u64,
    pub calls_this_hour: u64,
    pub hourly_quota: f64,
    pub emergency_mode: bool,
    pub timestamp: DateTime<Utc>,
}

pub struct CostController {
    monthly_budget: f64,
    _budget_file: Option<String>,
    planner_calls: RwLock<u64>,
    executor_calls: RwLock<u64>,
    planner_cost: RwLock<f64>,
    executor_cost: RwLock<f64>,
    task_stats: RwLock<HashMap<String, TaskStats>>,
    current_hour_calls: RwLock<u64>,
    current_hour_start: RwLock<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TaskStats {
    total_calls: u64,
    planner_calls: u64,
    success_rate: f64,
    avg_conf: f64,
}

impl CostController {
    pub fn new(monthly_budget: f64) -> Self {
        Self {
            monthly_budget,
            _budget_file: None,
            planner_calls: RwLock::new(0),
            executor_calls: RwLock::new(0),
            planner_cost: RwLock::new(0.0),
            executor_cost: RwLock::new(0.0),
            task_stats: RwLock::new(HashMap::new()),
            current_hour_calls: RwLock::new(0),
            current_hour_start: RwLock::new(Utc::now()),
        }
    }

    pub fn should_use_planner(&self, task_type: &str, state: &PlannerState) -> bool {
        let remaining = self.monthly_budget - *self.planner_cost.read();
        if remaining < self.monthly_budget * 0.1 {
            let stats = self.task_stats.read();
            if let Some(s) = stats.get(task_type) {
                return s.success_rate < 0.3;
            }
            return false;
        }

        let stats = self.task_stats.read();
        let task_stats = stats.get(task_type);

        if let Some(s) = task_stats {
            if s.avg_conf < 0.6 && s.success_rate < 0.7 {
                return true;
            }
            if s.avg_conf > 0.9 && s.success_rate > 0.9 {
                return false;
            }
        }

        state.confidence < 0.7 || state.difficulty > 0.7
    }

    pub fn record_planner_call(&self, task_type: &str, success: bool) {
        *self.planner_calls.write() += 1;
        *self.planner_cost.write() += 0.05;
        *self.current_hour_calls.write() += 1;
        self.update_task_stats(task_type, success, true);
    }

    pub fn record_executor_call(&self, task_type: &str, success: bool) {
        *self.executor_calls.write() += 1;
        *self.executor_cost.write() += 0.001;
        *self.current_hour_calls.write() += 1;
        self.update_task_stats(task_type, success, false);
    }

    fn update_task_stats(&self, task_type: &str, _success: bool, used_planner: bool) {
        let mut stats = self.task_stats.write();
        let entry = stats.entry(task_type.to_string()).or_insert(TaskStats {
            total_calls: 0,
            planner_calls: 0,
            success_rate: 0.5,
            avg_conf: 0.7,
        });
        entry.total_calls += 1;
        if used_planner {
            entry.planner_calls += 1;
        }
    }

    pub fn snapshot(&self) -> BudgetSnapshot {
        let remaining = self.monthly_budget - *self.planner_cost.read() - *self.executor_cost.read();
        let hourly = self.monthly_budget / (30.0 * 24.0);
        let calls = *self.current_hour_calls.read();
        let emergency = remaining < self.monthly_budget * 0.1;

        BudgetSnapshot {
            remaining_budget: remaining.max(0.0),
            total_calls: *self.planner_calls.read() + *self.executor_calls.read(),
            calls_this_hour: calls,
            hourly_quota: hourly * 10.0,
            emergency_mode: emergency,
            timestamp: Utc::now(),
        }
    }

    pub fn reset_monthly(&self) {
        *self.planner_calls.write() = 0;
        *self.executor_calls.write() = 0;
        *self.planner_cost.write() = 0.0;
        *self.executor_cost.write() = 0.0;
        *self.current_hour_calls.write() = 0;
        *self.current_hour_start.write() = Utc::now();
        self.task_stats.write().clear();
    }
}

pub struct PlannerState {
    pub confidence: f64,
    pub difficulty: f64,
}

impl PlannerState {
    pub fn new(confidence: f64, difficulty: f64) -> Self {
        Self {
            confidence,
            difficulty,
        }
    }
}
