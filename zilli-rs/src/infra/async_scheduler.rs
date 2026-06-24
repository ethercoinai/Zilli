use futures::future::join_all;
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicI32, Ordering};
use std::sync::Arc;
use strum::{Display, EnumString};
use tokio::sync::Semaphore;
use tokio::time::{timeout, Duration};

#[derive(Debug, Clone, Serialize, Deserialize, Display, EnumString, PartialEq)]
pub enum RolloutStatus {
    Pending,
    Running,
    Completed,
    Timeout,
    Failed,
    Cancelled,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RolloutResult {
    pub task_id: String,
    pub reward: Option<f64>,
    pub tokens: u64,
    pub completed: bool,
    pub error: Option<String>,
    pub status: RolloutStatus,
    pub elapsed_sec: f64,
}

type RolloutFn = Arc<dyn Fn(String) -> Result<String, String> + Send + Sync>;

pub struct AsyncRolloutScheduler {
    max_concurrent: usize,
    semaphore: Arc<Semaphore>,
    total_scheduled: AtomicI32,
    total_completed: AtomicI32,
}

impl AsyncRolloutScheduler {
    pub fn new(max_concurrent: usize) -> Self {
        Self {
            max_concurrent,
            semaphore: Arc::new(Semaphore::new(max_concurrent)),
            total_scheduled: AtomicI32::new(0),
            total_completed: AtomicI32::new(0),
        }
    }

    pub async fn schedule(
        &self,
        rollout_fn: RolloutFn,
        tasks: Vec<String>,
        timeout_per_task_secs: u64,
    ) -> Vec<RolloutResult> {
        let mut handles = Vec::new();

        for (i, task) in tasks.into_iter().enumerate() {
            self.total_scheduled.fetch_add(1, Ordering::SeqCst);
            let sem = self.semaphore.clone();
            let fn_clone = rollout_fn.clone();

            let handle = tokio::spawn(async move {
                let permit = sem.acquire().await;
                let _permit = match permit {
                    Ok(p) => p,
                    Err(_) => return RolloutResult {
                        task_id: format!("task_{}", i),
                        reward: None,
                        tokens: 0,
                        completed: false,
                        error: Some("Semaphore closed, scheduler shutting down".into()),
                        status: RolloutStatus::Failed,
                        elapsed_sec: 0.0,
                    },
                };
                let start = std::time::Instant::now();
                let task_id = format!("task_{}", i);

                let result = timeout(
                    Duration::from_secs(timeout_per_task_secs),
                    tokio::task::spawn_blocking(move || fn_clone(task)),
                )
                .await;

                let elapsed = start.elapsed().as_secs_f64();

                match result {
                    Ok(Ok(Ok(_text))) => RolloutResult {
                        task_id,
                        reward: Some(1.0),
                        tokens: 100,
                        completed: true,
                        error: None,
                        status: RolloutStatus::Completed,
                        elapsed_sec: elapsed,
                    },
                    Ok(Ok(Err(e))) => RolloutResult {
                        task_id,
                        reward: Some(0.0),
                        tokens: 50,
                        completed: false,
                        error: Some(e),
                        status: RolloutStatus::Failed,
                        elapsed_sec: elapsed,
                    },
                    Ok(Err(e)) => RolloutResult {
                        task_id,
                        reward: None,
                        tokens: 0,
                        completed: false,
                        error: Some(format!("Join error: {}", e)),
                        status: RolloutStatus::Failed,
                        elapsed_sec: elapsed,
                    },
                    Err(_) => RolloutResult {
                        task_id,
                        reward: None,
                        tokens: 0,
                        completed: false,
                        error: Some("Timeout".into()),
                        status: RolloutStatus::Timeout,
                        elapsed_sec: elapsed,
                    },
                }
            });

            handles.push(handle);
        }

        let results = join_all(handles).await;
        let mut final_results = Vec::new();

        for r in results {
            match r {
                Ok(result) => {
                    self.total_completed.fetch_add(1, Ordering::SeqCst);
                    final_results.push(result);
                }
                Err(e) => {
                    final_results.push(RolloutResult {
                        task_id: "unknown".into(),
                        reward: None,
                        tokens: 0,
                        completed: false,
                        error: Some(format!("Task panicked: {}", e)),
                        status: RolloutStatus::Failed,
                        elapsed_sec: 0.0,
                    });
                }
            }
        }

        final_results
    }

    pub fn get_stats(&self) -> serde_json::Value {
        serde_json::json!({
            "max_concurrent": self.max_concurrent,
            "total_scheduled": self.total_scheduled.load(Ordering::SeqCst),
            "total_completed": self.total_completed.load(Ordering::SeqCst),
        })
    }
}
