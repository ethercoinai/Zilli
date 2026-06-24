use async_trait::async_trait;
use std::time::Duration;
use tokio::time;

use super::runner::Trigger;

pub struct FixedIntervalTrigger {
    interval: Duration,
    jitter: f64,
}

impl FixedIntervalTrigger {
    pub fn new(interval_secs: u64, jitter: f64) -> Self {
        Self {
            interval: Duration::from_secs(interval_secs),
            jitter,
        }
    }
}

#[async_trait]
impl Trigger for FixedIntervalTrigger {
    async fn wait(&self) -> bool {
        let jitter_ms = (self.jitter * self.interval.as_secs_f64() * 1000.0).abs();
        let jitter = if jitter_ms > 0.0 {
            let offset = rand::random::<f64>() * jitter_ms;
            Duration::from_millis(offset as u64)
        } else {
            Duration::from_millis(0)
        };
        time::sleep(self.interval + jitter).await;
        true
    }

    async fn reset(&self) {}
}

pub struct EventTrigger {
    event: tokio::sync::Notify,
}

impl EventTrigger {
    pub fn new() -> Self {
        Self {
            event: tokio::sync::Notify::new(),
        }
    }

    pub fn fire(&self) {
        self.event.notify_one();
    }
}

#[async_trait]
impl Trigger for EventTrigger {
    async fn wait(&self) -> bool {
        self.event.notified().await;
        true
    }

    async fn reset(&self) {}
}

pub struct DynamicIntervalTrigger {
    base_interval: Duration,
    max_interval: Duration,
    current_backoff: f64,
}

impl DynamicIntervalTrigger {
    pub fn new(base_secs: u64, max_secs: u64) -> Self {
        Self {
            base_interval: Duration::from_secs(base_secs),
            max_interval: Duration::from_secs(max_secs),
            current_backoff: 1.0,
        }
    }
}

#[async_trait]
impl Trigger for DynamicIntervalTrigger {
    async fn wait(&self) -> bool {
        let interval = self
            .base_interval
            .mul_f64(self.current_backoff)
            .min(self.max_interval);
        time::sleep(interval).await;
        true
    }

    async fn reset(&self) {
        // Reset is a no-op in this simple implementation
    }
}
