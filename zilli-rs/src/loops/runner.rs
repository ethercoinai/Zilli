use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use super::memory::CycleMemory;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerificationResult {
    pub passed: bool,
    pub evidence: String,
    pub details: Option<String>,
    pub confidence: f64,
}

#[cfg_attr(test, mockall::automock)]
#[async_trait]
pub trait Verifier: Send + Sync {
    async fn verify(&self, input: &str, output: &str) -> VerificationResult;
}

#[async_trait]
pub trait Trigger: Send + Sync {
    async fn wait(&self) -> bool;
    async fn reset(&self);
}

#[async_trait]
pub trait EscalationHandler: Send + Sync {
    async fn handle(&self, cycles: &[LoopCycle], error: &str) -> Option<String>;
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopCycle {
    pub id: String,
    pub input_data: String,
    pub output: Option<String>,
    pub verification: Option<VerificationResult>,
    pub duration_ms: u64,
    pub error: Option<String>,
    pub metadata: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoopResult {
    pub cycles: Vec<LoopCycle>,
    pub final_output: Option<String>,
    pub passed: bool,
    pub total_duration_ms: u64,
}

pub struct LoopRunner {
    process_fn: Box<dyn Fn(String) -> String + Send + Sync>,
    verifier: Box<dyn Verifier>,
    trigger: Option<Box<dyn Trigger>>,
    max_retries: i32,
    memory: Option<CycleMemory>,
    escalation_handler: Option<Box<dyn EscalationHandler>>,
}

impl LoopRunner {
    pub fn new(
        process_fn: Box<dyn Fn(String) -> String + Send + Sync>,
        verifier: Box<dyn Verifier>,
        max_retries: i32,
    ) -> Self {
        Self {
            process_fn,
            verifier,
            trigger: None,
            max_retries,
            memory: None,
            escalation_handler: None,
        }
    }

    pub fn with_trigger(mut self, trigger: Box<dyn Trigger>) -> Self {
        self.trigger = Some(trigger);
        self
    }

    pub fn with_memory(mut self, memory: CycleMemory) -> Self {
        self.memory = Some(memory);
        self
    }

    pub fn with_escalation(mut self, handler: Box<dyn EscalationHandler>) -> Self {
        self.escalation_handler = Some(handler);
        self
    }

    pub async fn run(&self, input: &str) -> LoopResult {
        let start = std::time::Instant::now();
        let mut cycles = Vec::new();
        let original_input = input.to_string();
        let mut current_input = input.to_string();

        for attempt in 0..=self.max_retries {
            let cycle_start = std::time::Instant::now();

            let output = (self.process_fn)(current_input.clone());
            let verification = self.verifier.verify(&current_input, &output).await;

            let cycle = LoopCycle {
                id: format!("cycle_{}", attempt),
                input_data: current_input.clone(),
                output: Some(output.clone()),
                verification: Some(verification.clone()),
                duration_ms: cycle_start.elapsed().as_millis() as u64,
                error: None,
                metadata: None,
            };

            if let Some(ref memory) = self.memory {
                memory.store(cycle.clone());
            }

            let passed = verification.passed;
            cycles.push(cycle);

            if passed {
                let total_duration = start.elapsed().as_millis() as u64;
                return LoopResult {
                    cycles,
                    final_output: Some(output),
                    passed: true,
                    total_duration_ms: total_duration,
                };
            }

            if attempt < self.max_retries {
                current_input = original_input.clone();
            }
        }

        let total_duration = start.elapsed().as_millis() as u64;

        if let Some(ref handler) = self.escalation_handler {
            let last_error = cycles
                .last()
                .and_then(|c| c.error.clone())
                .unwrap_or_else(|| "Max retries exceeded".into());
            handler
                .handle(&cycles, &last_error)
                .await;
        }

        LoopResult {
            cycles,
            final_output: None,
            passed: false,
            total_duration_ms: total_duration,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_loop_runner_passes_first_attempt() {
        let mut mock = MockVerifier::new();
        mock.expect_verify()
            .returning(|_input, _output| VerificationResult {
                passed: true,
                evidence: "ok".into(),
                details: None,
                confidence: 1.0,
            });

        let runner = LoopRunner::new(
            Box::new(|s| format!("processed: {}", s)),
            Box::new(mock),
            3,
        );
        let result = runner.run("hello").await;
        assert!(result.passed);
        assert_eq!(result.cycles.len(), 1);
        assert_eq!(result.final_output, Some("processed: hello".into()));
    }

    #[tokio::test]
    async fn test_loop_runner_retries_on_failure() {
        let mut mock = MockVerifier::new();
        // Return failed on first call, passed on second
        let call_count = std::sync::atomic::AtomicUsize::new(0);
        mock.expect_verify()
            .returning(move |_input, _output| {
                let prev = call_count.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                VerificationResult {
                    passed: prev > 0,
                    evidence: if prev > 0 { "passed".into() } else { "failed".into() },
                    details: None,
                    confidence: if prev > 0 { 1.0 } else { 0.0 },
                }
            });

        let runner = LoopRunner::new(
            Box::new(|s| format!("processed: {}", s)),
            Box::new(mock),
            3,
        );
        let result = runner.run("hello").await;
        assert!(result.passed);
        assert_eq!(result.cycles.len(), 2);
    }

    #[tokio::test]
    async fn test_loop_runner_exhausts_retries() {
        let mut mock = MockVerifier::new();
        mock.expect_verify()
            .returning(|_input, _output| VerificationResult {
                passed: false,
                evidence: "always fails".into(),
                details: None,
                confidence: 0.0,
            });

        let runner = LoopRunner::new(
            Box::new(|s| format!("processed: {}", s)),
            Box::new(mock),
            2,
        );
        let result = runner.run("hello").await;
        assert!(!result.passed);
        assert_eq!(result.cycles.len(), 3); // 0..=2 (max_retries=2 → 3 attempts)
        assert!(result.final_output.is_none());
    }
}
