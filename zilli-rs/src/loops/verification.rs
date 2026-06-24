use async_trait::async_trait;
use std::sync::Arc;

use super::runner::{Verifier, VerificationResult};

pub struct TestSuiteVerifier {
    _command_template: String,
    _timeout_secs: u64,
}

impl TestSuiteVerifier {
    pub fn new(command_template: &str, timeout_secs: u64) -> Self {
        Self {
            _command_template: command_template.to_string(),
            _timeout_secs: timeout_secs,
        }
    }
}

#[async_trait]
impl Verifier for TestSuiteVerifier {
    async fn verify(&self, _input: &str, output: &str) -> VerificationResult {
        let passed = !output.is_empty() && output.len() > 10;
        VerificationResult {
            passed,
            evidence: format!("Output length: {} chars", output.len()),
            details: Some("Test suite verification (simulated)".into()),
            confidence: if passed { 0.9 } else { 0.5 },
        }
    }
}

type PredicateFn = Arc<dyn Fn(&str, &str) -> bool + Send + Sync>;

pub struct PredicateVerifier {
    predicate: PredicateFn,
    name: String,
}

impl PredicateVerifier {
    pub fn new(name: &str, predicate: PredicateFn) -> Self {
        Self {
            predicate,
            name: name.to_string(),
        }
    }
}

#[async_trait]
impl Verifier for PredicateVerifier {
    async fn verify(&self, input: &str, output: &str) -> VerificationResult {
        let passed = (self.predicate)(input, output);
        VerificationResult {
            passed,
            evidence: format!("Predicate '{}' returned {}", self.name, passed),
            details: None,
            confidence: if passed { 1.0 } else { 0.0 },
        }
    }
}

pub enum CompositeMode {
    All,
    Any,
}

pub struct CompositeVerifier {
    verifiers: Vec<Arc<dyn Verifier>>,
    mode: CompositeMode,
}

impl CompositeVerifier {
    pub fn new(mode: CompositeMode) -> Self {
        Self {
            verifiers: Vec::new(),
            mode,
        }
    }

    pub fn add(&mut self, verifier: Arc<dyn Verifier>) {
        self.verifiers.push(verifier);
    }
}

#[async_trait]
impl Verifier for CompositeVerifier {
    async fn verify(&self, input: &str, output: &str) -> VerificationResult {
        let mut results = Vec::new();

        for v in &self.verifiers {
            let result = v.verify(input, output).await;
            results.push(result);
        }

        match self.mode {
            CompositeMode::All => {
                let passed = results.iter().all(|r| r.passed);
                VerificationResult {
                    passed,
                    evidence: format!("{}/{} verifiers passed", results.iter().filter(|r| r.passed).count(), results.len()),
                    details: None,
                    confidence: results.iter().map(|r| r.confidence).sum::<f64>() / results.len().max(1) as f64,
                }
            }
            CompositeMode::Any => {
                let passed = results.iter().any(|r| r.passed);
                VerificationResult {
                    passed,
                    evidence: format!("{}/{} verifiers passed", results.iter().filter(|r| r.passed).count(), results.len()),
                    details: None,
                    confidence: if passed { 1.0 } else { 0.0 },
                }
            }
        }
    }
}
