use std::collections::VecDeque;
use parking_lot::RwLock;

pub struct LengthElasticController {
    initial_cap: usize,
    min_cap: usize,
    max_cap: usize,
    current_cap: RwLock<usize>,
    history: RwLock<VecDeque<f64>>,
    _window_size: usize,
}

impl LengthElasticController {
    pub fn new(initial_cap: usize, min_cap: usize, max_cap: usize) -> Self {
        Self {
            initial_cap,
            min_cap,
            max_cap: max_cap.max(min_cap),
            current_cap: RwLock::new(initial_cap),
            history: RwLock::new(VecDeque::with_capacity(100)),
            _window_size: 20,
        }
    }

    pub fn adapt(&self, sequence_lengths: &[usize]) {
        if sequence_lengths.is_empty() {
            return;
        }

        let avg_len = sequence_lengths.iter().sum::<usize>() as f64 / sequence_lengths.len() as f64;
        let _max_len = *sequence_lengths.iter().max().unwrap_or(&0) as f64;
        let p95 = self.percentile(sequence_lengths, 0.95);

        let mut history = self.history.write();
        history.push_back(avg_len);
        if history.len() > 100 {
            history.pop_front();
        }

        let trend: f64 = if history.len() > 1 {
            let recent = history.iter().rev().take(10).sum::<f64>() / 10.0;
            let older = history.iter().rev().skip(10).take(10).sum::<f64>() / (10.0_f64).max(1.0);
            recent - older
        } else {
            0.0
        };

        let mut cap = self.current_cap.write();

        if p95 > *cap as f64 * 0.9 && trend > 0.0 {
            *cap = (*cap as f64 * 1.5).min(self.max_cap as f64) as usize;
        } else if p95 < *cap as f64 * 0.3 && trend < 0.0 {
            *cap = (*cap as f64 * 0.75).max(self.min_cap as f64) as usize;
        }
    }

    fn percentile(&self, values: &[usize], p: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        let mut sorted = values.to_vec();
        sorted.sort_unstable();
        let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
        sorted[idx.clamp(0, sorted.len() - 1)] as f64
    }

    pub fn get_cap(&self) -> usize {
        *self.current_cap.read()
    }

    pub fn get_stats(&self) -> serde_json::Value {
        serde_json::json!({
            "current_cap": *self.current_cap.read(),
            "initial_cap": self.initial_cap,
            "min_cap": self.min_cap,
            "max_cap": self.max_cap,
            "history_len": self.history.read().len(),
        })
    }
}

pub struct LayoutAwareDispatcher;

impl LayoutAwareDispatcher {
    pub fn dispatch(&self, data: &[String], num_workers: usize) -> Vec<Vec<String>> {
        let mut result = vec![Vec::new(); num_workers.max(1)];
        for (i, item) in data.iter().enumerate() {
            result[i % num_workers.max(1)].push(item.clone());
        }
        result
    }
}
