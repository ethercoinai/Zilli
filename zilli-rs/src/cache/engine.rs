use chrono::{DateTime, Utc};
use lru::LruCache;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::infra::clock::{Clock, RealClock};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheConfig {
    pub enabled: bool,
    pub memory_size: usize,
    pub ttl_seconds: u64,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            memory_size: 1000,
            ttl_seconds: 3600,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub prompt_hash: String,
    pub response_text: String,
    pub model_name: String,
    pub tokens_in: u64,
    pub tokens_out: u64,
    pub created_at: DateTime<Utc>,
    pub hit_count: u64,
}

impl CacheEntry {
    pub fn is_expired(&self, ttl: &std::time::Duration) -> bool {
        let elapsed = Utc::now() - self.created_at;
        elapsed.num_seconds() >= ttl.as_secs() as i64
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheStats {
    pub entries: usize,
    pub hits: u64,
    pub misses: u64,
    pub memory_entries: usize,
}

pub struct CacheEngine {
    config: CacheConfig,
    cache: RwLock<LruCache<String, CacheEntry>>,
    hits: AtomicU64,
    misses: AtomicU64,
    clock: Box<dyn Clock>,
}

fn hash_key(prompt: &str, model_name: &str, temperature: f64, max_tokens: Option<i32>) -> String {
    let mut hasher = Sha256::new();
    hasher.update(prompt.as_bytes());
    hasher.update(model_name.as_bytes());
    hasher.update(temperature.to_le_bytes());
    hasher.update(max_tokens.unwrap_or(0).to_le_bytes());
    hex::encode(hasher.finalize())
}

impl CacheEngine {
    pub fn new(config: CacheConfig) -> Self {
        let size = NonZeroUsize::new(config.memory_size.max(1)).unwrap();
        Self {
            config,
            cache: RwLock::new(LruCache::new(size)),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            clock: Box::new(RealClock),
        }
    }

    pub fn with_clock(config: CacheConfig, clock: Box<dyn Clock>) -> Self {
        let size = NonZeroUsize::new(config.memory_size.max(1)).unwrap();
        Self {
            config,
            cache: RwLock::new(LruCache::new(size)),
            hits: AtomicU64::new(0),
            misses: AtomicU64::new(0),
            clock,
        }
    }

    pub fn get(&self, prompt: &str, model_name: &str, temperature: f64, max_tokens: Option<i32>) -> Option<CacheEntry> {
        if !self.config.enabled {
            self.misses.fetch_add(1, Ordering::Relaxed);
            return None;
        }

        let key = hash_key(prompt, model_name, temperature, max_tokens);
        let mut cache = self.cache.write();

        if let Some(entry) = cache.get(&key) {
            let ttl = std::time::Duration::from_secs(self.config.ttl_seconds);
            if entry.is_expired(&ttl) {
                cache.pop(&key);
                self.misses.fetch_add(1, Ordering::Relaxed);
                return None;
            }
            let mut entry = entry.clone();
            entry.hit_count += 1;
            self.hits.fetch_add(1, Ordering::Relaxed);
            Some(entry)
        } else {
            self.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    pub fn set(
        &self,
        prompt: &str,
        model_name: &str,
        response_text: &str,
        tokens_in: u64,
        tokens_out: u64,
        temperature: f64,
        max_tokens: Option<i32>,
    ) {
        if !self.config.enabled {
            return;
        }

        let key = hash_key(prompt, model_name, temperature, max_tokens);
        let entry = CacheEntry {
            prompt_hash: key.clone(),
            response_text: response_text.to_string(),
            model_name: model_name.to_string(),
            tokens_in,
            tokens_out,
            created_at: self.clock.now(),
            hit_count: 0,
        };

        self.cache.write().put(key, entry);
    }

    pub fn invalidate(&self, prompt: &str, model_name: &str, temperature: f64, max_tokens: Option<i32>) {
        let key = hash_key(prompt, model_name, temperature, max_tokens);
        self.cache.write().pop(&key);
    }

    pub fn clear(&self) {
        let size = NonZeroUsize::new(self.config.memory_size.max(1)).unwrap();
        *self.cache.write() = LruCache::new(size);
    }

    pub fn stats(&self) -> CacheStats {
        let cache = self.cache.read();
        CacheStats {
            entries: cache.len(),
            hits: self.hits.load(Ordering::Relaxed),
            misses: self.misses.load(Ordering::Relaxed),
            memory_entries: cache.len(),
        }
    }
}

pub fn noop_cache() -> CacheEngine {
    CacheEngine::new(CacheConfig {
        enabled: false,
        memory_size: 1,
        ttl_seconds: 0,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_set_get() {
        let cfg = CacheConfig { enabled: true, memory_size: 100, ttl_seconds: 3600 };
        let cache = CacheEngine::new(cfg);
        cache.set("hello", "model-a", "world", 10, 20, 0.0, None);
        let entry = cache.get("hello", "model-a", 0.0, None);
        assert!(entry.is_some());
        assert_eq!(entry.unwrap().response_text, "world");
    }

    #[test]
    fn test_cache_miss() {
        let cfg = CacheConfig { enabled: true, memory_size: 100, ttl_seconds: 3600 };
        let cache = CacheEngine::new(cfg);
        let entry = cache.get("nonexistent", "model-a", 0.0, None);
        assert!(entry.is_none());
    }

    #[test]
    fn test_cache_expiry() {
        let cfg = CacheConfig { enabled: true, memory_size: 100, ttl_seconds: 0 };
        let cache = CacheEngine::new(cfg);
        cache.set("hello", "model-a", "world", 10, 20, 0.0, None);
        let entry = cache.get("hello", "model-a", 0.0, None);
        assert!(entry.is_none(), "entry with TTL=0 should be expired immediately");
    }

    #[test]
    fn test_cache_disabled() {
        let cfg = CacheConfig { enabled: false, memory_size: 100, ttl_seconds: 3600 };
        let cache = CacheEngine::new(cfg);
        cache.set("hello", "model-a", "world", 10, 20, 0.0, None);
        let entry = cache.get("hello", "model-a", 0.0, None);
        assert!(entry.is_none());
    }
}
