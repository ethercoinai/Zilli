use chrono::{DateTime, Utc};

pub trait Clock: Send + Sync {
    fn now(&self) -> DateTime<Utc>;
}

#[derive(Clone, Debug)]
pub struct RealClock;

impl Clock for RealClock {
    fn now(&self) -> DateTime<Utc> {
        Utc::now()
    }
}

impl Default for RealClock {
    fn default() -> Self {
        Self
    }
}

#[derive(Clone, Debug)]
pub struct MockClock {
    current: DateTime<Utc>,
}

impl MockClock {
    pub fn new(fixed: DateTime<Utc>) -> Self {
        Self { current: fixed }
    }

    pub fn advance(&mut self, seconds: i64) {
        const MAX_BOUNDS: i64 = 1_000_000_000;
        let s = seconds.clamp(-MAX_BOUNDS, MAX_BOUNDS);
        self.current = self.current + chrono::Duration::seconds(s);
    }
}

impl Clock for MockClock {
    fn now(&self) -> DateTime<Utc> {
        self.current
    }
}
