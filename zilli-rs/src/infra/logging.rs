use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use tracing_subscriber::prelude::*;

static TRACE_ID: OnceLock<AtomicU64> = OnceLock::new();

pub fn get_trace_id() -> u64 {
    let counter = TRACE_ID.get_or_init(|| AtomicU64::new(1));
    counter.fetch_add(1, Ordering::SeqCst)
}

pub fn setup_logging() {
    use tracing_subscriber::util::SubscriberInitExt;
    let format = tracing_subscriber::fmt::format()
        .with_level(true)
        .with_target(true)
        .with_thread_ids(false)
        .compact();

    let _ = tracing_subscriber::registry()
        .with(tracing_subscriber::fmt::layer().event_format(format))
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .try_init();
}

pub struct StructuredFormatter;
