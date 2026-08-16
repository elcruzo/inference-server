//! Process-wide serving metrics (latency + tokens). No auth.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

#[derive(Default)]
pub struct Metrics {
    pub requests_total: AtomicU64,
    pub requests_ok: AtomicU64,
    pub tokens_total: AtomicU64,
    pub latency_ms_total: AtomicU64,
    pub scheduler_steps: AtomicU64,
}

impl Metrics {
    pub fn record_ok(&self, tokens: u64, started: Instant) {
        let ms = started.elapsed().as_millis() as u64;
        self.requests_total.fetch_add(1, Ordering::Relaxed);
        self.requests_ok.fetch_add(1, Ordering::Relaxed);
        self.tokens_total.fetch_add(tokens, Ordering::Relaxed);
        self.latency_ms_total.fetch_add(ms, Ordering::Relaxed);
    }

    pub fn record_error(&self) {
        self.requests_total.fetch_add(1, Ordering::Relaxed);
    }

    pub fn tick_sched(&self) {
        self.scheduler_steps.fetch_add(1, Ordering::Relaxed);
    }

    pub fn snapshot_json(&self) -> String {
        let req = self.requests_total.load(Ordering::Relaxed);
        let ok = self.requests_ok.load(Ordering::Relaxed);
        let tokens = self.tokens_total.load(Ordering::Relaxed);
        let lat = self.latency_ms_total.load(Ordering::Relaxed);
        let steps = self.scheduler_steps.load(Ordering::Relaxed);
        let mean_latency_ms = if ok == 0 { 0.0 } else { lat as f64 / ok as f64 };
        let tokens_per_sec = if lat == 0 {
            0.0
        } else {
            tokens as f64 / (lat as f64 / 1000.0)
        };
        format!(
            "{{\"requests_total\":{req},\"requests_ok\":{ok},\"tokens_total\":{tokens},\
\"latency_ms_total\":{lat},\"mean_latency_ms\":{mean_latency_ms:.3},\
\"tokens_per_sec\":{tokens_per_sec:.3},\"scheduler_steps\":{steps}}}"
        )
    }
}
