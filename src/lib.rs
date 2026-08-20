//! HTTP inference server: tiny char GPT, iteration-level continuous batching, SSE, metrics.
//! std-only Rust: handwritten matmuls, dense per-job KV, waiting FIFO → active ≤ max_batch.

pub mod json;
pub mod lm;
pub mod metrics;
pub mod scheduler;
pub mod server;
pub mod tensor;

pub use lm::{LanguageModel, Lcg};
pub use scheduler::{Job, Scheduler, TokenEvent};
pub use server::{serve, serve_listener};

pub const MAX_TOKENS_CAP: usize = 256;
pub const DEFAULT_MAX_TOKENS: usize = 16;
pub const GEN_TIMEOUT_SECS: u64 = 30;
