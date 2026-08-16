//! From-scratch HTTP inference server: tiny GPT, continuous batching, SSE, metrics.

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
