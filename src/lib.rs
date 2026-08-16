//! From-scratch HTTP inference server: bigram LM, continuous-batching-lite, SSE.

pub mod json;
pub mod lm;
pub mod scheduler;
pub mod server;

pub use lm::{LanguageModel, Lcg, CORPUS};
pub use scheduler::{Job, Scheduler, TokenEvent};
pub use server::{serve, serve_listener};

pub const MAX_TOKENS_CAP: usize = 256;
pub const DEFAULT_MAX_TOKENS: usize = 16;
pub const GEN_TIMEOUT_SECS: u64 = 30;
