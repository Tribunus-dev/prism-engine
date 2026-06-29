//! Gemma 4 full-transformer GPU megakernel.
//!
//! Splits the original monolithic `megakernel.rs` into three concerns:
//! - [`kernels`] — architecture constants, Metal shader source, on-the-fly compilation
//! - [`kv`] — ternary KV cache block constants
//! - [`pipeline`] — persistent dispatch, work queue, buffer management, host API

pub mod kernels;
pub mod kv;
pub mod pipeline;

pub use kernels::*;
pub use kv::*;
pub use pipeline::*;
