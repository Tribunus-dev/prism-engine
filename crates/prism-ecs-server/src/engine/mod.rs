//! Prism inference engine — model loading, inference dispatch, streaming,
//! multimodal, tokenization, and measured evaluation.
//!
//! Absorbed from the `prism-runtime` crate into the ECS server crate's
//! crate-level namespace to keep the hot-path inference code organized
//! under one roof.

pub mod bpe_tokenizer;
pub mod cpu;
pub mod cpu_executor;
pub mod engine;
pub mod inference;
pub mod measured;
pub mod model;
pub mod multimodal;
pub mod safetensors;
pub mod sampling;
pub mod streaming;

pub use cpu as cpu_executor_legacy;
pub mod ecs_engine { pub use super::engine::*; }

#[cfg(feature = "metal-dispatch")]
pub mod metal;
pub use engine::PrismEngine;
pub use measured::MeasuredEvaluator;
