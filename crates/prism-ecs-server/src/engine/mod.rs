//! Prism inference engine — model loading, inference dispatch, streaming,
//! multimodal, tokenization, and measured evaluation.
//!
//! Absorbed from the `prism-runtime` crate into the ECS server crate's
//! crate-level namespace to keep the hot-path inference code organized
//! under one roof.

pub mod bpe_tokenizer;
pub mod cpu;
pub mod engine;
pub mod inference;
pub mod measured;
pub mod model;
pub mod multimodal;
pub mod safetensors;
pub mod sampling;
pub mod streaming;

#[cfg(feature = "metal-dispatch")]
pub mod metal;
pub use engine::PrismEngine;
pub use measured::MeasuredEvaluator;
