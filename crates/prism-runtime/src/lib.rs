//! Prism runtime — inference engine for `.cimage` models.
//!
//! Loads a `.cimage` file via [model::Model], then executes inference
//! through [inference::InferenceEngine]. Per-tensor format dispatch
//! routes to Metal kernels (when `metal-dispatch` feature is enabled)
//! or CPU fallback using the quantization codec families from
//! `prism-ecs-quantization`.

pub mod cpu;
pub mod inference;
pub mod metal;
pub mod model;
pub mod sampling;
