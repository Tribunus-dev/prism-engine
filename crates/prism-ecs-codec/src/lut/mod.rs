//! Lookup-table (LUT) codec surface — palettized weight tables, FP16
//! math kernels, INT8 KV-cache quantization helpers, and model
//! graph descriptors.
//!
//! This module owns the canonical authority for the
//! backend-neutral LUT codec: the model architecture descriptor
//! that drives AOT compilation, the palettized matrix format
//! (codebook + packed 4-bit indices), the FP16 math kernels used
//! as a CPU fallback for inference, and the symmetric INT8
//! per-token quantization used by the KV cache.
//!
//! All types here are backend-neutral. Hardware-specific
//! execution paths (Metal, ANE, MLX) live in their respective
//! runtime crates and consume these contracts.

pub mod evaluator;
pub mod graph;
pub mod quantization;
pub mod table_builder;
