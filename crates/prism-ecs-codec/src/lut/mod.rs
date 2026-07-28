//! Lookup-table (LUT) codec surface — palettized weight tables, FP16
//! math kernels, INT8 KV-cache quantization helpers, and model
//! graph descriptors.
//!
//! This module owns the canonical authority for the
//! backend-neutral LUT codec: the model architecture descriptor
//! that drives AOT compilation, the palettized matrix format
//! (codebook + packed 4-bit indices), the FP16 math kernels used
//! as a CPU fallback for inference, the symmetric INT8
//! per-token quantization used by the KV cache, and the
//! [`CompiledTensor`] data type that AOT compilation produces.
//!
//! All types here are backend-neutral. Hardware-specific
//! execution paths (Metal, ANE, MLX) and the AOT compile
//! orchestration (CImage I/O, GGUF parsing) live in the
//! engine's compile path and consume these contracts.

pub mod compile;
pub mod evaluator;
pub mod graph;
pub mod quantization;
pub mod table_builder;
