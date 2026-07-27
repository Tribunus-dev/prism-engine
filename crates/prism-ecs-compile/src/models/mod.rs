//! Re-exported FP16 model data structures used at compile time.
//!
//! This submodule owns the canonical authority for compile-time data
//! structures that the model compiler produces for downstream consumers
//! (CPU-side token embedding tables today; future per-model CPU-side
//! structures should land here as additional submodules).

pub mod embedding;

pub use embedding::{f16_bits, TokenEmbedding, TokenEmbeddingError};
