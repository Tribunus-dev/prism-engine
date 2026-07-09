//! BitNet b1.58 2B4T native ternary weight importer.
//!
//! This module provides a native importer for BitNet b1.58 2B4T models,
//! ingesting already-ternary {-1, 0, +1} weights from HuggingFace safetensors
//! into Prism's cimage format.
//!
//! The importer extracts `BitLinear` weight tensors (each already ternary),
//! packs them into 2-bit codes via `crate::ternary::pack::pack_ternary_codes`,
//! and emits cimage shards through the phased builder system.

pub mod importer;
pub mod phases;

#[cfg(test)]
pub mod tests;
