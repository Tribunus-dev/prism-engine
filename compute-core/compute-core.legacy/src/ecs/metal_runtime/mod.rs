//! Metal runtime — re-exported from the standalone `prism-metal-runtime` crate.
//! Gated behind `metal-dispatch` feature + macOS.

#[cfg(feature = "metal-dispatch")]
pub use prism_metal_runtime::*;
