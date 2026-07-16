//! Prism AMD XDNA NPU runtime — codegen, compilation, and dispatch
//! for AMD Ryzen AI NPUs (XDNA architecture with AIE2/AIE2P engines).
//!
//! Follows the `prism-metal-runtime` pattern: compile source → binary →
//! dispatch → evidence.

pub mod codegen;
pub mod compiler;
pub mod dispatch;

pub use codegen::{lower_matmul_to_amd_npu, lower_to_amd_npu, AmdNpuLowerError};
pub use compiler::compile_amd_npu;
pub use dispatch::dispatch_amd_npu;
