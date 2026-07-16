//! Prism Intel NPU runtime — code generation and dispatch for Intel NPU
//! (Meteor Lake+) accelerators.
//!
//! Follows the `prism-metal-runtime` pattern: compile source → binary →
//! dispatch → evidence.

pub mod codegen;
pub mod compiler;
pub mod dispatch;

pub use codegen::{lower_matmul_to_intel_npu, lower_to_intel_npu, IntelNpuLowerError};
pub use compiler::compile_intel_npu;
pub use dispatch::dispatch_intel_npu;
