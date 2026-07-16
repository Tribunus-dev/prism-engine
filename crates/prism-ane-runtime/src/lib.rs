//! Prism ANE runtime — MIL codegen, CoreML compilation, and ANE dispatch.
//!
//! This crate provides three capabilities following the `prism-metal-runtime`
//! pattern (compile source → binary → dispatch → evidence):
//!
//! 1. **Codegen** (`codegen`) — lowers ECS-native IR operations (e.g.
//!    `linalg.matmul`) into MIL-like textual pseudo-code describing a neural
//!    network program for the Apple Neural Engine.
//! 2. **Compiler** (`compiler`) — invokes CoreML to compile a MIL program into
//!    a loadable model package. Feature-gated behind `coreml`.
//! 3. **Dispatch** (`dispatch`) — loads and runs a compiled ANE model with
//!    synchronous inference and timing evidence collection.
//!
//! The codegen module was migrated from `prism-ecs-ir::backend_ane`.

pub mod codegen;
pub mod compiler;
pub mod dispatch;

pub use codegen::{lower_matmul_to_ane, lower_to_ane, AneLowerError};
pub use compiler::{compile_mil, AneBinary};
pub use dispatch::{dispatch, is_ane_available, TensorDescriptor, TimingEvidence};
