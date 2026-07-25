//! This module is the directory index for the `phase_graph` subsystem — the
//! canonical Prism name for the compact executable phase-graph IR that used
//! to live in a single `tinygrad_core.rs` file. It declares the submodules
//! and re-exports the public types so consumers continue to write
//! `prism_spatial_ir::UOp`, `prism_spatial_ir::TinyGraph`, etc., unchanged.
//! It does not own a domain authority of its own.

pub mod capture;
pub mod graph;
pub mod kernel_group;
pub mod kernel_op;
pub mod plan;
pub mod render;
pub mod scalar;
pub mod shape;
pub mod uop;

#[cfg(test)]
mod tests;

pub use crate::phase_graph::capture::{CaptureExecutor, CapturePlan, LoweredKernel, TinyJitCache};
pub use crate::phase_graph::graph::{GraphError, TinyGraph};
pub use crate::phase_graph::kernel_group::KernelGroup;
pub use crate::phase_graph::kernel_op::{BroadcastBinaryOperation, KernelOp, LoweringTarget};
pub use crate::phase_graph::plan::{
    BufferAllocation, ExecutionReceipt, MemoryPlan, ReplayPlan,
};
pub use crate::phase_graph::uop::{UOp, UOpId, UOpKind};
