//! ExecutionGraph — the execution-oriented graph produced from ModelIr + RepresentationPlan.
//!
//! Re-exported from `prism-ecs-ir` (phase 2 of compute-core dependency removal).

pub use prism_ecs_ir::cimage_types::{
    BufferValue, ExecutionEdge, ExecutionGraph, ExecutionLane, ExecutionOp, ExecutionOpKind,
    ExecutionRegion, FusionConstraints, GraphRegionId, MemoryPlan, RuntimeStatePlan,
};
