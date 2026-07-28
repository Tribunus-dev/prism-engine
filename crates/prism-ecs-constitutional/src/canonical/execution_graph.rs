//! ExecutionGraph — the execution-oriented graph produced from
//! `ModelIr` + `RepresentationPlan`. Authority: the
//! execution-oriented view.
//!
//! The execution graph is the data structure that ties
//! representation decisions (codec, scale structure) to
//! concrete operations (regions, lanes, memory plan) that the
//! kernel ABI can target. The data types live in
//! `prism_ecs_ir::cimage_types` and are re-exported here so the
//! compiler pipeline has a single, stable import path:
//! `prism_ecs_constitutional::canonical::ExecutionGraph`.

pub use prism_ecs_ir::cimage_types::{
    BufferValue, ExecutionEdge, ExecutionGraph, ExecutionLane, ExecutionOp, ExecutionOpKind,
    ExecutionRegion, FusionConstraints, GraphRegionId, MemoryPlan, RuntimeStatePlan,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execution_graph_identity_re_exports_compile() {
        // Type-level smoke test for the re-export surface.
        let _lane = ExecutionLane::Cpu;
    }
}
