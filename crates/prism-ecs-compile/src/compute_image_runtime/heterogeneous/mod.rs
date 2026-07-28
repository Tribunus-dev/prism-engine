//! Heterogeneous execution image — pure data types for multi-lane
//! execution images (Metal / Core ML / Accelerate / etc.).
//!
//! The engine-coupled implementations (lane-specific executors,
//! cross-lane dispatch, MIL program construction) stay at
//! `compute-core/src/ecs/compute_image/legacy_compute_image_runtime/heterogeneous/`.

pub mod types;

pub use types::{
    CompileCostEstimate, CostConfidence, DependencyClass, HeterogeneousExecutionImage,
    LaneCapability, ModelIdentity, NumericalContract, OperatorId, PhaseCapabilityMatrix,
    PhaseEdge, PhaseEdgeKind, PhaseGraph, PhaseId, PhaseKind, PhaseNode, PhaseValue,
    ShapeContract, UnsupportedReason, ValueId,
};
