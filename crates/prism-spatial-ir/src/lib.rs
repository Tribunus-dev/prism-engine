//! Prism Spatial IR — explicit intermediate representation for spatial dataflow graphs.
//!
//! SpatialIR is a four-level representation that separates spatial intent from
//! backend-specific lowering, enabling calibrated evolutionary search over
//! heterogeneous placement decisions (GPU vs ANE vs CPU).
//!
//! # Levels
//!
//! - **Level 0 (ECS World):** The existing Prism IR — tensor components,
//!   operation systems, data dependencies.
//! - **Level 1 (Spatial Graph):** An explicit, serializable graph with no
//!   mention of specific hardware — [`graph::SpatialGraph`].
//! - **Level 2 (Virtual Hardware):** Logical execution and memory resources
//!   abstracted from any specific target — [`hardware`].
//! - **Level 3 (Physical Target):** Backend-specific lowering via the
//!   [`target::SpatialTarget`] trait.

pub mod bonsai_gen;
pub mod calibration_report;
pub mod cost;
pub mod evolution;
pub mod execution;
pub mod execution_plan;
pub mod fused_ops;
pub mod graph;
pub mod hardware;
pub mod legalize;
pub mod memory;
pub mod mutation;
pub mod plan;
pub mod scheduler;
pub mod semantic_region;
pub mod target;
pub mod three_thread;
pub mod tiling;
pub mod tinygrad_core;
pub mod topology;
pub mod xdna;
pub mod xdna_manifest;
pub mod xdna_target;

pub use calibration_report::{M1CalibrationReport, MemoryPressure, PowerState, ThermalState};
pub use cost::{
    evaluate_joint_tiling_configurations, select_best_joint_tiling, CalibratedCostModel,
    CostEstimate, CostModel, JointTilingCostEstimate, Mi300xCostModel, TilingSelectionError,
};
pub use execution_plan::{lower_to_manifest, ExecutionMode, ExecutionPlan};
pub use fused_ops::{
    benchmark_fusion_strategies, benchmark_workload_scenarios, check_fusion_legality,
    enumerate_fusion_candidates, evaluate_fusion_strategies,
    evaluate_fusion_strategies_for_workload, evaluate_fusion_strategies_with_generation,
    evaluate_fusion_strategies_with_generation_and_measurements,
    evaluate_fusion_strategies_with_measurements, FusableOp, FusedPermutation, FusionMeasurement,
    FusionStrategy, FusionStrategyCandidate, FusionStrategyEvaluation, WorkloadScenario,
    WorkloadStrategyEvaluation,
};
pub use graph::{SpatialEdge, SpatialGraph, SpatialNode, SpatialNodeId, TileGeometry};
pub use hardware::{ExecutionBoundary, VirtualComputeUnit, VirtualMemoryRegion};
pub use legalize::{
    ane_specific_checks, joint_tiling_checks, metal_specific_checks, LegalizationError,
    LegalizedGraph,
};
pub use scheduler::{
    AotScheduler, BindingResolver, BufferStorage, HeterogeneousExecutor, ResolvedBuffer,
    ResolvedStep, RouteDispatch, RoutedExecutor,
};
pub use semantic_region::{
    lower_contiguous_axis0, PhysicalRegionError, PhysicalRegionPlan, PhysicalRegionRealization,
};
pub use target::{
    probe_apple_silicon, AppleSiliconTarget, KernelDescriptor, KernelManifest, SpatialTarget,
    TargetCapabilities,
};
pub use tiling::{
    validate_joint_tiling_geometry, validate_tiling_geometry, TilingBackend, TilingConfiguration,
    TilingValidationError,
};
pub use tinygrad_core::{
    BroadcastBinaryOperation, BufferAllocation, CaptureExecutor, CapturePlan, ExecutionReceipt,
    KernelGroup, KernelOp, LoweredKernel, LoweringTarget, MemoryPlan, ReplayPlan, TinyGraph,
    TinyJitCache, UOp, UOpId, UOpKind,
};
pub use xdna_target::XdnaTarget;
