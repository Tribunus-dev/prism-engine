//! Compiler-emitted concurrency plan.
//!
//! This module owns the **parallelism contract** for a heterogeneous
//! image: which phases are independently ready, which groups may run
//! in parallel, which edges require serialization, what lane
//! capacity is required, and hints about expected overlap.
//!
//! The runtime may provide more capacity than declared, but may not
/// run a concurrency-required image below its declared safe minimum
/// without downgrading to a serial plan.

use serde::{Deserialize, Serialize};

use super::backend_plan::CostConfidence;
use super::phase_ir::PhaseId;
use super::resource_plan::SlotId;
use super::shared::ExecutionLane;

/// The compiler-emitted concurrency plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompiledConcurrencyPlan {
    pub ready_sets: Vec<ReadySetTemplate>,
    pub parallel_groups: Vec<ParallelGroup>,
    pub serialization_edges: Vec<SerializationEdge>,
    pub lane_caps: LaneCapacityRequirements,
    pub overlap_hints: Vec<OverlapHint>,
}

/// Template for a ready set — phases that are independently
/// dispatchable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReadySetTemplate {
    pub ready_set_id: ReadySetId,
    pub phases: Vec<PhaseId>,
}

/// A parallel group — phases that may be dispatched before awaiting
/// any member.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParallelGroup {
    pub group_id: ParallelGroupId,
    pub phases: Vec<PhaseId>,
    pub required_distinct_slots: Vec<SlotId>,
    pub allowed_lanes: Vec<ExecutionLane>,
    pub expected_overlap_kind: OverlapKind,
}

/// How overlap is expected to manifest.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum OverlapKind {
    /// True concurrent execution across lanes.
    ConcurrentLanes,
    /// Pipelined execution within a single lane.
    PipelineWithinLane,
    /// Interleaved execution across sequences.
    InterleavedSequences,
    /// Sequential — no overlap possible.
    Sequential,
}

/// A serialization constraint between phases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializationEdge {
    pub from: PhaseId,
    pub to: PhaseId,
    pub reason: SerializationReason,
}

/// Why two phases must be serialized.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub enum SerializationReason {
    DataDependency,
    MutableSlot,
    LaneCapacity,
    Barrier,
    AdmissionGate,
    NumericalConstraint,
}

/// Compiler-emitted lane capacity requirements.
///
/// The runtime may provide more capacity, but may not run a
/// concurrency-required image below its declared safe minimum
/// without downgrading to a serial plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaneCapacityRequirements {
    pub metal_in_flight_min: u32,
    pub ane_in_flight_min: u32,
    pub accelerate_workers_min: u32,
    pub iosurface_ring_depth_min: u32,
    pub completion_queue_min: u32,
}

impl Default for LaneCapacityRequirements {
    fn default() -> Self {
        Self {
            metal_in_flight_min: 1,
            ane_in_flight_min: 1,
            accelerate_workers_min: 1,
            iosurface_ring_depth_min: 2,
            completion_queue_min: 1,
        }
    }
}

/// A hint about expected overlap between specific phases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverlapHint {
    pub phase_a: PhaseId,
    pub phase_b: PhaseId,
    pub expected_overlap_kind: OverlapKind,
    pub confidence: CostConfidence,
}

pub type ReadySetId = u64;
pub type ParallelGroupId = u64;
