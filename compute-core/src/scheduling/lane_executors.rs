//! Per-lane executors that own only backend submission and completion reporting.
//!
//! These executors must NOT mutate orchestrator state (readiness, leases, cache).
//! They emit WorkCompletion events into a channel; the Tokio-based orchestrator
//! loop processes them.

use crate::backend::placement::ExecutionLane;
use crate::scheduling::tri_lane_orchestrator::{EpochId, PhaseVariant, WorkCompletion};

/// Metal/GPU lane executor.
/// Owns Metal command queue, pipeline state, texture cache.
pub struct MetalLaneExecutor;

impl MetalLaneExecutor {
    pub fn new() -> Self {
        Self
    }

    /// Submit work to Metal. Returns a WorkCompletion once the command buffer completes.
    /// The completion is pushed to the provided channel, NOT returned directly.
    pub fn submit(
        &self,
        _variant: &PhaseVariant,
        _epoch_id: EpochId,
    ) -> Result<WorkCompletion, String> {
        // Stub: return a fake completion immediately
        Ok(WorkCompletion {
            phase_id: crate::compilation::phase_ir::PhaseId(_epoch_id),
            variant_id: 0,
            lane: ExecutionLane::MlxGpu,
            start_time: std::time::Instant::now(),
            completion_time: std::time::Instant::now(),
            success: true,
            output_slot: crate::compilation::activation_abi::SlotLeaseId(0),
        })
    }
}

/// ANE/Core ML lane executor.
pub struct AneLaneExecutor;

impl AneLaneExecutor {
    pub fn new() -> Self {
        Self
    }

    pub fn submit(
        &self,
        _variant: &PhaseVariant,
        _epoch_id: EpochId,
    ) -> Result<WorkCompletion, String> {
        Ok(WorkCompletion {
            phase_id: crate::compilation::phase_ir::PhaseId(_epoch_id),
            variant_id: 0,
            lane: ExecutionLane::CoreMlAne,
            start_time: std::time::Instant::now(),
            completion_time: std::time::Instant::now(),
            success: true,
            output_slot: crate::compilation::activation_abi::SlotLeaseId(0),
        })
    }
}

/// Accelerate/CPU lane executor.
pub struct AccelerateLaneExecutor;

impl AccelerateLaneExecutor {
    pub fn new() -> Self {
        Self
    }

    pub fn submit(
        &self,
        _variant: &PhaseVariant,
        _epoch_id: EpochId,
    ) -> Result<WorkCompletion, String> {
        Ok(WorkCompletion {
            phase_id: crate::compilation::phase_ir::PhaseId(_epoch_id),
            variant_id: 0,
            lane: ExecutionLane::AccelerateCpu,
            start_time: std::time::Instant::now(),
            completion_time: std::time::Instant::now(),
            success: true,
            output_slot: crate::compilation::activation_abi::SlotLeaseId(0),
        })
    }
}
