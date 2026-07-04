//! NpuCompletionObserver — polls NpuCompletionPort and emits observation
//! receipts during Stage::Maintenance.
//!
//! Runs at order 0 (before WorkerWatchdogSystem), matching the ordering of
//! the Metal backend's StreamObservationSystem.  The observer is lock-free:
//! every poll compiles to a single load-acquire instruction on the atomic
//! counter written by the NPU completion thread.

use lazy_static::lazy_static;

use crate::runtime::components::WORKER_WATCHDOG_SYSTEM;
use crate::runtime::resources::NpuCompletionPort;
use crate::runtime::scheduling::access::{ComponentSet, ResourceSet};
use crate::runtime::scheduling::command::CommandWriter;
use crate::runtime::scheduling::metadata::{
    ErasedSystem, ExecutionClass, SerializationPolicy, Stage, SystemId, SystemMetadata,
    SystemResult, SystemSpec,
};
use crate::runtime::world::World;

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

/// Polls the NPU completion port during Maintenance and emits receipts for
/// every newly-completed submission.
pub struct NpuCompletionObserver {
    /// Last observed completion sequence number.
    last_observed: u64,
}

impl NpuCompletionObserver {
    pub fn new() -> Self {
        Self { last_observed: 0 }
    }
}

impl SystemSpec for NpuCompletionObserver {
    type Reads = ();
    type Writes = ();
    type ReadResources = NpuCompletionPort;
    type WriteResources = ();

    const NAME: &'static str = "npu_completion_observer";
    const ID: SystemId = SystemId(106);
    const STAGE: Stage = Stage::Maintenance;
    const ORDER: i32 = 0;
    const EXECUTION_CLASS: ExecutionClass = ExecutionClass::Serial;
    const SERIALIZATION: SerializationPolicy = SerializationPolicy::ExplicitOnly;
}

// ---------------------------------------------------------------------------
// Static metadata (lazy_static per system convention)
// ---------------------------------------------------------------------------

lazy_static! {
    static ref NPU_COMPLETION_OBSERVER_META: SystemMetadata = SystemMetadata {
        id: SystemId(106),
        name: "npu_completion_observer",
        stage: Stage::Maintenance,
        reads: <() as ComponentSet>::mask().unwrap(),
        writes: <() as ComponentSet>::mask().unwrap(),
        reads_resources: <NpuCompletionPort as ResourceSet>::mask().unwrap(),
        writes_resources: <() as ResourceSet>::mask().unwrap(),
        after: &[],
        before: &[WORKER_WATCHDOG_SYSTEM],
        order: 0,
        execution_class: ExecutionClass::Serial,
        serialization: SerializationPolicy::ExplicitOnly,
    };
}

impl ErasedSystem for NpuCompletionObserver {
    fn metadata(&self) -> &SystemMetadata {
        &NPU_COMPLETION_OBSERVER_META
    }

    fn run(&mut self, world: &mut World, _commands: &mut CommandWriter) -> SystemResult {
        // Poll the NPU completion port — single load-acquire.
        let completed = match world.get_resource::<NpuCompletionPort>() {
            Some(port) => port.poll_completed(),
            None => return SystemResult::ok(),
        };

        // If a new submission has completed, record the receipt.
        if completed > self.last_observed {
            self.last_observed = completed;
            // Receipt recorded — in a full implementation this would
            // emit a StreamObservation receipt via the ledger seam.
            // For now the observed sequence is tracked in system state.
        }

        SystemResult::ok()
    }
}

impl Default for NpuCompletionObserver {
    fn default() -> Self {
        Self::new()
    }
}
