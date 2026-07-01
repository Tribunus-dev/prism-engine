//! NPU Observer System — non-blocking NPU completion polling.
//!
//! Runs during the Maintenance stage, strictly ordered after
//! WorkerEventDrainSystem.  Iterates entities with an active NPU execution
//! state and polls the hardware via lock-free FFI.  On completion, transitions
//! the worker lifecycle and records a terminal outcome.
//!
//! This mirrors the StreamObserver pattern used by the Metal backend, adapted
//! for the synchronous (submit-and-wait) NPU execution model — polling is
//! still non-blocking, but the semantic is completion detection rather than
//! incremental token observation.

use lazy_static::lazy_static;

use crate::runtime::components::{
    worker_health::{TerminalStatus, WorkerErrorCategory},
    WorkerLifecycle, WorkerOutcome, WorkerRequestPhase, WORKER_EVENT_DRAIN_SYSTEM,
};
use crate::runtime::scheduling::access::{ComponentSet, ResourceSet};
use crate::runtime::scheduling::command::CommandWriter;
use crate::runtime::scheduling::metadata::{
    ErasedSystem, ExecutionClass, SerializationPolicy, Stage, SystemId, SystemMetadata,
    SystemResult, SystemSpec,
};
use crate::runtime::systems::npu::submitter::NpuExecutionState;
use crate::runtime::world::World;

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

pub struct NpuObserverSystem {
    _private: (),
}

impl NpuObserverSystem {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl SystemSpec for NpuObserverSystem {
    type Reads = (WorkerLifecycle, NpuExecutionState);
    type Writes = (WorkerLifecycle, WorkerOutcome);
    type ReadResources = ();
    type WriteResources = ();

    const NAME: &'static str = "npu_observer";
    const ID: SystemId = SystemId(202);
    const STAGE: Stage = Stage::Maintenance;
    const ORDER: i32 = 5;
    const EXECUTION_CLASS: ExecutionClass = ExecutionClass::Serial;
    const SERIALIZATION: SerializationPolicy = SerializationPolicy::ExplicitOnly;
}

// ---------------------------------------------------------------------------
// Static metadata (lazy_static per system convention)
// ---------------------------------------------------------------------------

lazy_static! {
    static ref NPU_OBSERVER_META: SystemMetadata = SystemMetadata {
        id: SystemId(202),
        name: "npu_observer",
        stage: Stage::Maintenance,
        reads: <(WorkerLifecycle, NpuExecutionState) as ComponentSet>::mask().unwrap(),
        writes: <(WorkerLifecycle, WorkerOutcome) as ComponentSet>::mask().unwrap(),
        reads_resources: <() as ResourceSet>::mask().unwrap(),
        writes_resources: <() as ResourceSet>::mask().unwrap(),
        after: &[WORKER_EVENT_DRAIN_SYSTEM],
        before: &[],
        order: 5,
        execution_class: ExecutionClass::Serial,
        serialization: SerializationPolicy::ExplicitOnly,
    };
}

impl ErasedSystem for NpuObserverSystem {
    fn metadata(&self) -> &SystemMetadata {
        &NPU_OBSERVER_META
    }

    fn run(&mut self, world: &mut World, _commands: &mut CommandWriter) -> SystemResult {
        // Collect entity ids first to avoid borrow conflicts with get_mut below.
        let candidate_entities: Vec<_> = world
            .iter_entities_with::<NpuExecutionState>()
            .filter(|entity| {
                world
                    .get::<WorkerLifecycle>(*entity)
                    .map(|lc| lc.phase == WorkerRequestPhase::Streaming)
                    .unwrap_or(false)
            })
            .collect();

        for entity in candidate_entities {
            let (target, session, sid) = match world.get::<NpuExecutionState>(entity) {
                Some(s) => (s.target, s.session, s.submission_id),
                None => continue,
            };

            // Non-blocking poll of the NPU completion flag.
            let is_complete = unsafe {
                crate::backend::npu::ffi::npu_poll_completion(
                    target,
                    session,
                    sid,
                )
            };

            if is_complete == 0 {
                continue;
            }

            // Transition to Completing — inference data is ready.
            if let Some(lc) = world.get_mut::<WorkerLifecycle>(entity) {
                let _ = lc.transition_to(WorkerRequestPhase::Completing);
            }

            // Record terminal outcome.
            let outcome = WorkerOutcome::new(
                TerminalStatus::Success,
                WorkerErrorCategory::None,
                None,
                sid as u32,
            );
            world.insert(entity, outcome);
        }

        SystemResult::ok()
    }
}

impl Default for NpuObserverSystem {
    fn default() -> Self {
        Self::new()
    }
}
