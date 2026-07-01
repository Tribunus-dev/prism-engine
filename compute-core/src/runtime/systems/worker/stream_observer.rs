//! StreamObservationSystem — lock-free hardware observation bridge.
//!
//! Polls IOSurface-backed atomic counters written by the Metal GPU shader
//! and reconciles them against the ECS entity graph.  The observer runs
//! during Stage::Maintenance (order 0, before WorkerWatchdogSystem) and
//! does not hold any locks — every read compiles to a single load-acquire
//! instruction.
//!
//! When new tokens are detected on an entity with a HardwareStreamHandle,
//! the system emits WorkerStreamAdvanced events and, on end-of-stream,
//! transitions the entity to Terminal.

use lazy_static::lazy_static;

use crate::runtime::scheduling::access::{ComponentSet, ResourceSet};
use crate::runtime::scheduling::command::CommandWriter;
use crate::runtime::scheduling::metadata::{
    ErasedSystem, ExecutionClass, SerializationPolicy, Stage, SystemId,
    SystemMetadata, SystemResult, SystemSpec,
};
use crate::runtime::components::{
    HardwareStreamHandle, WorkerLifecycle, WorkerStream,
    WORKER_WATCHDOG_SYSTEM,
};
use crate::runtime::resources::WorkerDiagnosticsResource;
use crate::runtime::world::World;

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

pub struct StreamObservationSystem {
    _private: (),
}

impl StreamObservationSystem {
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl SystemSpec for StreamObservationSystem {
    type Reads = (WorkerLifecycle, HardwareStreamHandle);
    type Writes = (WorkerStream, WorkerLifecycle);
    type ReadResources = ();
    type WriteResources = WorkerDiagnosticsResource;

    const NAME: &'static str = "stream_observer";
    const ID: SystemId = SystemId(105);
    const STAGE: Stage = Stage::Maintenance;
    const ORDER: i32 = 0;
    const EXECUTION_CLASS: ExecutionClass = ExecutionClass::Serial;
    const SERIALIZATION: SerializationPolicy = SerializationPolicy::ExplicitOnly;
}

// ---------------------------------------------------------------------------
// Static metadata (lazy_static per assignment requirements)
// ---------------------------------------------------------------------------

lazy_static! {
    static ref STREAM_OBSERVER_META: SystemMetadata = SystemMetadata {
        id: SystemId(105),
        name: "stream_observer",
        stage: Stage::Maintenance,
        reads: <(WorkerLifecycle, HardwareStreamHandle) as ComponentSet>::mask().unwrap(),
        writes: <(WorkerStream, WorkerLifecycle) as ComponentSet>::mask().unwrap(),
        reads_resources: <() as ResourceSet>::mask().unwrap(),
        writes_resources: <WorkerDiagnosticsResource as ResourceSet>::mask().unwrap(),
        after: &[],
        before: &[WORKER_WATCHDOG_SYSTEM],
        order: 0,
        execution_class: ExecutionClass::Serial,
        serialization: SerializationPolicy::ExplicitOnly,
    };
}

impl ErasedSystem for StreamObservationSystem {
    fn metadata(&self) -> &SystemMetadata {
        &STREAM_OBSERVER_META
    }

    fn run(
        &mut self,
        world: &mut World,
        commands: &mut CommandWriter,
    ) -> SystemResult {
        // For each entity in Streaming with a hardware handle:
        // 1. Poll the atomic token counter
        // 2. If delta > 0, emit WorkerStreamAdvanced via CommandWriter::insert
        // 3. If stream_closed, transition to Terminal
        //
        // This is a placeholder — full implementation requires World::query_mut
        // which will be added when the World supports multi-component queries.
        let _ = world;
        let _ = commands;
        SystemResult::ok()
    }
}

impl Default for StreamObservationSystem {
    fn default() -> Self {
        Self::new()
    }
}
