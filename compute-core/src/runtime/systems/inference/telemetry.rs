//! TelemetryObservationSystem — ECS system that collects inference telemetry
//! during the maintenance stage.
//!
//! Registered as ID 201 in the scheduler; runs during `Stage::Maintenance`
//! and samples session telemetry, cache hit rates, token throughput, and
//! working-set memory pressure.  Writes to the diagnostic resource so that
//! other systems (e.g. watchdog, budget reaper) can react to pressure.

use lazy_static::lazy_static;

use crate::runtime::scheduling::access::{ComponentSet, ResourceSet};
use crate::runtime::scheduling::command::CommandWriter;
use crate::runtime::scheduling::metadata::{
    ErasedSystem, ExecutionClass, SerializationPolicy, Stage, SystemId,
    SystemMetadata, SystemResult, SystemSpec,
};
use crate::runtime::world::World;

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

/// ECS system that collects inference telemetry during maintenance ticks.
///
/// Samples per-session counters (tokens generated, cache hit rate, working
/// set pressure) and writes them into the diagnostics resource.  Runs as a
/// serial system at the default maintenance order so it observes all sessions
/// in a consistent state.
///
/// No component or resource bindings are required yet — the skeleton
/// placeholder runs as a no-op until the full telemetry pipeline is wired.
pub struct TelemetryObservationSystem {
    _private: (),
}

impl TelemetryObservationSystem {
    /// Create a new telemetry observation system.
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl SystemSpec for TelemetryObservationSystem {
    type Reads = ();
    type Writes = ();
    type ReadResources = ();
    type WriteResources = ();

    const NAME: &'static str = "telemetry_observation";
    const ID: SystemId = SystemId(201);
    const STAGE: Stage = Stage::Maintenance;
    const ORDER: i32 = 0;
    const EXECUTION_CLASS: ExecutionClass = ExecutionClass::Serial;
    const SERIALIZATION: SerializationPolicy = SerializationPolicy::ExplicitOnly;
}

// ---------------------------------------------------------------------------
// Static metadata (lazy_static per assignment requirements)
// ---------------------------------------------------------------------------

lazy_static! {
    static ref TELEMETRY_OBSERVATION_META: SystemMetadata = SystemMetadata {
        id: SystemId(201),
        name: "telemetry_observation",
        stage: Stage::Maintenance,
        reads: <() as ComponentSet>::mask().unwrap(),
        writes: <() as ComponentSet>::mask().unwrap(),
        reads_resources: <() as ResourceSet>::mask().unwrap(),
        writes_resources: <() as ResourceSet>::mask().unwrap(),
        after: &[],
        before: &[],
        order: 0,
        execution_class: ExecutionClass::Serial,
        serialization: SerializationPolicy::ExplicitOnly,
    };
}

impl ErasedSystem for TelemetryObservationSystem {
    fn metadata(&self) -> &SystemMetadata {
        &TELEMETRY_OBSERVATION_META
    }

    fn run(
        &mut self,
        _world: &mut World,
        _commands: &mut CommandWriter,
    ) -> SystemResult {
        // Placeholder — full implementation will:
        // 1. Query for all session entities
        // 2. Sample counters: tokens_generated, cache_hit_rate, working_set_pressure
        // 3. Write into WorkerDiagnosticsResource or a dedicated telemetry resource
        // 4. Emit warning commands when pressure exceeds thresholds
        SystemResult::ok()
    }
}

impl Default for TelemetryObservationSystem {
    fn default() -> Self {
        Self::new()
    }
}
