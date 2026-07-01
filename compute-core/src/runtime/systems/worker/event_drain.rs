//! WorkerEventDrainSystem — drains events from WorkerEventSource and reconciles
//! them against the ECS entity graph.
//!
//! Drains worker stdout pipes through WorkerEventSource (the ECS resource wrapping
//! the legacy IPC reader), parses each event envelope, and reconciles it against
//! the entity graph.  On completion the system delivers the response through the
//! WorkerResponseRegistry's oneshot channel if a pending request was registered.
//!
//! Runs during Stage::Maintenance (order 0) and processes at most 64 events
//! per tick.  For each event envelope it resolves the target entity by
//! matching `(request_id, worker_id, assignment_generation)`, validates the
//! event against the current lifecycle phase, and applies the appropriate
//! state transition.
//!
//! | Event kind    | Action                                                    |
//! |---------------|-----------------------------------------------------------|
/// | Heartbeat     | Update `WorkerHeartbeat`, reset miss counter               |
/// | Token         | Record in `WorkerStream`; transition `AwaitingFirstEvent`→`Streaming` |
/// | Completion    | Transition `Completing`; write `WorkerOutcome::success`; deliver via Registry |
/// | Failure       | Transition `Failed`; write `WorkerOutcome::failure`        |

use std::sync::OnceLock;

use crate::runtime::scheduling::command::CommandWriter;
use crate::runtime::scheduling::metadata::{
    ErasedSystem, ExecutionClass, SerializationPolicy, Stage, SystemId,
    SystemMetadata, SystemResult, SystemSpec,
};
use crate::runtime::world::{Entity, World};
use crate::runtime::components::{
    WorkerAssignment, WorkerHeartbeat, WorkerLifecycle, WorkerOutcome, WorkerRequest,
    WorkerStream, WorkerRequestPhase,
    worker_health::{TerminalStatus, WorkerErrorCategory},
    WORKER_EVENT_DRAIN_SYSTEM,
};
use crate::runtime::resources::{
    MonotonicClockResource, WorkerDiagnosticsResource, WorkerEventSource,
    WorkerResponseRegistry, EventKind,
};

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

pub struct WorkerEventDrainSystem {
    metadata: OnceLock<SystemMetadata>,
}

impl WorkerEventDrainSystem {
    pub fn new() -> Self {
        Self {
            metadata: OnceLock::new(),
        }
    }
}

impl Default for WorkerEventDrainSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemSpec for WorkerEventDrainSystem {
    type Reads = (WorkerAssignment, WorkerLifecycle, WorkerRequest, WorkerHeartbeat);
    type Writes = (WorkerLifecycle, WorkerStream, WorkerHeartbeat, WorkerOutcome);
    type ReadResources = (WorkerEventSource, WorkerResponseRegistry, MonotonicClockResource);
    type WriteResources = (WorkerDiagnosticsResource, WorkerResponseRegistry);

    const NAME: &'static str = "worker_event_drain";
    const ID: SystemId = WORKER_EVENT_DRAIN_SYSTEM;
    const STAGE: Stage = Stage::Maintenance;
    const ORDER: i32 = 0;
    const EXECUTION_CLASS: ExecutionClass = ExecutionClass::Serial;
    const SERIALIZATION: SerializationPolicy = SerializationPolicy::ExplicitOnly;
}

impl ErasedSystem for WorkerEventDrainSystem {
    fn metadata(&self) -> &SystemMetadata {
        self.metadata
            .get_or_init(|| <WorkerEventDrainSystem as SystemSpec>::metadata().expect("WorkerEventDrainSystem metadata"))
    }

    fn run(
        &mut self,
        world: &mut World,
        _commands: &mut CommandWriter,
    ) -> SystemResult {
        // -- 1. Drain event batch -------------------------------------------------
        let events = {
            let Some(event_source) = world.get_resource::<WorkerEventSource>() else {
                return SystemResult::ok();
            };
            let events = event_source.drain_batch(64);
            if events.is_empty() {
                return SystemResult::ok();
            }
            events
        };

        // Build a snapshot of all candidates: entities that have both a
        // WorkerAssignment and WorkerRequest.  We do this once rather than
        // searching per event.
        let mut candidates: Vec<(Entity, String, String, u32)> = Vec::new();
        for entity in world.iter_entities_with::<WorkerAssignment>() {
            if let (Some(assignment), Some(request)) = (
                world.get::<WorkerAssignment>(entity),
                world.get::<WorkerRequest>(entity),
            ) {
                candidates.push((
                    entity,
                    request.request_id.clone(),
                    assignment.worker_id.clone(),
                    assignment.generation,
                ));
            }
        }

        // -- 2. Process each event ------------------------------------------------
        for event in &events {
            // Resolve entity by (request_id, worker_id, generation).
            let entity = match candidates.iter().find(|(_, rid, wid, gen)| {
                rid == &event.request_id
                    && wid == &event.worker_id
                    && *gen == event.assignment_generation
            }) {
                Some((e, _, _, _)) => *e,
                None => {
                    world.get_resource::<WorkerDiagnosticsResource>()
                        .map(|d| d.record_stale_event_drop());
                    continue;
                }
            };

            let Some(lifecycle) = world.get::<WorkerLifecycle>(entity) else {
                world.get_resource::<WorkerDiagnosticsResource>()
                    .map(|d| d.record_stale_event_drop());
                continue;
            };

            let phase = lifecycle.phase;
            let valid = match &event.kind {
                EventKind::Heartbeat => true,
                EventKind::Token { .. } => {
                    matches!(
                        phase,
                        WorkerRequestPhase::AwaitingFirstEvent
                            | WorkerRequestPhase::Streaming
                    )
                }
                EventKind::Completion { .. } => {
                    matches!(
                        phase,
                        WorkerRequestPhase::AwaitingFirstEvent
                            | WorkerRequestPhase::Streaming
                            | WorkerRequestPhase::Completing
                    )
                }
                EventKind::Failure { .. } => phase.is_active(),
                EventKind::Progress { .. } => {
                    matches!(
                        phase,
                        WorkerRequestPhase::AwaitingFirstEvent
                            | WorkerRequestPhase::Streaming
                    )
                }
            };

            if !valid {
                world.get_resource::<WorkerDiagnosticsResource>()
                    .map(|d| d.record_lifecycle_rejection());
                continue;
            }

            // Dispatch by event kind.
            match &event.kind {
                EventKind::Heartbeat => {
                    if let Some(hb) = world.get_mut::<WorkerHeartbeat>(entity) {
                        hb.mark_received();
                    }
                }

                EventKind::Token { token_id, bytes } => {
                    if let Some(stream) = world.get_mut::<WorkerStream>(entity) {
                        stream.record_output(Some(*token_id), bytes.len() as u64);
                    }
                    // First token: AwaitingFirstEvent -> Streaming.
                    if phase == WorkerRequestPhase::AwaitingFirstEvent {
                        if let Some(lc) = world.get_mut::<WorkerLifecycle>(entity) {
                            let _ = lc.transition_to(WorkerRequestPhase::Streaming);
                        }
                    }
                }

                EventKind::Completion { .. } => {
                    // Transition to Completing if not already there.
                    if phase != WorkerRequestPhase::Completing {
                        if let Some(lc) = world.get_mut::<WorkerLifecycle>(entity) {
                            let _ = lc.transition_to(WorkerRequestPhase::Completing);
                        }
                    }
                    let outcome = WorkerOutcome::new(
                        TerminalStatus::Success,
                        WorkerErrorCategory::None,
                        None,
                        event.assignment_generation,
                    );
                    world.insert(entity, outcome);

                    // Deliver the response through the oneshot channel if a
                    // pending request was registered.
                    if let Some(registry) = world.get_resource::<WorkerResponseRegistry>() {
                        let _ = registry.deliver_response(
                            &event.request_id,
                            format!("ok:{}:{} tokens",
                                event.assignment_generation,
                                ""),
                        );
                    }
                }

                EventKind::Failure { category: _, code } => {
                    let outcome = WorkerOutcome::new(
                        TerminalStatus::Failed,
                        WorkerErrorCategory::ProcessCrash,
                        *code,
                        event.assignment_generation,
                    );
                    world.insert(entity, outcome);
                    if let Some(lc) = world.get_mut::<WorkerLifecycle>(entity) {
                        let _ = lc.transition_to(WorkerRequestPhase::Failed);
                    }
                }

                EventKind::Progress { .. } => {
                    // Informational — no structural mutation.
                }
            }
        }

        SystemResult::ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::world::World;

    fn setup_world() -> World {
        let mut world = World::with_capacity(64);
        world.register_component::<WorkerAssignment>();
        world.register_component::<WorkerLifecycle>();
        world.register_component::<WorkerRequest>();
        world.register_component::<WorkerHeartbeat>();
        world.register_component::<WorkerStream>();
        world.register_component::<WorkerOutcome>();
        world.insert_resource(WorkerEventSource::new());
        world.insert_resource(WorkerDiagnosticsResource::new());
        world.insert_resource(WorkerResponseRegistry::new());
        world.insert_resource(MonotonicClockResource::new());
        world
    }

    #[test]
    fn no_events_is_ok() {
        let mut world = setup_world();
        let mut system = WorkerEventDrainSystem::new();
        let mut buffer = Vec::new();
        let mut cw = CommandWriter::new(&mut buffer, Stage::Maintenance, WORKER_EVENT_DRAIN_SYSTEM);
        let result = system.run(&mut world, &mut cw);
        assert!(matches!(result, SystemResult::Ok));
    }

    #[test]
    fn empty_world_is_ok() {
        let mut world = World::with_capacity(8);
        world.insert_resource(WorkerEventSource::new());
        world.insert_resource(WorkerDiagnosticsResource::new());
        world.insert_resource(WorkerResponseRegistry::new());
        world.insert_resource(MonotonicClockResource::new());
        let mut system = WorkerEventDrainSystem::new();
        let mut buffer = Vec::new();
        let mut cw = CommandWriter::new(&mut buffer, Stage::Maintenance, WORKER_EVENT_DRAIN_SYSTEM);
        let result = system.run(&mut world, &mut cw);
        assert!(matches!(result, SystemResult::Ok));
    }

    #[test]
    fn metadata_is_valid() {
        let metadata = <WorkerEventDrainSystem as SystemSpec>::metadata().expect("metadata");
        assert_eq!(metadata.name, "worker_event_drain");
        assert_eq!(metadata.id, WORKER_EVENT_DRAIN_SYSTEM);
        assert_eq!(metadata.stage, Stage::Maintenance);
        assert_eq!(metadata.order, 0);
    }

    #[test]
    fn system_id_constant() {
        assert_eq!(WORKER_EVENT_DRAIN_SYSTEM, SystemId(101));
    }
}
