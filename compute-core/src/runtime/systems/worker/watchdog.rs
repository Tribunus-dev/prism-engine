//! WorkerWatchdogSystem — liveness monitoring for in-flight worker requests.
//!
//! Runs during Stage::Maintenance (order 1, after WorkerEventDrainSystem).
//! Inspects entities in active phases (Dispatching, AwaitingFirstEvent,
//! Streaming, CancelRequested) and promotes workers whose heartbeat or
//! first-event deadline has been exceeded to a missed-heartbeat counter.
//! After `max_consecutive_misses` consecutive misses, the watchdog calls
//! WorkerPoolResource to kill/recover the worker and transitions the entity
//! to Abandoned with an appropriate WorkerOutcome.  Any pending request in
//! WorkerResponseRegistry is also cleaned up.

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use crate::runtime::scheduling::command::CommandWriter;
use crate::runtime::scheduling::metadata::{
    ErasedSystem, ExecutionClass, SerializationPolicy, Stage, SystemId,
    SystemMetadata, SystemResult, SystemSpec,
};
use crate::runtime::world::World;
use crate::runtime::components::{
    WorkerAssignment, WorkerHeartbeat, WorkerLifecycle, WorkerOutcome, WorkerRequest,
    WorkerRequestPhase,
    worker_health::{TerminalStatus, WorkerErrorCategory},
    WORKER_EVENT_DRAIN_SYSTEM, WORKER_WATCHDOG_SYSTEM,
};
use crate::runtime::resources::{
    MonotonicClockResource, WorkerDiagnosticsResource, WorkerPoolResource,
    WorkerResponseRegistry,
};

// ---------------------------------------------------------------------------
// Watchdog policy
// ---------------------------------------------------------------------------

/// Tuning parameters for the worker watchdog escalation policy.
#[derive(Debug, Clone)]
pub struct WorkerWatchdogPolicy {
    /// Maximum time to wait for the first event after dispatch.
    pub first_event_timeout: Duration,
    /// Maximum time between heartbeats before a miss is recorded.
    pub heartbeat_timeout: Duration,
    /// Consecutive misses before the watchdog triggers recovery.
    pub max_consecutive_misses: u32,
}

impl Default for WorkerWatchdogPolicy {
    fn default() -> Self {
        Self {
            first_event_timeout: Duration::from_secs(30),
            heartbeat_timeout: Duration::from_secs(10),
            max_consecutive_misses: 3,
        }
    }
}

// ---------------------------------------------------------------------------
// System
// ---------------------------------------------------------------------------

pub struct WorkerWatchdogSystem {
    metadata: OnceLock<SystemMetadata>,
    /// Watchdog policy — tunable parameters for escalation thresholds.
    pub policy: WorkerWatchdogPolicy,
}

impl WorkerWatchdogSystem {
    pub fn new() -> Self {
        Self {
            metadata: OnceLock::new(),
            policy: WorkerWatchdogPolicy::default(),
        }
    }

    /// Create a watchdog with a custom policy.
    pub fn with_policy(policy: WorkerWatchdogPolicy) -> Self {
        Self {
            metadata: OnceLock::new(),
            policy,
        }
    }
}

impl Default for WorkerWatchdogSystem {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemSpec for WorkerWatchdogSystem {
    type Reads = (WorkerAssignment, WorkerLifecycle, WorkerHeartbeat, WorkerRequest);
    type Writes = (WorkerLifecycle, WorkerOutcome);
    type ReadResources = MonotonicClockResource;
    type WriteResources = (WorkerPoolResource, WorkerDiagnosticsResource, WorkerResponseRegistry);

    const NAME: &'static str = "worker_watchdog";
    const ID: SystemId = WORKER_WATCHDOG_SYSTEM;
    const STAGE: Stage = Stage::Maintenance;
    const ORDER: i32 = 1;
    const AFTER: &'static [SystemId] = &[WORKER_EVENT_DRAIN_SYSTEM];
    const EXECUTION_CLASS: ExecutionClass = ExecutionClass::Serial;
    const SERIALIZATION: SerializationPolicy = SerializationPolicy::ExplicitOnly;
}

impl ErasedSystem for WorkerWatchdogSystem {
    fn metadata(&self) -> &SystemMetadata {
        self.metadata
            .get_or_init(|| <Self as SystemSpec>::metadata().expect("WorkerWatchdogSystem metadata"))
    }

    fn run(
        &mut self,
        world: &mut World,
        _commands: &mut CommandWriter,
    ) -> SystemResult {
        // Reference time for all elapsed calculations.
        // Snapshot Copy values to avoid holding Ref borrows across mutable operations.
        let now = {
            let Some(clock) = world.get_resource::<MonotonicClockResource>() else {
                return SystemResult::ok();
            };
            clock.now()
        };

        // Collect entities that are in an active, non-terminal phase.
        // We snapshot first to avoid borrow conflicts when mutating components
        // during iteration.
        let actives: Vec<_> = world
            .iter_entities_with::<WorkerLifecycle>()
            .filter(|entity| {
                world
                    .get::<WorkerLifecycle>(*entity)
                    .map_or(false, |lc| lc.phase.is_active())
            })
            .collect();

        for entity in actives {
            let phase = world
                .get::<WorkerLifecycle>(entity)
                .map(|lc| lc.phase);

            let Some(phase) = phase else {
                continue;
            };

            // Determine the timeout and time-since-last-activity for this entity.
            let (since_last_event, timeout) =
                match compute_idle_time(world, entity, phase, now) {
                    Some(v) => v,
                    None => continue,
                };

            // No timeout exceeded.
            if since_last_event < timeout {
                continue;
            }

            // Increment the missed-heartbeat counter.
            if let Some(hb) = world.get_mut::<WorkerHeartbeat>(entity) {
                hb.mark_missed();
            }

            // Check whether the miss threshold has been reached.
            let misses = world
                .get::<WorkerHeartbeat>(entity)
                .map_or(0, |hb| hb.consecutive_misses);

            if misses < self.policy.max_consecutive_misses {
                continue;
            }

            // Escalate: request recovery, write Abandoned outcome, transition.
            let request_id = world
                .get::<WorkerRequest>(entity)
                .map(|r| r.request_id.clone());
            let worker_id = world
                .get::<WorkerAssignment>(entity)
                .map(|a| a.worker_id.clone());

            if let Some(wid) = &worker_id {
                // Re-borrow resources in a narrow scope to avoid holding Refs
                // across subsequent mutable operations.
                if let Some(diagnostics) = world.get_resource::<WorkerDiagnosticsResource>() {
                    diagnostics.record_watchdog_escalation();
                }
                if let Some(pool) = world.get_resource::<WorkerPoolResource>() {
                    let _ = pool.request_recovery(wid);
                }

                // Clean up any pending request in the registry using request_id.
                if let Some(registry) = world.get_resource::<WorkerResponseRegistry>() {
                    if let Some(rid) = &request_id {
                        let _ = registry.remove_pending(rid);
                    }
                }
            }

            let assignment_gen = world
                .get::<WorkerAssignment>(entity)
                .map_or(0, |a| a.generation);

            let outcome = WorkerOutcome::new(
                TerminalStatus::Abandoned,
                WorkerErrorCategory::Timeout,
                None,
                assignment_gen,
            );
            world.insert(entity, outcome);

            if let Some(lc) = world.get_mut::<WorkerLifecycle>(entity) {
                let _ = lc.transition_to(WorkerRequestPhase::Abandoned);
            }
        }

        SystemResult::ok()
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Compute the idle duration and applicable timeout for an entity based on
/// its current phase.
///
/// Returns `None` when the entity lacks the required components to compute
/// idle time.
fn compute_idle_time(
    world: &World,
    entity: crate::runtime::world::Entity,
    phase: WorkerRequestPhase,
    now: Instant,
) -> Option<(Duration, Duration)> {
    match phase {
        WorkerRequestPhase::AwaitingFirstEvent => {
            // Time since assignment (dispatch) — NOT global elapsed.
            let assignment = world.get::<WorkerAssignment>(entity)?;
            let age = now.saturating_duration_since(assignment.assigned_at);
            Some((age, Duration::from_secs(30)))
        }
        WorkerRequestPhase::Dispatching | WorkerRequestPhase::Streaming | WorkerRequestPhase::CancelRequested => {
            // Time since last heartbeat.
            let heartbeat = world.get::<WorkerHeartbeat>(entity)?;
            let age = now.saturating_duration_since(heartbeat.last_heartbeat_at);
            Some((age, Duration::from_secs(10)))
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runtime::world::World;
    use crate::runtime::components::WorkerAssignment;

    fn setup_world() -> World {
        let mut world = World::with_capacity(64);
        world.register_component::<WorkerAssignment>();
        world.register_component::<WorkerLifecycle>();
        world.register_component::<WorkerHeartbeat>();
        world.register_component::<WorkerRequest>();
        world.register_component::<WorkerOutcome>();
        world.insert_resource(MonotonicClockResource::new());
        world.insert_resource(WorkerDiagnosticsResource::new());
        world.insert_resource(WorkerPoolResource::new());
        world
    }

    #[test]
    fn empty_world_is_ok() {
        let mut world = setup_world();
        let mut system = WorkerWatchdogSystem::new();
        let mut buffer = Vec::new();
        let mut cw = CommandWriter::new(&mut buffer, Stage::Maintenance, WORKER_WATCHDOG_SYSTEM);
        let result = system.run(&mut world, &mut cw);
        assert!(matches!(result, SystemResult::Ok));
    }

    #[test]
    fn system_without_active_entities_is_ok() {
        let mut world = setup_world();
        // Spawn an entity but leave it in Queued (which is not an active phase
        // monitored by the watchdog).
        let entity = world.spawn().expect("spawn");
        world.insert(entity, WorkerAssignment::new("w-1", 0));
        world.insert(entity, WorkerLifecycle::new());
        world.insert(entity, WorkerHeartbeat::new("w-1", 0));

        let mut system = WorkerWatchdogSystem::new();
        let mut buffer = Vec::new();
        let mut cw = CommandWriter::new(&mut buffer, Stage::Maintenance, WORKER_WATCHDOG_SYSTEM);
        let result = system.run(&mut world, &mut cw);
        assert!(matches!(result, SystemResult::Ok));
    }

    #[test]
    fn metadata_is_valid() {
        let metadata = <WorkerWatchdogSystem as SystemSpec>::metadata().expect("metadata");
        assert_eq!(metadata.name, "worker_watchdog");
        assert_eq!(metadata.id, WORKER_WATCHDOG_SYSTEM);
        assert_eq!(metadata.stage, Stage::Maintenance);
        assert_eq!(metadata.order, 1);
        assert_eq!(metadata.after, &[WORKER_EVENT_DRAIN_SYSTEM]);
    }

    #[test]
    fn has_after_edge_to_event_drain() {
        assert_eq!(<WorkerWatchdogSystem as SystemSpec>::AFTER, &[WORKER_EVENT_DRAIN_SYSTEM]);
    }

    #[test]
    fn default_policy_values() {
        let policy = WorkerWatchdogPolicy::default();
        assert_eq!(policy.first_event_timeout, Duration::from_secs(30));
        assert_eq!(policy.heartbeat_timeout, Duration::from_secs(10));
        assert_eq!(policy.max_consecutive_misses, 3);
    }

    #[test]
    fn system_id_constant() {
        assert_eq!(WORKER_WATCHDOG_SYSTEM, SystemId(102));
    }
}
