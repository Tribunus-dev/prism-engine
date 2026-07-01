//! WorkerIngressSystem — drains the ingress queue, dispatches requests to
//! healthy workers, and manages lifecycle transitions.
//!
//! This system runs during `Stage::Intake` and is responsible for the
//! Queued → Dispatching → AwaitingFirstEvent (or → Failed) transition.

use crate::runtime::scheduling::command::CommandWriter;
use crate::runtime::scheduling::metadata::*;
use crate::runtime::components::{
    WorkerAssignment, WorkerHeartbeat, WorkerLifecycle, WorkerOutcome,
    WorkerRequest, WorkerRequestPhase,
    WORKER_INGRESS_SYSTEM,
};
use crate::runtime::components::worker_health::{TerminalStatus, WorkerErrorCategory};
use crate::runtime::components::worker_request::RequestClass;
use crate::runtime::resources::*;
use crate::runtime::world::{Entity, World};

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

/// Maximum number of ingress entries processed per tick.
pub const BATCH_SIZE: usize = 16;

/// Default request class used when the ingress entry carries no class
/// metadata (placeholder until the bridge path is fully wired).
pub const DEFAULT_REQUEST_CLASS: RequestClass = RequestClass::Generate;

/// Default assignment generation for first-time dispatch.
pub const INITIAL_GENERATION: u32 = 1;

// ---------------------------------------------------------------------------
// WorkerIngressSystem
// ---------------------------------------------------------------------------

/// Processes incoming worker requests from the ingress queue.
///
/// For each entry in the queue the system spawns an entity, inserts request
/// and lifecycle components, selects a healthy worker, emits assignment and
/// heartbeat commands, attempts IPC dispatch, and transitions the lifecycle
/// to either `AwaitingFirstEvent` or `Failed`.
pub struct WorkerIngressSystem {
    _private: (),
}

impl WorkerIngressSystem {
    /// Create a new ingress system.
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for WorkerIngressSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SystemSpec — compile-time declaration
// ---------------------------------------------------------------------------

impl SystemSpec for WorkerIngressSystem {
    type Reads = (WorkerRequest, WorkerLifecycle);
    type Writes = (WorkerAssignment, WorkerLifecycle, WorkerHeartbeat, WorkerOutcome);
    type ReadResources = (WorkerIngressQueue, WorkerResponseRegistry);
    type WriteResources = (WorkerPoolResource, WorkerDiagnosticsResource);

    const NAME: &'static str = "worker_ingress";
    const ID: SystemId = WORKER_INGRESS_SYSTEM;
    const STAGE: Stage = Stage::Intake;
    const ORDER: i32 = 0;
    const EXECUTION_CLASS: ExecutionClass = ExecutionClass::Serial;
    const SERIALIZATION: SerializationPolicy = SerializationPolicy::ExplicitOnly;
}

// ---------------------------------------------------------------------------
// ErasedSystem — object-safe runtime
// ---------------------------------------------------------------------------

impl ErasedSystem for WorkerIngressSystem {
    fn metadata(&self) -> &SystemMetadata {
        // NOTE: metadata() must return a 'static reference because
        // ErasedSystem is stored as Box<dyn ErasedSystem>.  We use a
        // LazyLock to compute it once at first access.
        static META: std::sync::LazyLock<SystemMetadata> =
            std::sync::LazyLock::new(|| {
                <WorkerIngressSystem as SystemSpec>::metadata()
                    .expect("WorkerIngressSystem metadata construction")
            });
        &META
    }

    fn run(
        &mut self,
        world: &mut World,
        commands: &mut CommandWriter,
    ) -> SystemResult {
        // ---- 1. Drain ingress queue (scoped borrow) ----
        let entries = {
            let queue = match world.get_resource_mut::<WorkerIngressQueue>() {
                Some(q) => q,
                None => {
                    return SystemResult::err(
                        "WorkerIngressQueue resource not registered",
                    );
                }
            };
            queue.drain(BATCH_SIZE)
        };

        if entries.is_empty() {
            return SystemResult::ok();
        }

        // ---- 2-8. Process each entry ----
        for entry in entries {
            let entity = Entity(entry.entity_id);

            // ---- 2a. Spawn entity if the bridge did not ----
            let entity = if entity.0 == 0 {
                match world.spawn() {
                    Some(e) => e,
                    None => {
                        // World at capacity — skip this entry; the queue
                        // entry is already consumed, so the request is lost.
                        // Real deployments should back-pressure the bridge.
                        Self::record_diagnostics(world);
                        continue;
                    }
                }
            } else {
                entity
            };

            // ---- 2b. Insert request and lifecycle ----
            // Use the World API for immediate availability; subsequent
            // mutations to lifecycle happen in-place.
            world.insert(entity, WorkerRequest::new(
                entry.request_id.clone(),
                entry.payload.clone(),
                DEFAULT_REQUEST_CLASS,
            ));
            world.insert(entity, WorkerLifecycle::new());

            // ---- 2c. Validate entity is alive and Queued ----
            if !world.is_alive(entity) {
                Self::record_diagnostics(world);
                continue;
            }

            let lifecycle_phase = {
                let lc = match world.get::<WorkerLifecycle>(entity) {
                    Some(lc) => lc,
                    None => {
                        Self::record_diagnostics(world);
                        continue;
                    }
                };
                lc.phase
            };

            if lifecycle_phase != WorkerRequestPhase::Queued {
                Self::record_diagnostics(world);
                continue;
            }

            // ---- 3. Select a healthy worker ----
            let worker_id = match world.get_resource::<WorkerPoolResource>()
                .and_then(|pool| pool.select_healthy_worker())
            {
                Some(id) => id,
                None => {
                    // No worker available — entry stays in queue via
                    // re-push (not yet consumed at resource level because
                    // we drain before spawning). Currently the entry is
                    // consumed from the drain; a future slice should
                    // re-queue or coordinate with admission control.
                    Self::record_diagnostics(world);
                    continue;
                }
            };

            // ---- 4. Emit commands for new components ----
            // WorkerAssignment and WorkerHeartbeat are inserted via the
            // provenance-stamped command buffer.
            if commands
                .insert(entity, WorkerAssignment::new(&worker_id, INITIAL_GENERATION))
                .is_err()
            {
                continue;
            }
            if commands
                .insert(entity, WorkerHeartbeat::new(&worker_id, INITIAL_GENERATION))
                .is_err()
            {
                continue;
            }

            // ---- 5. Transition lifecycle Queued → Dispatching ----
            let lc = match world.get_mut::<WorkerLifecycle>(entity) {
                Some(lc) => lc,
                None => continue,
            };
            if lc.transition_to(WorkerRequestPhase::Dispatching).is_err() {
                // Transition denied — skip silently.
                continue;
            }

            // ---- 6. Issue request through pool ----
            let (request_id, payload) = {
                let req = match world.get::<WorkerRequest>(entity) {
                    Some(r) => r,
                    None => continue,
                };
                (req.request_id.clone(), req.payload.clone())
            };

            let send_result = world
                .get_resource::<WorkerPoolResource>()
                .ok_or_else(|| "WorkerPoolResource not registered".to_string())
                .and_then(|pool| pool.send_request(&worker_id, &request_id, &payload));

            // ---- 7-8. Post-dispatch lifecycle ----
            let lc = match world.get_mut::<WorkerLifecycle>(entity) {
                Some(lc) => lc,
                None => continue,
            };

            match send_result {
                Ok(()) => {
                    // IPC send succeeded → AwaitingFirstEvent
                    if lc
                        .transition_to(WorkerRequestPhase::AwaitingFirstEvent)
                        .is_ok()
                    {
                        // Register a response sink placeholder.
                        // A future slice will wire this to the actual
                        // response channel.
                        Self::register_sink(world, &entry);
                    }
                }
                Err(_) => {
                    // IPC send failed → Failed
                    // Emit the outcome component via command buffer.
                    let _ = commands.insert(
                        entity,
                        WorkerOutcome::new(
                            TerminalStatus::Failed,
                            WorkerErrorCategory::Internal,
                            None,
                            INITIAL_GENERATION,
                        ),
                    );
                    let _ = lc.transition_to(WorkerRequestPhase::Failed);
                    Self::record_diagnostics(world);
                }
            }
        }

        SystemResult::ok()
    }
}

// ---------------------------------------------------------------------------
// Private helpers
// ---------------------------------------------------------------------------

impl WorkerIngressSystem {
    /// Increment the stale/drop counter on the diagnostics resource.
    fn record_diagnostics(world: &mut World) {
        if let Some(diag) = world.get_resource::<WorkerDiagnosticsResource>() {
            diag.stale_event_drops.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }

    /// Register an opaque response sink placeholder so the event drain
    /// system can route responses back.
    fn register_sink(world: &mut World, _entry: &IngressEntry) {
        // Placeholder: a future slice will register a concrete responder
        // with WorkerResponseRegistry using the bridge_correlation_key.
        //
        // let registry = world.get_resource::<WorkerResponseRegistry>();
        // registry.register_sink(
        //     &entry.request_id,
        //     Entity(entry.entity_id),
        //     Box::new(placeholder_sink),
        // );
        let _ = world;
        let _ = _entry;
    }
}
