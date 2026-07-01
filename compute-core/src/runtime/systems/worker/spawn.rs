//! WorkerSpawnSystem — spawns OS worker processes and assigns them to
//! request entities.
//!
//! Runs during `Stage::Intake` (order 1, after [`WorkerIngressSystem`] at
//! order 0).  Queries entities whose lifecycle phase is `Queued` and that
//! lack a `WorkerAssignment`, spawns the worker binary via
//! [`WorkerProcessManager`], and transitions the lifecycle to `Dispatching`.

use crate::runtime::resources::worker_process_manager::{
    WorkerProcessManager, WorkerId,
};
use crate::runtime::resources::WorkerDiagnosticsResource;
use crate::runtime::scheduling::command::CommandWriter;
use crate::runtime::scheduling::metadata::{
    ExecutionClass, ErasedSystem, SerializationPolicy, Stage, SystemId,
    SystemMetadata, SystemResult, SystemSpec,
};
use crate::runtime::world::{Entity, World};
use crate::runtime::components::{
    WorkerAssignment, WorkerLifecycle, WorkerRequestPhase,
};

// ---------------------------------------------------------------------------
// Default paths (overridable via environment)
// ---------------------------------------------------------------------------

/// Environment variable for the worker binary path.
const ENV_WORKER_BINARY: &str = "PRISM_WORKER_BINARY";
/// Environment variable for the model path passed to the worker.
const ENV_WORKER_MODEL: &str = "PRISM_WORKER_MODEL";
/// Fallback binary path when the environment variable is not set.
const DEFAULT_WORKER_BINARY: &str = "./prism-worker";
/// Fallback model path when the environment variable is not set.
const DEFAULT_WORKER_MODEL: &str = "/models/default";

/// Read a worker binary path from the environment, falling back to a
/// compile-time default.
fn worker_binary_path() -> String {
    std::env::var(ENV_WORKER_BINARY)
        .unwrap_or_else(|_| DEFAULT_WORKER_BINARY.to_string())
}

/// Read a model path from the environment, falling back to a compile-time
/// default.
fn worker_model_path() -> String {
    std::env::var(ENV_WORKER_MODEL)
        .unwrap_or_else(|_| DEFAULT_WORKER_MODEL.to_string())
}

// ---------------------------------------------------------------------------
// WorkerSpawnSystem
// ---------------------------------------------------------------------------

/// Spawns OS worker processes for entities awaiting assignment.
///
/// # Algorithm
///
/// 1. Iterate all entities that have a `WorkerLifecycle` component.
/// 2. Skip entities whose phase is not `Queued` or that already have a
///    `WorkerAssignment` component.
/// 3. Call [`WorkerProcessManager::spawn_worker`] with the configured
///    binary and model paths.
/// 4. Insert a [`WorkerAssignment`] component on the entity.
/// 5. Transition the [`WorkerLifecycle`] to `Dispatching`.
pub struct WorkerSpawnSystem {
    _private: (),
}

impl WorkerSpawnSystem {
    /// Create a new spawn system.
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl Default for WorkerSpawnSystem {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SystemSpec — compile-time declaration
// ---------------------------------------------------------------------------

impl SystemSpec for WorkerSpawnSystem {
    type Reads = WorkerLifecycle;
    type Writes = (WorkerAssignment, WorkerLifecycle);
    type ReadResources = ();
    type WriteResources = (WorkerProcessManager, WorkerDiagnosticsResource);

    const NAME: &'static str = "worker_spawn";
    const ID: SystemId = SystemId(104);
    const STAGE: Stage = Stage::Intake;
    const ORDER: i32 = 1;
    const EXECUTION_CLASS: ExecutionClass = ExecutionClass::Serial;
    const SERIALIZATION: SerializationPolicy = SerializationPolicy::ExplicitOnly;
}

// ---------------------------------------------------------------------------
// ErasedSystem — object-safe runtime
// ---------------------------------------------------------------------------

impl ErasedSystem for WorkerSpawnSystem {
    fn metadata(&self) -> &SystemMetadata {
        static META: std::sync::LazyLock<SystemMetadata> =
            std::sync::LazyLock::new(|| {
                <WorkerSpawnSystem as SystemSpec>::metadata()
                    .expect("WorkerSpawnSystem metadata construction")
            });
        &META
    }

    fn run(
        &mut self,
        world: &mut World,
        _commands: &mut CommandWriter,
    ) -> SystemResult {
        // ---- 1. Collect candidate entities ----
        // We iterate *all* entities that have a WorkerLifecycle component
        // and filter to those that are Queued with no WorkerAssignment.
        let candidates: Vec<Entity> = world
            .iter_entities_with::<WorkerLifecycle>()
            .filter(|entity| {
                // Must be Queued ...
                let phase_ok = world
                    .get::<WorkerLifecycle>(*entity)
                    .map(|lc| lc.phase == WorkerRequestPhase::Queued)
                    .unwrap_or(false);
                // ... and not already assigned.
                let unassigned = !world.has::<WorkerAssignment>(*entity);
                phase_ok && unassigned
            })
            .collect();

        if candidates.is_empty() {
            return SystemResult::ok();
        }

        // ---- 2. Resolve binary and model paths (once per tick) ----
        let binary_path = worker_binary_path();
        let model_path = worker_model_path();

        // ---- 3. Spawn a worker process for each candidate ----
        for entity in &candidates {
            // Scope the mutable resource borrow so it is dropped before
            // subsequent world operations on the same entity.
            let spawn_result = {
                let mgr = match world.get_resource_mut::<WorkerProcessManager>() {
                    Some(mgr) => mgr,
                    None => {
                        return SystemResult::err(
                            "WorkerProcessManager resource not registered",
                        );
                    }
                };
                mgr.spawn_worker(&binary_path, &model_path)
            };

            let worker_id: WorkerId = match spawn_result {
                Ok(id) => id,
                Err(_e) => {
                    // Record diagnostic and continue; the entity stays
                    // Queued for retry on the next tick.
                    if let Some(diag) =
                        world.get_resource_mut::<WorkerDiagnosticsResource>()
                    {
                        diag.record_restart_request();
                    }
                    continue;
                }
            };

            // ---- 4. Insert WorkerAssignment ----
            world.insert(
                *entity,
                WorkerAssignment::new(worker_id.to_string(), 1),
            );

            // ---- 5. Transition lifecycle Queued -> Dispatching ----
            if let Some(lc) = world.get_mut::<WorkerLifecycle>(*entity) {
                let _ = lc.transition_to(WorkerRequestPhase::Dispatching);
                let _ = lc; // release mutable borrow
            }
        }

        SystemResult::ok()
    }
}
