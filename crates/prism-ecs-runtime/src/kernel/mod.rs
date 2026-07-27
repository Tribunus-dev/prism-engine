//! Runtime kernel — the constitutional kernel that owns the authoritative
//! world and the typed command surface.
//!
//! This module is the kernel's directory index. The godfile that lived
//! here (`kernel.rs`, 1979 LOC) has been decomposed into five
//! single-authority sub-modules:
//!
//! - [`markers`] — `PlannedMarker`, `AdmittedMarker`, `PublishedMarker`
//!   (canonical, just `Component` impls).
//! - [`command_dispatch`] — `Command`, `CommandResult`, `CommitOutcome`,
//!   `CommandEnvelope`, and the canonical submit/replay path through
//!   the world (canonical data shapes; execution-boundary submit path
//!   per criterion 3).
//! - [`agent_snapshot`] — `AgentSnapshot` and the `query_agents` read
//!   projection (canonical).
//! - [`kernel_health`] — `KernelHealth` and the canonical health
//!   computation (canonical).
//! - [`executor_loop`] — kernel-side tick loop, snapshot persistence,
//!   restart recovery, and the typed `KernelTickExecutor` port
//!   (execution-boundary per criterion 3).
//!
//! Authority: this directory index owns the kernel-level wiring — the
//! `RuntimeKernel` and `KernelHandle` types, their constructors, the
//! `unsafe impl Send/Sync` for `KernelHandle`, and the re-exports
//! that form the kernel's public surface. It does **not** own any
//! canonical fact of its own; every authority lives in a sub-module.
//!
//! ## Classification
//!
//! The `RuntimeKernelInner` struct holds process-local state (world
//! `RwLock`, `parking_lot::Mutex<Option<schedule>>`, `AtomicU64`,
//! `mpsc::Receiver` for the state stream, `Box<dyn port>` for every
//! registered adapter) and therefore crosses criterion 3. The kernel
//! is execution-boundary; the data shapes it owns are canonical.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use prism_ecs_core::{global_context, StateStream, TraceContext, World};

use crate::ports::{
    CommandStore, KernelClock, LeaseCoordinator, ProviderSelectionRequest, ProviderSelector,
    RecoveryReport, RuntimeError, SnapshotStore, StaticProviderSelector, TickReceiptStore,
};

use command_dispatch::CommandDispatchContext;

pub mod agent_snapshot;
pub mod command_dispatch;
pub mod executor_loop;
pub mod kernel_health;
pub mod markers;

// Re-exports for the public kernel surface (matches the original
// `pub use kernel::{...}` block in lib.rs).
pub use agent_snapshot::{query_agents, AgentSnapshot};
pub use command_dispatch::{Command, CommandEnvelope, CommandResult, CommitOutcome};
pub use kernel_health::KernelHealth;
pub use markers::{AdmittedMarker, PlannedMarker, PublishedMarker};

// ── RuntimeKernelInner / RuntimeKernel / KernelHandle ──────────────────────

/// Owned kernel state shared by [`RuntimeKernel`] and [`KernelHandle`].
///
/// Every field is either immutable, behind a synchronization
/// primitive, or only mutated through exclusive access. `unsafe impl
/// Send + Sync` on `KernelHandle` is justified by this invariant.
#[allow(dead_code)]
pub(crate) struct RuntimeKernelInner {
    pub world: Arc<std::sync::RwLock<World>>,
    pub command_store: Box<dyn CommandStore>,
    pub snapshot_store: Box<dyn SnapshotStore>,
    pub tick_receipt_store: Box<dyn TickReceiptStore>,
    pub lease_coordinator: Box<dyn LeaseCoordinator>,
    pub _clock: Box<dyn KernelClock>,
    pub provider_selector: Arc<dyn ProviderSelector>,
    pub backend_resources: crate::backend::BackendExecutionRegistry,
    pub sequence: AtomicU64,
    pub trace: TraceContext,
    pub state_stream: StateStream,
}

/// The runtime kernel — owns the authoritative World.
pub struct RuntimeKernel {
    pub(crate) inner: Arc<RuntimeKernelInner>,
    /// Registered schedule for tick execution.
    pub(crate) schedule: parking_lot::Mutex<Option<crate::schedule::RuntimeSchedule>>,
}

/// Thread-safe handle to the runtime kernel.
#[derive(Clone)]
pub struct KernelHandle {
    pub(crate) inner: Arc<RuntimeKernelInner>,
}

// SAFETY: `KernelHandle` exposes no mutable references to its inner
// state without holding the kernel's internal lock. All cross-thread
// access is serialized through that lock, so `Send` is sound. The
// inner types are `Sync` because every field is either immutable,
// behind a synchronization primitive, or only mutated through
// exclusive access.
unsafe impl Send for KernelHandle {}
// SAFETY: see `Send` impl above. `Sync` follows because the handle's
// shared-reference operations all delegate to methods that take the
// kernel's internal lock; no shared reference escapes the lock.
unsafe impl Sync for KernelHandle {}

impl RuntimeKernel {
    /// Borrow the inner state and build the four sub-module contexts.
    fn contexts(&self) -> ExecContexts<'_> {
        let exec = executor_loop::context(
            &self.inner.world,
            &*self.inner.command_store,
            &*self.inner.lease_coordinator,
            &*self.inner.snapshot_store,
            &*self.inner.tick_receipt_store,
            &self.inner.sequence,
            &self.inner.trace,
            &self.inner.state_stream,
            &self.inner.provider_selector,
            &self.inner.backend_resources,
            &self.schedule,
        );
        let dispatch = CommandDispatchContext {
            world: &self.inner.world,
            command_store: &*self.inner.command_store,
            lease_coordinator: &*self.inner.lease_coordinator,
            sequence: &self.inner.sequence,
            trace: &self.inner.trace,
            state_stream: &self.inner.state_stream,
        };
        ExecContexts { exec, dispatch }
    }
}

struct ExecContexts<'a> {
    exec: executor_loop::ExecutorLoopContext<'a>,
    dispatch: CommandDispatchContext<'a>,
}

impl RuntimeKernel {
    /// Create a kernel with default in-memory ports.
    pub fn new() -> Self {
        Self::with_ports(
            Box::new(crate::test_adapters::InMemoryCommandStore::new()),
            Box::new(crate::test_adapters::InMemorySnapshotStore::new()),
            Box::new(crate::test_adapters::InMemoryTickReceiptStore::new()),
            Box::new(crate::test_adapters::InMemoryLeaseCoordinator::new()),
            Box::new(crate::test_adapters::DeterministicClock::new(1000)),
        )
    }

    pub fn with_ports(
        command_store: Box<dyn CommandStore>,
        snapshot_store: Box<dyn SnapshotStore>,
        tick_receipt_store: Box<dyn TickReceiptStore>,
        lease_coordinator: Box<dyn LeaseCoordinator>,
        clock: Box<dyn KernelClock>,
    ) -> Self {
        Self::with_ports_and_provider_selector(
            command_store,
            snapshot_store,
            tick_receipt_store,
            lease_coordinator,
            clock,
            Arc::new(StaticProviderSelector::default()),
        )
    }

    /// Create a kernel with explicit provider selection authority while
    /// retaining all existing persistence and lease ports.
    pub fn with_ports_and_provider_selector(
        command_store: Box<dyn CommandStore>,
        snapshot_store: Box<dyn SnapshotStore>,
        tick_receipt_store: Box<dyn TickReceiptStore>,
        lease_coordinator: Box<dyn LeaseCoordinator>,
        clock: Box<dyn KernelClock>,
        provider_selector: Arc<dyn ProviderSelector>,
    ) -> Self {
        let trace = global_context();
        Self {
            inner: Arc::new(RuntimeKernelInner {
                world: Arc::new(std::sync::RwLock::new(World::new())),
                command_store,
                snapshot_store,
                tick_receipt_store,
                lease_coordinator,
                _clock: clock,
                provider_selector,
                backend_resources: crate::backend::BackendExecutionRegistry::new(),
                sequence: AtomicU64::new(0),
                trace: trace.clone(),
                state_stream: StateStream::global(),
            }),
            schedule: parking_lot::Mutex::new(None),
        }
    }

    /// Create a kernel with an existing world and default in-memory
    /// ports. Used by the daemon to integrate the kernel with the
    /// authoritative PrismWorld.
    pub fn with_existing_world(world: Arc<std::sync::RwLock<World>>) -> Self {
        Self::with_existing_world_and_ports(
            world,
            Box::new(crate::test_adapters::InMemoryCommandStore::new()),
            Box::new(crate::test_adapters::InMemorySnapshotStore::new()),
            Box::new(crate::test_adapters::InMemoryTickReceiptStore::new()),
            Box::new(crate::test_adapters::InMemoryLeaseCoordinator::new()),
            Box::new(crate::test_adapters::DeterministicClock::new(1000)),
        )
    }

    /// Create a kernel with an existing world and custom ports.
    pub fn with_existing_world_and_ports(
        world: Arc<std::sync::RwLock<World>>,
        command_store: Box<dyn CommandStore>,
        snapshot_store: Box<dyn SnapshotStore>,
        tick_receipt_store: Box<dyn TickReceiptStore>,
        lease_coordinator: Box<dyn LeaseCoordinator>,
        clock: Box<dyn KernelClock>,
    ) -> Self {
        Self::with_existing_world_and_ports_and_provider_selector(
            world,
            command_store,
            snapshot_store,
            tick_receipt_store,
            lease_coordinator,
            clock,
            Arc::new(StaticProviderSelector::default()),
        )
    }

    /// Create a kernel over an existing authoritative world with
    /// explicit provider selection authority.
    pub fn with_existing_world_and_ports_and_provider_selector(
        world: Arc<std::sync::RwLock<World>>,
        command_store: Box<dyn CommandStore>,
        snapshot_store: Box<dyn SnapshotStore>,
        tick_receipt_store: Box<dyn TickReceiptStore>,
        lease_coordinator: Box<dyn LeaseCoordinator>,
        clock: Box<dyn KernelClock>,
        provider_selector: Arc<dyn ProviderSelector>,
    ) -> Self {
        let trace = global_context();
        Self {
            inner: Arc::new(RuntimeKernelInner {
                world,
                command_store,
                snapshot_store,
                tick_receipt_store,
                lease_coordinator,
                _clock: clock,
                provider_selector,
                backend_resources: crate::backend::BackendExecutionRegistry::new(),
                sequence: AtomicU64::new(0),
                trace: trace.clone(),
                state_stream: StateStream::global(),
            }),
            schedule: parking_lot::Mutex::new(None),
        }
    }

    /// Return a clonable handle that exposes the canonical kernel
    /// operations to other threads.
    pub fn handle(&self) -> KernelHandle {
        KernelHandle {
            inner: self.inner.clone(),
        }
    }

    /// Compute the canonical health of the kernel.
    pub fn health(&self) -> KernelHealth {
        let ctxs = self.contexts();
        executor_loop::health(&ctxs.exec).expect("health on owned world")
    }

    /// Recover the kernel from the command store + snapshot store.
    pub fn recover(&self) -> Result<RecoveryReport, RuntimeError> {
        let ctxs = self.contexts();
        executor_loop::recover(&ctxs.exec)
    }

    /// Register a schedule for tick execution.
    pub fn set_schedule(&self, schedule: crate::schedule::RuntimeSchedule) {
        let ctxs = self.contexts();
        executor_loop::set_schedule(&ctxs.exec, schedule);
    }

    /// Run a single tick on the registered schedule.
    pub fn run_tick(&self) -> Result<crate::schedule::TickReceipt, RuntimeError> {
        let ctxs = self.contexts();
        executor_loop::run_tick(&ctxs.exec)
    }

    /// Run a tick and persist the receipt.
    pub fn run_kernel_tick(&self, instance_id: &str) -> Result<(), RuntimeError> {
        let ctxs = self.contexts();
        executor_loop::run_kernel_tick(&ctxs.exec, instance_id)
    }

    /// Run ticks until `target_tick` (inclusive) is reached.
    pub fn run_tick_to(
        &self,
        target_tick: u64,
    ) -> Result<Vec<crate::schedule::TickReceipt>, RuntimeError> {
        let ctxs = self.contexts();
        executor_loop::run_tick_to(&ctxs.exec, target_tick)
    }

    /// Capture the canonical world snapshot.
    pub fn capture_snapshot(&self) -> Result<crate::ports::WorldSnapshot, RuntimeError> {
        let ctxs = self.contexts();
        executor_loop::capture_snapshot(&ctxs.exec)
    }

    /// Capture and persist a snapshot.
    pub fn save_snapshot(&self) -> Result<(), RuntimeError> {
        let ctxs = self.contexts();
        executor_loop::save_snapshot(&ctxs.exec)
    }

    /// Graceful shutdown: persist final snapshot.
    pub fn shutdown(&self) -> Result<(), RuntimeError> {
        let ctxs = self.contexts();
        executor_loop::shutdown(&ctxs.exec)
    }
}

impl Default for RuntimeKernel {
    fn default() -> Self {
        Self::new()
    }
}

impl KernelHandle {
    /// Subscribe to the authoritative ECS runtime state stream.
    pub fn state_stream(&self) -> std::sync::mpsc::Receiver<prism_ecs_core::StateRecord> {
        self.inner.state_stream.subscribe()
    }

    /// Capture the canonical state snapshot.
    pub fn state_snapshot(&self) -> prism_ecs_core::StateSnapshot {
        self.inner.state_stream.snapshot()
    }

    /// Publish a state record to the canonical state stream.
    pub fn publish_state(
        &self,
        domain: impl Into<String>,
        phase: impl Into<String>,
        kind: impl Into<String>,
        status: impl Into<String>,
        state: std::collections::BTreeMap<String, serde_json::Value>,
    ) {
        self.inner
            .state_stream
            .emit(&self.inner.trace, domain, phase, kind, status, state);
    }

    /// Persistent compiled-artifact/backend resources for kernel dispatch.
    pub fn backend_resources(&self) -> crate::backend::BackendExecutionRegistry {
        self.inner.backend_resources.clone()
    }

    /// Register a compiled kernel artifact.
    pub fn register_kernel_artifact(
        &self,
        artifact: prism_ecs_kernel::KernelArtifact,
    ) -> Result<crate::backend::KernelArtifactBinding, RuntimeError> {
        let noop = noop_schedule_lock();
        let ctx = executor_loop::context(
            &self.inner.world,
            &*self.inner.command_store,
            &*self.inner.lease_coordinator,
            &*self.inner.snapshot_store,
            &*self.inner.tick_receipt_store,
            &self.inner.sequence,
            &self.inner.trace,
            &self.inner.state_stream,
            &self.inner.provider_selector,
            &self.inner.backend_resources,
            noop,
        );
        executor_loop::register_kernel_artifact(&ctx, artifact)
    }

    /// Attach an already-registered artifact to a work entity.
    pub fn bind_kernel_artifact(
        &self,
        work_entity: u64,
        binding: crate::backend::KernelArtifactBinding,
    ) -> Result<(), RuntimeError> {
        let noop = noop_schedule_lock();
        let ctx = executor_loop::context(
            &self.inner.world,
            &*self.inner.command_store,
            &*self.inner.lease_coordinator,
            &*self.inner.snapshot_store,
            &*self.inner.tick_receipt_store,
            &self.inner.sequence,
            &self.inner.trace,
            &self.inner.state_stream,
            &self.inner.provider_selector,
            &self.inner.backend_resources,
            noop,
        );
        executor_loop::bind_kernel_artifact(&ctx, work_entity, binding)
    }

    /// Build the provider-neutral dispatcher backed by this kernel's
    /// persistent backend resources.
    pub fn kernel_dispatcher(&self) -> Arc<crate::backend::KernelBackendDispatcher> {
        let noop = noop_schedule_lock();
        let ctx = executor_loop::context(
            &self.inner.world,
            &*self.inner.command_store,
            &*self.inner.lease_coordinator,
            &*self.inner.snapshot_store,
            &*self.inner.tick_receipt_store,
            &self.inner.sequence,
            &self.inner.trace,
            &self.inner.state_stream,
            &self.inner.provider_selector,
            &self.inner.backend_resources,
            noop,
        );
        executor_loop::kernel_dispatcher(&ctx)
    }

    /// Select the provider for an operation.
    pub fn select_provider(
        &self,
        request: &ProviderSelectionRequest,
    ) -> crate::ports::ProviderSelectionReceipt {
        let noop = noop_schedule_lock();
        let ctx = executor_loop::context(
            &self.inner.world,
            &*self.inner.command_store,
            &*self.inner.lease_coordinator,
            &*self.inner.snapshot_store,
            &*self.inner.tick_receipt_store,
            &self.inner.sequence,
            &self.inner.trace,
            &self.inner.state_stream,
            &self.inner.provider_selector,
            &self.inner.backend_resources,
            noop,
        );
        executor_loop::select_provider(&ctx, request)
    }

    /// Submit a typed command for execution with atomic epoch fencing.
    pub fn submit(
        &self,
        envelope: CommandEnvelope,
    ) -> Result<CommitOutcome, RuntimeError> {
        let ctx = CommandDispatchContext {
            world: &self.inner.world,
            command_store: &*self.inner.command_store,
            lease_coordinator: &*self.inner.lease_coordinator,
            sequence: &self.inner.sequence,
            trace: &self.inner.trace,
            state_stream: &self.inner.state_stream,
        };
        command_dispatch::submit(envelope, &ctx)
    }

    /// Query all agent entities with their phase and lifecycle.
    pub fn query_agents(&self) -> Vec<AgentSnapshot> {
        let world = self
            .inner
            .world
            .read()
            .unwrap_or_else(|e| panic!("world read lock poisoned: {e}"));
        query_agents(&world).expect("query_agents on owned world")
    }

    /// Lock the world and return a read-only guard.
    pub fn lock_world(&self) -> std::sync::RwLockReadGuard<'_, prism_ecs_core::World> {
        self.inner
            .world
            .read()
            .unwrap_or_else(|e| panic!("world read lock poisoned: {e}"))
    }

    /// Compute the canonical kernel health.
    pub fn health(&self) -> KernelHealth {
        let noop = noop_schedule_lock();
        let ctx = executor_loop::context(
            &self.inner.world,
            &*self.inner.command_store,
            &*self.inner.lease_coordinator,
            &*self.inner.snapshot_store,
            &*self.inner.tick_receipt_store,
            &self.inner.sequence,
            &self.inner.trace,
            &self.inner.state_stream,
            &self.inner.provider_selector,
            &self.inner.backend_resources,
            noop,
        );
        executor_loop::health(&ctx).expect("health on owned world")
    }
}

/// Construct a `RuntimeKernel` with default in-memory ports.
///
/// Equivalent to `RuntimeKernel::new()`. Re-exported so callers can
/// write `prism_ecs_runtime::create_kernel()`.
pub fn create_kernel() -> RuntimeKernel {
    RuntimeKernel::new()
}

// ── Schedule-lock helper for the KernelHandle fast paths ──────────────────

/// Return a `&'static` reference to a `parking_lot::Mutex` that holds
/// `None` for the schedule. Used by `KernelHandle` methods that do not
/// need the schedule (e.g. `register_kernel_artifact`,
/// `bind_kernel_artifact`, `kernel_dispatcher`, `select_provider`,
/// `health`).
///
/// This is sound because the only operation that reads the schedule
/// (tick/snapshot/recover) is invoked through `RuntimeKernel` which
/// holds the real `schedule` field. The static fallback mutex is
/// empty and `get_schedule_hash` returns zeros in that case — the
/// canonical schedule is only readable through `RuntimeKernel`.
fn noop_schedule_lock(
) -> &'static parking_lot::Mutex<Option<crate::schedule::RuntimeSchedule>> {
    use std::sync::OnceLock;
    static NOOP: OnceLock<parking_lot::Mutex<Option<crate::schedule::RuntimeSchedule>>> =
        OnceLock::new();
    NOOP.get_or_init(|| parking_lot::Mutex::new(None))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `create_kernel()` returns a usable kernel with no schedule.
    #[test]
    fn create_kernel_returns_working_kernel() {
        let kernel = create_kernel();
        let handle = kernel.handle();
        let h = handle.health();
        assert_eq!(h.status, "running");
        assert_eq!(h.entity_count, 0);
        // No schedule registered — hash is all zeros.
        let snap_ctx = executor_loop::context(
            &kernel.inner.world,
            &*kernel.inner.command_store,
            &*kernel.inner.lease_coordinator,
            &*kernel.inner.snapshot_store,
            &*kernel.inner.tick_receipt_store,
            &kernel.inner.sequence,
            &kernel.inner.trace,
            &kernel.inner.state_stream,
            &kernel.inner.provider_selector,
            &kernel.inner.backend_resources,
            &kernel.schedule,
        );
        assert_eq!(executor_loop::get_schedule_hash(&snap_ctx), [0u8; 32]);
    }

    /// A spawned agent is visible through the kernel's query path.
    #[test]
    fn spawn_adds_agent_visible_through_query() {
        let kernel = RuntimeKernel::new();
        let handle = kernel.handle();

        let outcome = handle
            .submit(CommandEnvelope::new(Command::SpawnAgent {
                parent_id: 0,
                task: "test agent".to_string(),
                max_steps: 10,
            }))
            .expect("spawn should succeed");
        let entity_id = match outcome.result {
            CommandResult::Spawned { entity_id } => entity_id,
            other => panic!("expected Spawned, got {other:?}"),
        };
        assert!(entity_id > 0);

        let agents = handle.query_agents();
        assert_eq!(agents.len(), 1);
        assert_eq!(agents[0].entity_id, entity_id);
        assert_eq!(agents[0].phase, "Planning");
    }

    /// `recover()` on a fresh kernel returns the `fresh` report.
    #[test]
    fn recover_on_fresh_kernel_returns_fresh_state() {
        let kernel = RuntimeKernel::new();
        let report = kernel.recover().expect("recover");
        assert_eq!(report.recovery_state, "fresh");
        assert_eq!(report.replayed_commands, 0);
    }

    /// `save_snapshot` + `recover` round-trip on a fresh kernel.
    #[test]
    fn save_then_recover_returns_fresh_report() {
        let kernel = RuntimeKernel::new();
        kernel.save_snapshot().expect("save");
        let report = kernel.recover().expect("recover");
        assert_eq!(report.recovery_state, "fresh");
        // Snapshot watermark is recorded.
        assert_eq!(report.snapshot_sequence, 0);
    }

    /// The static noop schedule lock is shared and reports `None`.
    #[test]
    fn noop_schedule_lock_is_empty() {
        let lock = noop_schedule_lock();
        assert!(lock.lock().is_none());
    }
}
