//! Kernel-side tick loop, snapshot persistence, and restart recovery.
//!
//! Authority: this module owns the kernel's facade over the 8-stage
//! `RuntimeSchedule` — `set_schedule`, `run_tick`, `run_kernel_tick`,
//! `run_tick_to` — and over the world snapshots the kernel
//! captures, persists, and replays during recovery. The 8 stages
//! themselves live in [`crate::schedule`]; this module is the
//! kernel-level entry point that owns the schedule's `parking_lot::Mutex`
//! and serializes ticks through the world lock.
//!
//! ## Classification
//!
//! This module is **execution-boundary** by criterion 3 — it owns the
//! `parking_lot::Mutex<Option<RuntimeSchedule>>`, the world `RwLock`, and
//! the `AtomicU64` sequence counter. The data shapes it produces
//! (`TickReceipt`, `WorldSnapshot`, `RecoveryReport`) are canonical; the
//! path that produces them is not. The typed port for any engine-side
//! effect executor is [`KernelTickExecutor`] — the engine implements
//! this trait to plug its MLX/ANE/CPU execution into the kernel's tick
//! loop. The existing [`crate::ports::WorkDispatcher`] already covers
//! the per-dispatch boundary; `KernelTickExecutor` is the per-tick
//! boundary, designed for future cutover.
//!
//! ## Engine counterpart
//!
//! `compute-core/src/ecs/core/executor.rs` (1,308 LOC) and
//! `executor_projection.rs` (1,074 LOC) are execution-boundary math
//! code (MLX arrays, hardware calls). They are not absorbed here; the
//! future engine adapter will implement [`KernelTickExecutor`].

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use prism_ecs_core::{global_context, Entity, StateStream, TraceContext, World, WorldEpoch};

use crate::backend::{BackendExecutionRegistry, KernelArtifactBinding, KernelBackendDispatcher};
use crate::ports::{
    CommandStore, LeaseCoordinator, ProviderSelectionReceipt, ProviderSelectionRequest,
    ProviderSelector, RecoveryReport, RuntimeError, SnapshotStore, TickReceiptStore,
};
use crate::schedule::{RuntimeSchedule, TickReceipt};

use super::command_dispatch::{
    apply_recovered_command, capture_world_snapshot, CommandDispatchContext,
};
use super::kernel_health::{compute_health_locked, KernelHealth};

// ── Typed port for engine-side tick execution ──────────────────────────────

/// Per-tick executor port for the engine.
///
/// The kernel's `RuntimeSchedule` already implements an 8-stage
/// per-tick loop over the constitutional world. Any backend that wants
/// to take over effect-side execution (MLX, ANE, CPU, remote) does so
/// by implementing this trait and registering the impl on the
/// kernel's `BackendExecutionRegistry`.
///
/// The trait is intentionally minimal: one method, one receipt, one
/// error type. The engine's existing `core/executor.rs` will
/// eventually implement this; the immediate contract is that callers
/// must handle [`RuntimeError`] uniformly across backends.
pub trait KernelTickExecutor: Send + Sync {
    /// Execute one kernel tick. The implementation may invoke the
    /// engine's prologue/layer/epilogue code paths, mutate its own
    /// caches, and return a receipt.
    ///
    /// The receipt is persisted by the kernel's `TickReceiptStore`;
    /// implementations should not persist it themselves.
    fn execute_tick(&self, ctx: &TickExecutorContext) -> Result<TickReceipt, RuntimeError>;
}

/// Context passed to a [`KernelTickExecutor::execute_tick`] call.
///
/// `instance_id` identifies the daemon that owns the kernel; backends
/// that want to attribute ticks per daemon include it in their own
/// metrics. `world_epoch` is the world epoch at the start of the
/// tick — implementations that need fencing should compare against
/// `expected_epoch` if present.
#[derive(Debug, Clone)]
pub struct TickExecutorContext {
    pub instance_id: String,
    pub world_epoch: u64,
    pub expected_epoch: Option<u64>,
}

impl TickExecutorContext {
    pub fn new(instance_id: impl Into<String>, world_epoch: u64) -> Self {
        Self {
            instance_id: instance_id.into(),
            world_epoch,
            expected_epoch: None,
        }
    }

    pub fn with_expected_epoch(mut self, expected: u64) -> Self {
        self.expected_epoch = Some(expected);
        self
    }
}

// ── Borrowed view over the kernel's state for the executor loop ────────────

/// Borrowed view over the parts of `RuntimeKernelInner` that the
/// executor loop needs. The schedule is exposed as the raw
/// `parking_lot::Mutex` reference rather than a wrapper type, so the
/// caller can hold the lock for the duration of a tick.
pub(super) struct ExecutorLoopContext<'a> {
    pub world: &'a Arc<std::sync::RwLock<World>>,
    pub command_store: &'a dyn CommandStore,
    pub lease_coordinator: &'a dyn LeaseCoordinator,
    pub snapshot_store: &'a dyn SnapshotStore,
    pub tick_receipt_store: &'a dyn TickReceiptStore,
    pub sequence: &'a AtomicU64,
    pub trace: &'a TraceContext,
    pub state_stream: &'a StateStream,
    pub provider_selector: &'a Arc<dyn ProviderSelector>,
    pub backend_resources: &'a BackendExecutionRegistry,
    pub schedule_lock: &'a parking_lot::Mutex<Option<RuntimeSchedule>>,
}

// ── Schedule + tick ────────────────────────────────────────────────────────

/// Register a schedule for tick execution.
pub fn set_schedule(ctx: &ExecutorLoopContext<'_>, schedule: RuntimeSchedule) {
    *ctx.schedule_lock.lock() = Some(schedule);
}

/// Run a single tick on the registered schedule.
///
/// Returns `RuntimeError::Entity("no schedule registered")` if no
/// schedule has been registered. Tick errors are surfaced as-is from
/// the schedule.
pub fn run_tick(ctx: &ExecutorLoopContext<'_>) -> Result<TickReceipt, RuntimeError> {
    let sched = ctx.schedule_lock.lock();
    match sched.as_ref() {
        Some(s) => s.run_tick(),
        None => Err(RuntimeError::Entity("no schedule registered".into())),
    }
}

/// Run a tick and persist the receipt through the tick receipt store.
pub fn run_kernel_tick(
    ctx: &ExecutorLoopContext<'_>,
    instance_id: &str,
) -> Result<(), RuntimeError> {
    let receipt = run_tick(ctx)?;
    ctx.tick_receipt_store
        .save(&receipt, instance_id)
        .map_err(|e| RuntimeError::Receipt(e.to_string()))?;
    Ok(())
}

/// Run ticks until the given target tick number (inclusive) is reached.
/// Returns receipts for every tick executed.
pub fn run_tick_to(
    ctx: &ExecutorLoopContext<'_>,
    target_tick: u64,
) -> Result<Vec<TickReceipt>, RuntimeError> {
    let mut receipts = Vec::new();
    loop {
        let receipt = run_tick(ctx)?;
        let tick = receipt.tick_number;
        receipts.push(receipt);
        if tick >= target_tick {
            break;
        }
    }
    Ok(receipts)
}

// ── Snapshot + shutdown ────────────────────────────────────────────────────

/// Return the schedule hash, or zeroed if no schedule is set.
pub fn get_schedule_hash(ctx: &ExecutorLoopContext<'_>) -> [u8; 32] {
    ctx.schedule_lock
        .lock()
        .as_ref()
        .map(|s| s.schedule_hash())
        .unwrap_or([0u8; 32])
}

/// Capture the canonical world snapshot for persistence or recovery.
pub fn capture_snapshot(
    ctx: &ExecutorLoopContext<'_>,
) -> Result<crate::ports::WorldSnapshot, RuntimeError> {
    let schedule_hash = get_schedule_hash(ctx);
    capture_world_snapshot(ctx.world, ctx.sequence, schedule_hash)
}

/// Capture and persist a snapshot through the snapshot store.
pub fn save_snapshot(ctx: &ExecutorLoopContext<'_>) -> Result<(), RuntimeError> {
    let snapshot = capture_snapshot(ctx)?;
    ctx.snapshot_store.save(&snapshot)
}

/// Graceful shutdown: persist final snapshot.
pub fn shutdown(ctx: &ExecutorLoopContext<'_>) -> Result<(), RuntimeError> {
    let snapshot = capture_snapshot(ctx)?;
    ctx.snapshot_store.save(&snapshot)?;
    Ok(())
}

// ── Recovery ───────────────────────────────────────────────────────────────

/// Recover kernel state from the command store and snapshot store.
///
/// The path is:
/// 1. Read `CommandWatermarks` and the latest snapshot.
/// 2. If a snapshot is present, verify its checksum.
/// 3. Replace the world with a fresh `World::new()` (every entity and
///    component is reconstructed from command replay).
/// 4. Replay every completed command from sequence 0 via
///    [`apply_recovered_command`].
/// 5. Validate the allocator against the snapshot (advisory only —
///    entity IDs may legitimately differ if the allocator is
///    deterministic).
/// 6. Restore the sequence counter to `watermarks.last_committed_sequence + 1`.
/// 7. Reconcile every unresolved command by transitioning its state
///    to `"recovery_required"` for the operator to re-derive or
///    cancel.
pub fn recover(ctx: &ExecutorLoopContext<'_>) -> Result<RecoveryReport, RuntimeError> {
    let watermarks = ctx.command_store.high_water_marks()?;
    let snapshot = ctx.snapshot_store.load_latest()?;

    // Verify snapshot if present (for validation only)
    if let Some(ref snap) = &snapshot {
        if !snap.verify() {
            return Err(RuntimeError::Journal(
                "snapshot checksum mismatch — no fallback available".into(),
            ));
        }
    }

    // Start from a pristine world — every entity and component is
    // reconstructed from command replay, ensuring consistent entity IDs.
    {
        let mut world = ctx
            .world
            .write()
            .map_err(|e| RuntimeError::Entity(format!("world write lock poisoned: {e}")))?;
        *world = World::new();
    }

    // Replay ALL completed commands from sequence 0
    let all = ctx.command_store.completed_after(0)?;
    for cmd in &all {
        let dispatch_ctx = CommandDispatchContext {
            world: ctx.world,
            command_store: ctx.command_store,
            lease_coordinator: ctx.lease_coordinator,
            sequence: ctx.sequence,
            trace: ctx.trace,
            state_stream: ctx.state_stream,
        };
        apply_recovered_command(cmd, &dispatch_ctx)?;
    }
    let replayed_count = all.len() as u64;

    // After replay, validate allocator against snapshot (if present)
    if let Some(ref snap) = &snapshot {
        let reconstructed = ctx
            .world
            .read()
            .map_err(|e| RuntimeError::Entity(format!("world read lock poisoned: {e}")))?;
        let reconstructed_alloc =
            prism_ecs_core::snapshot::export_allocator_snapshot(&reconstructed);
        if reconstructed_alloc != snap.payload.allocator_data {
            eprintln!(
                "Kernel: allocator differs from snapshot (acceptable if entity IDs differ)"
            );
        }
    }

    // Set sequence counter after replay
    ctx.sequence
        .store(watermarks.last_committed_sequence + 1, Ordering::SeqCst);

    // Reconcile unresolved commands
    let unresolved = ctx.command_store.unresolved()?;
    let unresolved_count = unresolved.len() as u64;
    for cmd in &unresolved {
        ctx.command_store
            .transition_state(cmd.sequence, "recovery_required")?;
    }

    Ok(RecoveryReport {
        recovery_state: if replayed_count > 0 {
            "recovered".to_string()
        } else {
            "fresh".to_string()
        },
        snapshot_epoch: snapshot
            .as_ref()
            .map(|s| s.payload.world_epoch)
            .unwrap_or(0),
        snapshot_sequence: watermarks.last_committed_sequence,
        replayed_commands: replayed_count,
        unresolved_commands: unresolved_count,
        world_epoch_before: snapshot
            .as_ref()
            .map(|s| s.payload.world_epoch)
            .unwrap_or(0),
    })
}

// ── Provider selection + backend integration ──────────────────────────────

/// Select the provider for an operation through the kernel-owned
/// provider authority. The returned receipt records every attempted
/// provider and the reason a fallback was used.
pub fn select_provider(
    ctx: &ExecutorLoopContext<'_>,
    request: &ProviderSelectionRequest,
) -> ProviderSelectionReceipt {
    ctx.provider_selector.select(request)
}

/// Snapshot the kernel's `BackendExecutionRegistry` for callers that
/// want to construct a `KernelBackendDispatcher`.
pub fn backend_resources(ctx: &ExecutorLoopContext<'_>) -> BackendExecutionRegistry {
    ctx.backend_resources.clone()
}

/// Register a compiled kernel artifact and return the ECS binding that
/// can be attached to a work entity before it enters the schedule.
pub fn register_kernel_artifact(
    ctx: &ExecutorLoopContext<'_>,
    artifact: prism_ecs_kernel::KernelArtifact,
) -> Result<KernelArtifactBinding, RuntimeError> {
    let digest = artifact.manifest.manifest_digest.clone();
    let result = ctx.backend_resources.register_artifact(artifact);
    ctx.state_stream.emit(
        ctx.trace,
        "runtime",
        "model_registration",
        "artifact_registered",
        if result.is_ok() {
            "completed"
        } else {
            "failed"
        },
        std::collections::BTreeMap::from([(
            String::from("artifact_digest"),
            serde_json::json!(digest),
        )]),
    );
    result
}

/// Attach an already-registered artifact reference to a work entity in
/// the authoritative world.
pub fn bind_kernel_artifact(
    ctx: &ExecutorLoopContext<'_>,
    work_entity: u64,
    binding: KernelArtifactBinding,
) -> Result<(), RuntimeError> {
    let entity = Entity::new(work_entity, 0);
    let mut world = ctx
        .world
        .write()
        .map_err(|e| RuntimeError::Entity(format!("world write lock poisoned: {e}")))?;
    if !world.has_entity(entity) {
        return Err(RuntimeError::Entity(format!(
            "work entity {work_entity} does not exist"
        )));
    }
    world
        .add_component(entity, binding)
        .map_err(|error| RuntimeError::Entity(format!("bind kernel artifact: {error}")))
}

/// Build the provider-neutral dispatcher backed by this kernel's
/// persistent backend resources.
pub fn kernel_dispatcher(ctx: &ExecutorLoopContext<'_>) -> Arc<KernelBackendDispatcher> {
    Arc::new(KernelBackendDispatcher::new(backend_resources(ctx)))
}

// ── Kernel health ──────────────────────────────────────────────────────────

/// Compute the canonical health of the kernel.
pub fn health(ctx: &ExecutorLoopContext<'_>) -> Result<KernelHealth, RuntimeError> {
    compute_health_locked(ctx.world, ctx.sequence)
}

// ── Convenience constructors ───────────────────────────────────────────────

/// Build an [`ExecutorLoopContext`] from borrowed kernel state.
#[allow(clippy::too_many_arguments)]
pub fn context<'a>(
    world: &'a Arc<std::sync::RwLock<World>>,
    command_store: &'a dyn CommandStore,
    lease_coordinator: &'a dyn LeaseCoordinator,
    snapshot_store: &'a dyn SnapshotStore,
    tick_receipt_store: &'a dyn TickReceiptStore,
    sequence: &'a AtomicU64,
    trace: &'a TraceContext,
    state_stream: &'a StateStream,
    provider_selector: &'a Arc<dyn ProviderSelector>,
    backend_resources: &'a BackendExecutionRegistry,
    schedule_lock: &'a parking_lot::Mutex<Option<RuntimeSchedule>>,
) -> ExecutorLoopContext<'a> {
    ExecutorLoopContext {
        world,
        command_store,
        lease_coordinator,
        snapshot_store,
        tick_receipt_store,
        sequence,
        trace,
        state_stream,
        provider_selector,
        backend_resources,
        schedule_lock,
    }
}

/// Build the canonical trace + state-stream pair the kernel hands to
/// sub-modules. Equivalent to the original kernel's `global_context()`
/// + `StateStream::global()` calls.
pub fn trace_and_stream() -> (TraceContext, StateStream) {
    (global_context(), StateStream::global())
}

/// World-epoch helper used by the kernel handle's `lock_world` path.
pub fn current_world_epoch(world: &World) -> WorldEpoch {
    world.current_epoch()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ports::RuntimeError;
    use crate::test_adapters::{
        InMemoryCommandStore, InMemoryLeaseCoordinator, InMemorySnapshotStore,
        InMemoryTickReceiptStore,
    };
    use parking_lot::Mutex;
    use std::sync::atomic::AtomicU64;

    /// Test fixture: bundle the owned state so each test gets a
    /// fresh, fully-owned kernel state without `Box::leak`.
    struct Fixture {
        world: Arc<std::sync::RwLock<World>>,
        command_store: InMemoryCommandStore,
        lease_coordinator: InMemoryLeaseCoordinator,
        snapshot_store: InMemorySnapshotStore,
        tick_receipt_store: InMemoryTickReceiptStore,
        sequence: AtomicU64,
        trace: TraceContext,
        state_stream: StateStream,
        provider_selector: Arc<dyn ProviderSelector>,
        backend_resources: BackendExecutionRegistry,
        schedule_lock: Mutex<Option<RuntimeSchedule>>,
    }

    impl Fixture {
        fn new() -> Self {
            let (trace, state_stream) = trace_and_stream();
            Self {
                world: Arc::new(std::sync::RwLock::new(World::new())),
                command_store: InMemoryCommandStore::new(),
                lease_coordinator: InMemoryLeaseCoordinator::new(),
                snapshot_store: InMemorySnapshotStore::new(),
                tick_receipt_store: InMemoryTickReceiptStore::new(),
                sequence: AtomicU64::new(0),
                trace,
                state_stream,
                provider_selector: Arc::new(crate::StaticProviderSelector::default()),
                backend_resources: BackendExecutionRegistry::new(),
                schedule_lock: Mutex::new(None),
            }
        }

        fn ctx(&self) -> ExecutorLoopContext<'_> {
            ExecutorLoopContext {
                world: &self.world,
                command_store: &self.command_store,
                lease_coordinator: &self.lease_coordinator,
                snapshot_store: &self.snapshot_store,
                tick_receipt_store: &self.tick_receipt_store,
                sequence: &self.sequence,
                trace: &self.trace,
                state_stream: &self.state_stream,
                provider_selector: &self.provider_selector,
                backend_resources: &self.backend_resources,
                schedule_lock: &self.schedule_lock,
            }
        }
    }

    /// Tick without a registered schedule returns a typed Entity error.
    #[test]
    fn run_tick_without_schedule_returns_typed_error() {
        let fx = Fixture::new();
        let ctx = fx.ctx();
        let err = run_tick(&ctx).expect_err("expected error");
        assert!(matches!(err, RuntimeError::Entity(_)));
        assert_eq!(format!("{err}"), "entity error: no schedule registered");
    }

    /// `get_schedule_hash` is all zeros when no schedule is registered.
    #[test]
    fn schedule_hash_is_zero_when_unset() {
        let fx = Fixture::new();
        let ctx = fx.ctx();
        assert_eq!(get_schedule_hash(&ctx), [0u8; 32]);
    }

    /// `recover` on a fresh kernel returns the `fresh` report and
    /// leaves the sequence counter at 1 (watermarks 0 + 1).
    #[test]
    fn recover_on_fresh_kernel_returns_fresh_report() {
        let fx = Fixture::new();
        let ctx = fx.ctx();
        let report = recover(&ctx).expect("recover");
        assert_eq!(report.recovery_state, "fresh");
        assert_eq!(report.replayed_commands, 0);
        assert_eq!(report.unresolved_commands, 0);
        assert_eq!(fx.sequence.load(Ordering::SeqCst), 1);
    }

    /// `save_snapshot` writes a snapshot to the store and the
    /// checksum verifies on reload.
    #[test]
    fn save_snapshot_writes_and_verifies() {
        let fx = Fixture::new();
        let ctx = fx.ctx();
        save_snapshot(&ctx).expect("save snapshot");
        let loaded = fx.snapshot_store.load_latest().expect("load");
        let snap = loaded.expect("snapshot payload");
        assert!(snap.verify(), "snapshot checksum must verify");
    }

    /// `select_provider` returns a receipt from the configured selector.
    #[test]
    fn select_provider_returns_static_receipt() {
        let fx = Fixture::new();
        let ctx = fx.ctx();
        let receipt = select_provider(
            &ctx,
            &ProviderSelectionRequest {
                operation: "compile".into(),
                requested_provider: None,
                fallback_providers: vec!["cpu".into()],
            },
        );
        // The default static selector returns the first available provider.
        assert!(receipt.selected_provider.is_some());
    }

    /// `TickExecutorContext::with_expected_epoch` round-trips.
    #[test]
    fn tick_executor_context_holds_expected_epoch() {
        let ctx = TickExecutorContext::new("daemon-1", 7).with_expected_epoch(8);
        assert_eq!(ctx.instance_id, "daemon-1");
        assert_eq!(ctx.world_epoch, 7);
        assert_eq!(ctx.expected_epoch, Some(8));
    }

    /// `health` reports the empty-world state through the borrowed ctx.
    #[test]
    fn health_reports_fresh_state() {
        let fx = Fixture::new();
        let ctx = fx.ctx();
        let h = health(&ctx).expect("health");
        assert_eq!(h.entity_count, 0);
        assert_eq!(h.status, "running");
    }

    /// `capture_snapshot` produces a payload that round-trips through
    /// `WorldSnapshot::compute_checksum` and `verify`.
    #[test]
    fn capture_snapshot_round_trips_checksum() {
        let fx = Fixture::new();
        let ctx = fx.ctx();
        let snap = capture_snapshot(&ctx).expect("capture");
        assert!(snap.verify());
        assert_eq!(snap.payload.world_epoch, 0);
    }
}
