//! PRISM-PRODUCTION-HETEROGENEOUS-EXECUTOR-0001 — Heterogeneous execution kernel.
//!
//! Tokio actor that owns lane executors, scheduling policy, backpressure,
//! cancellation, and evidence receipts. One instance per loaded model runtime,
//! shared by all sessions using that model.
//!
//! # Architecture
//!
//! ```text
//!                          ┌─────────────────────┐
//!                          │ HeterogeneousExec.  │
//!   execute_epoch(…) ────► │  Handle (cloneable) │
//!                          └─────────┬───────────┘
//!                                    │ mpsc channel
//!                          ┌─────────▼───────────┐
//!                          │   Actor event loop   │
//!                          │                     │
//!   cmd_rx ────────────────┤  tokio::select!     │
//!   internal_completion_rx─┤                     │
//!                          │  ◄── LaneExecutors  │
//!                          └─────────────────────┘
//! ```
//!
//! Lane executors send [`WorkCompletion`] through an internal unbounded
//! channel. The actor converts each completion to a [`CompletionEvent`]
//! via [`work_completion_to_event`] for unified processing.

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::time::Instant;

use tokio::sync::{mpsc, oneshot};

use crate::backend::placement::ExecutionLane;
use crate::compilation::activation_abi::{ActivationAbi, SlotLeaseId};
use crate::compilation::phase_ir::PhaseId;
use crate::scheduling::accelerate_lane_executor::AccelerateLaneExecutor;
use crate::scheduling::ane_lane_executor::AneLaneExecutor;
use crate::scheduling::backpressure::{BackpressureController, BackpressureLevel};
use crate::scheduling::completion_bridge::work_completion_to_event;
use crate::scheduling::lane_capacity::{LaneCapacityConfig, LaneCapacityManager, LanePermit};
use crate::scheduling::lane_work::{
    next_work_id, BackendStatus, CompletionClock, LaneExecutor, LaneWorkRequest, MetalPipelineRef,
    StreamId, WorkCompletion, WorkId, WorkSubmission,
};
use crate::scheduling::metal_lane_executor::MetalLaneExecutor;
use crate::scheduling::receipt::{
    FallbackSummary, HeterogeneousExecutionReceipt, ReceiptCollector,
};
use crate::scheduling::scheduler_metrics::SchedulerMetrics;
use crate::scheduling::slot_lease_manager::SlotLeaseManager;
use crate::scheduling::tri_lane_orchestrator::{
    AdmissionStatus, PhaseVariant, PhaseVariantSet, VariantId,
};
use crate::scheduling::work_registry::{WorkKey, WorkRecord, WorkRegistry, WorkStatus};

// ── Error types ─────────────────────────────────────────────────────────

/// Errors produced by the heterogeneous executor.
#[derive(Debug)]
pub enum ExecutorError {
    /// The referenced session is not tracked by this executor.
    SessionNotFound(String),
    /// A lane executor refused the submission (non-retryable backend error).
    SubmitFailed(ExecutionLane, String),
    /// No admissible variant exists for the given phase.
    NoAdmissibleVariant(PhaseId),
    /// The executor is backpressured at or above the given level.
    Backpressure(BackpressureLevel),
    /// The executor has been shut down and cannot accept new work.
    Shutdown,
}

impl fmt::Display for ExecutorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExecutorError::SessionNotFound(s) => write!(f, "session {} not found", s),
            ExecutorError::SubmitFailed(lane, msg) => {
                write!(f, "lane {:?} submit failed: {}", lane, msg)
            }
            ExecutorError::NoAdmissibleVariant(id) => {
                write!(f, "no admissible variant for phase {:?}", id)
            }
            ExecutorError::Backpressure(level) => {
                write!(f, "backpressure: {:?}", level)
            }
            ExecutorError::Shutdown => write!(f, "executor shutting down"),
        }
    }
}

impl std::error::Error for ExecutorError {}

// ── Session submit request ──────────────────────────────────────────────

/// A complete epoch submission from an external caller (e.g. a session handler).
pub struct SessionSubmitRequest {
    /// Logical session identifier (durable across epochs).
    pub session_id: String,
    /// Model identity for routing / metric labelling.
    pub model_id: String,
    /// Monotonically increasing sequence counter within the session.
    pub sequence_id: u64,
    /// Ordered phase graph — one [`PhaseVariantSet`] per phase.
    pub phase_graph: Vec<PhaseVariantSet>,
    /// Dispatch priority for this epoch.
    pub priority: RequestPriority,
    /// External cancellation token; the caller may signal abort at any time.
    pub cancelled: Arc<std::sync::atomic::AtomicBool>,
}

/// Dispatch priority for an epoch.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestPriority {
    /// User-facing interactive request — minimal latency.
    Interactive,
    /// Standard throughput request.
    Normal,
    /// Low-priority batch work (speculative decode, re-rank).
    Low,
    /// Background maintenance (warmup, qualification).
    Background,
}

// ── Epoch execution result ──────────────────────────────────────────────

/// Result of executing one epoch.
#[derive(Debug)]
pub struct EpochExecutionResult {
    /// Handle to the terminal output activation (the last phase's output slot).
    pub terminal_output: OutputHandle,
    /// All receipts generated during this epoch.
    pub receipts: Vec<HeterogeneousExecutionReceipt>,
    /// Aggregated fallback summary for the epoch.
    pub fallback_summary: FallbackSummary,
}

/// Handle to the output of an epoch — either a slot lease or nothing.
#[derive(Debug)]
pub enum OutputHandle {
    /// The output is stored in the given slot lease.
    SlotLease(SlotLeaseId),
    /// No output was produced (e.g. cancelled before any phase completed).
    None,
}

// ── Commands for the actor ──────────────────────────────────────────────

/// Messages the actor processes one at a time.
#[allow(dead_code)]
enum ExecutorCommand {
    ExecuteEpoch {
        request: SessionSubmitRequest,
        response: oneshot::Sender<Result<EpochExecutionResult, ExecutorError>>,
    },
    Shutdown,
}

// ── HeterogeneousExecutor config ────────────────────────────────────────

/// Configuration for a [`HeterogeneousExecutor`] instance.
#[derive(Clone)]
pub struct ExecutorConfig {
    /// Capacity of the internal completion channel (number of queued
    /// [`WorkCompletion`]s before lane executor senders backpressure).
    pub completion_queue_capacity: usize,
    /// Capacity of the command channel from [`HeterogeneousExecutorHandle`].
    pub command_queue_capacity: usize,
    /// Maximum pending work items per session.
    pub per_session_pending_limit: usize,
    /// Maximum pending work items across all sessions.
    pub global_pending_limit: usize,
    /// Lane-specific capacity configuration.
    pub lane_capacity: LaneCapacityConfig,
}

impl Default for ExecutorConfig {
    fn default() -> Self {
        Self {
            completion_queue_capacity: 4096,
            command_queue_capacity: 2048,
            per_session_pending_limit: 128,
            global_pending_limit: 4096,
            lane_capacity: LaneCapacityConfig::default(),
        }
    }
}

// ── HeterogeneousExecutor — the actor ───────────────────────────────────

/// Tokio actor that owns all scheduling state for one model runtime.
///
/// The [`HeterogeneousExecutor::new`] constructor spawns the actor task
/// on the provided runtime handle and returns a cloneable
/// [`HeterogeneousExecutorHandle`] for external callers.
pub struct HeterogeneousExecutor {
    // ── Lane executors (one per available backend) ────────────────────
    metal: Option<MetalLaneExecutor>,
    ane: Option<AneLaneExecutor>,
    accelerate: Option<AccelerateLaneExecutor>,

    // ── Scheduling state ─────────────────────────────────────────────
    capacity: LaneCapacityManager,
    registry: WorkRegistry,
    slot_leases: SlotLeaseManager,
    backpressure: BackpressureController,

    // ── Completion plumbing ──────────────────────────────────────────
    /// Receiver for raw [`WorkCompletion`] from lane executors.
    internal_completion_rx: mpsc::UnboundedReceiver<WorkCompletion>,
    /// Sender cloned to each lane executor on submission.
    internal_completion_tx: mpsc::UnboundedSender<WorkCompletion>,

    // ── Bookkeeping ──────────────────────────────────────────────────
    /// Permits held per in-flight work item; released on completion.
    active_permits: HashMap<WorkId, LanePermit>,
    /// Maps session name → work key prefix (session_id + next sequence counter).
    session_state: HashMap<String, SessionState>,
    /// Maps session name → numeric stream identifier for lane executors.
    session_streams: HashMap<String, StreamId>,
    /// Monotonic stream-id allocator.
    next_stream_id: u64,
    /// Monotonic slot-id allocator for the slot lease manager.
    next_slot_id: u64,

    // ── Evidence ─────────────────────────────────────────────────────
    receipts: ReceiptCollector,
    metrics: Arc<SchedulerMetrics>,

    // ── Actor channel ────────────────────────────────────────────────
    cmd_rx: Option<mpsc::Receiver<ExecutorCommand>>,

    // ── Config ───────────────────────────────────────────────────────
    #[allow(dead_code)]
    config: ExecutorConfig,
    #[allow(dead_code)]
    model_identity: String,
}

/// Per-session mutable state.
struct SessionState {
    /// Next sequence number for this session.
    next_sequence: u64,
}

// ── Public handle ──────────────────────────────────────────────────────

/// Cloneable, `Send` handle for submitting work to a
/// [`HeterogeneousExecutor`] actor.
#[derive(Clone)]
pub struct HeterogeneousExecutorHandle {
    cmd_tx: mpsc::Sender<ExecutorCommand>,
    metrics: Arc<SchedulerMetrics>,
}

impl HeterogeneousExecutor {
    /// Create a new executor and spawn its actor task on `runtime_handle`.
    ///
    /// Returns a [`HeterogeneousExecutorHandle`] that callers clone and use
    /// to submit epochs.  The actor task runs for the lifetime of the runtime.
    pub fn new(
        model_identity: &str,
        metal: Option<MetalLaneExecutor>,
        ane: Option<AneLaneExecutor>,
        accelerate: Option<AccelerateLaneExecutor>,
        #[allow(dead_code)] config: ExecutorConfig,
        #[allow(dead_code)] runtime_handle: tokio::runtime::Handle,
    ) -> HeterogeneousExecutorHandle {
        let (cmd_tx, cmd_rx) = mpsc::channel(config.command_queue_capacity);
        let (internal_completion_tx, internal_completion_rx) = mpsc::unbounded_channel();
        let metrics = Arc::new(SchedulerMetrics::new());

        let mut executor = Self {
            metal,
            ane,
            accelerate,
            capacity: LaneCapacityManager::new(config.lane_capacity.clone()),
            registry: WorkRegistry::new(),
            slot_leases: SlotLeaseManager::new(),
            backpressure: BackpressureController::new(),
            internal_completion_rx,
            internal_completion_tx,
            active_permits: HashMap::new(),
            session_state: HashMap::new(),
            session_streams: HashMap::new(),
            next_stream_id: 1,
            next_slot_id: 1,
            receipts: ReceiptCollector::new(10_000),
            metrics: metrics.clone(),
            cmd_rx: Some(cmd_rx),
            config,
            model_identity: model_identity.to_string(),
        };

        runtime_handle.spawn(async move {
            executor.run().await;
        });

        HeterogeneousExecutorHandle { cmd_tx, metrics }
    }

    // ── Actor event loop ──────────────────────────────────────────────

    /// Main event loop: select on commands and internal completions.
    async fn run(&mut self) {
        let mut cmd_rx = self.cmd_rx.take().expect("cmd_rx already taken");

        loop {
            tokio::select! {
                cmd = cmd_rx.recv() => {
                    match cmd {
                        Some(ExecutorCommand::ExecuteEpoch { request, response }) => {
                            let result = self.handle_execute_epoch(request).await;
                            let _ = response.send(result);
                        }
                        Some(ExecutorCommand::Shutdown) | None => break,
                    }
                }
                completion = self.internal_completion_rx.recv() => {
                    match completion {
                        Some(wc) => self.handle_work_completion(wc),
                        None => break, // all senders dropped — shut down
                    }
                }
            }
        }
    }

    // ── Epoch dispatch ────────────────────────────────────────────────

    /// Process a complete epoch submission: select variants, reserve
    /// capacity & slots, submit to lane executors.
    async fn handle_execute_epoch(
        &mut self,
        request: SessionSubmitRequest,
    ) -> Result<EpochExecutionResult, ExecutorError> {
        // ── 1. Backpressure gate ─────────────────────────────────────
        let bp_level = self.backpressure.level();
        if bp_level >= BackpressureLevel::Severe {
            return Err(ExecutorError::Backpressure(bp_level));
        }

        // ── 2. Resolve session identity ──────────────────────────────
        let stream_id = self.resolve_stream_id(&request.session_id);
        let session_seq = self
            .session_state
            .entry(request.session_id.clone())
            .or_insert_with(|| SessionState { next_sequence: 0 });
        let epoch_sequence = session_seq.next_sequence;
        session_seq.next_sequence += 1;

        // ── 3. Walk the phase graph ──────────────────────────────────
        let fallback_summary = FallbackSummary::default();
        let mut last_output_lease: Option<SlotLeaseId> = None;

        for phase_set in &request.phase_graph {
            // Honour external cancellation before each phase.
            if request.cancelled.load(std::sync::atomic::Ordering::Relaxed) {
                break;
            }

            // Select best admissible variant.
            let (variant_idx, variant) = self.select_best_variant(phase_set)?;

            // Reserve lane capacity.
            let permit = self
                .capacity
                .try_acquire(variant.lane, &request.session_id)
                .ok_or_else(|| ExecutorError::Backpressure(self.backpressure.level()))?;

            // Allocate a work id.
            let work_id = next_work_id();

            // Acquire output slot lease.
            let slot_id = self.next_slot_id;
            self.next_slot_id += 1;
            let output_lease = self
                .slot_leases
                .acquire_write(slot_id, work_id, &request.session_id, phase_set.phase_id)
                .map_err(|e| ExecutorError::SubmitFailed(variant.lane, e))?;

            // Build the completion clock.
            let submit_ns = HeterogeneousExecutionReceipt::now_ns();
            let clock = CompletionClock::new(submit_ns);

            // Build the lane work request.
            let lane_request = LaneWorkRequest {
                work_id,
                session_id: stream_id,
                epoch_id: epoch_sequence,
                phase_id: phase_set.phase_id,
                variant_id: variant_idx as VariantId,
                lane: variant.lane,
                input_slots: vec![],
                output_slot: output_lease,
                input_abi: variant.input_abi.clone(),
                output_abi: variant.output_abi.clone(),
                artifact_key: variant.artifact_key.clone(),
                metal_pipeline: variant
                    .metal_pipeline
                    .as_ref()
                    .map(|fn_name| MetalPipelineRef {
                        function_name: fn_name.clone(),
                        pipeline_digest: String::new(),
                    }),
                completion_clock: clock,
            };

            // Register in the work registry.
            let _ = self.registry.register(WorkRecord {
                work_id,
                lane: variant.lane,
                session_id: request.session_id.clone(),
                phase_id: phase_set.phase_id,
                input_slots: vec![],
                output_slot: output_lease,
                status: WorkStatus::Created,
                created_at: Instant::now(),
                submitted_at: None,
                completed_at: None,
                attempt: 0,
                backend_timing: None,
            });

            // Transition: Created → Ready.
            let _ = self.registry.transition(work_id, WorkStatus::Ready);

            // Submit to the appropriate lane executor.
            let completion_tx = self.internal_completion_tx.clone();
            let _submission = self.submit_to_lane(variant.lane, lane_request, completion_tx)?;

            // Transition: Ready → Submitted (skip intermediate states for simplicity).
            let _ = self.registry.transition(work_id, WorkStatus::Submitted);

            // Track the permit for later release.
            self.active_permits.insert(work_id, permit);

            // Update metrics.
            self.increment_lane_in_flight(variant.lane);

            last_output_lease = Some(output_lease);
        }

        // ── 4. Drain any ready completions ───────────────────────────
        while let Ok(wc) = self.internal_completion_rx.try_recv() {
            self.handle_work_completion(wc);
        }

        // ── 5. Assemble result ───────────────────────────────────────
        let terminal_output = match last_output_lease {
            Some(lease) => OutputHandle::SlotLease(lease),
            None => OutputHandle::None,
        };

        Ok(EpochExecutionResult {
            terminal_output,
            receipts: self.receipts.drain(),
            fallback_summary,
        })
    }

    // ── Variant selection ─────────────────────────────────────────────

    /// Score each admissible variant and return the best one by index.
    fn select_best_variant<'a>(
        &self,
        phase_set: &'a PhaseVariantSet,
    ) -> Result<(usize, &'a PhaseVariant), ExecutorError> {
        let mut best_idx: Option<usize> = None;
        let mut best_score: f64 = f64::NEG_INFINITY;

        for (i, variant) in phase_set.variants.iter().enumerate() {
            if !matches!(variant.admission, AdmissionStatus::Admitted) {
                continue;
            }

            // Simple cost heuristic: prefer lower execution cost,
            // penalise qualification risk and queue depth.
            let exec_cost = variant.cost_estimate.execution_ns as f64;
            let risk_penalty = variant.cost_estimate.qualification_risk as f64 * 100.0;
            let queue_penalty = 50.0; // fixed penalty per queued item (avoid QueueEntry lookup)

            // Negate so that higher score = better.
            let score = -(exec_cost + risk_penalty + queue_penalty);

            if score > best_score {
                best_score = score;
                best_idx = Some(i);
            }
        }

        let idx = best_idx.ok_or(ExecutorError::NoAdmissibleVariant(phase_set.phase_id))?;
        Ok((idx, &phase_set.variants[idx]))
    }

    // ── Lane dispatch ─────────────────────────────────────────────────

    /// Route a work request to the correct lane executor.
    fn submit_to_lane(
        &mut self,
        lane: ExecutionLane,
        request: LaneWorkRequest,
        completion_tx: mpsc::UnboundedSender<WorkCompletion>,
    ) -> Result<WorkSubmission, ExecutorError> {
        match lane {
            ExecutionLane::MlxGpu => {
                let ex = self.metal.as_mut().ok_or_else(|| {
                    ExecutorError::SubmitFailed(lane, "Metal executor not available".into())
                })?;
                ex.submit(request, completion_tx)
                    .map_err(|e| ExecutorError::SubmitFailed(lane, e.to_string()))
            }
            ExecutionLane::CoreMlAne => {
                let ex = self.ane.as_mut().ok_or_else(|| {
                    ExecutorError::SubmitFailed(lane, "ANE executor not available".into())
                })?;
                ex.submit(request, completion_tx)
                    .map_err(|e| ExecutorError::SubmitFailed(lane, e.to_string()))
            }
            ExecutionLane::AccelerateCpu => {
                let ex = self.accelerate.as_mut().ok_or_else(|| {
                    ExecutorError::SubmitFailed(lane, "Accelerate executor not available".into())
                })?;
                ex.submit(request, completion_tx)
                    .map_err(|e| ExecutorError::SubmitFailed(lane, e.to_string()))
            }
            other => Err(ExecutorError::SubmitFailed(
                other,
                "unsupported lane".into(),
            )),
        }
    }

    // ── Completion processing ─────────────────────────────────────────

    /// Process a raw [`WorkCompletion`] from a lane executor.
    ///
    /// Converts to a [`CompletionEvent`] via the completion bridge and
    /// updates registry state, slot leases, capacity, and metrics.
    fn handle_work_completion(&mut self, wc: WorkCompletion) {
        let received_at = Instant::now();

        // Look up the work record to build a proper WorkKey.
        let work_key = self.build_work_key(&wc);

        // Convert to CompletionEvent via the bridge.
        let event = work_completion_to_event(wc.clone(), work_key, received_at);

        // ── Update work registry state ─────────────────────────────
        let new_status = match &event.backend_status {
            BackendStatus::Completed => WorkStatus::Completed,
            BackendStatus::Failed(_) => WorkStatus::ExecutionFailed,
            BackendStatus::Cancelled => WorkStatus::CancelledBeforeSubmit,
        };
        let _ = self.registry.transition(wc.work_id, new_status);

        // Only mark output ready for successful completions.
        if matches!(event.backend_status, BackendStatus::Completed) {
            let _ = self
                .registry
                .transition(wc.work_id, WorkStatus::OutputReady);
            let _ = self.slot_leases.mark_output_ready(wc.output_slot);
        }

        // ── Release lane capacity ──────────────────────────────────
        let session_for_release = self
            .registry
            .get(wc.work_id)
            .map(|r| r.session_id.clone())
            .unwrap_or_default();
        if let Some(permit) = self.active_permits.remove(&wc.work_id) {
            self.capacity.release(permit, &session_for_release);
            self.decrement_lane_in_flight(wc.lane);
        }

        // ── Build and record receipt ───────────────────────────────
        let receipt = HeterogeneousExecutionReceipt::from_timestamps(
            event.work_id,
            event.work_key,
            event.lane,
            0,      // attempt
            None,   // artifact_key
            None,   // qualification_key
            vec![], // input_slots
            event.output_lease,
            ActivationAbi::default_abi(),
            ActivationAbi::default_abi(),
            wall_instant_to_ns(event.submitted_at),
            wall_instant_to_ns(event.backend_started_at),
            wall_instant_to_ns(event.backend_ended_at),
            wall_instant_to_ns(event.callback_received_at),
            false, // fallback_used
            None,  // fallback_reason
            event.status,
            event.timing_quality,
        );
        self.receipts.record(receipt);

        // ── Backpressure feedback ──────────────────────────────────
        // TODO: feed backpressure events based on completion timing.
    }

    /// Construct a [`WorkKey`] from a [`WorkCompletion`] by consulting
    /// the work registry for the associated session and epoch.
    fn build_work_key(&self, wc: &WorkCompletion) -> WorkKey {
        if let Some(record) = self.registry.get(wc.work_id) {
            WorkKey {
                session_id: record.session_id.clone(),
                request_id: String::new(),
                sequence_id: 0,
                epoch_id: 0,
                phase_id: record.phase_id,
                attempt: record.attempt,
            }
        } else {
            WorkKey {
                session_id: String::new(),
                request_id: String::new(),
                sequence_id: 0,
                epoch_id: 0,
                phase_id: wc.phase_id,
                attempt: 0,
            }
        }
    }

    // ── Session bookkeeping ───────────────────────────────────────────

    /// Resolve (or allocate) a [`StreamId`] for the given session name.
    fn resolve_stream_id(&mut self, session_name: &str) -> StreamId {
        *self
            .session_streams
            .entry(session_name.to_string())
            .or_insert_with(|| {
                let id = StreamId(self.next_stream_id);
                self.next_stream_id += 1;
                id
            })
    }

    // ── Metrics helpers ───────────────────────────────────────────────

    fn increment_lane_in_flight(&self, lane: ExecutionLane) {
        match lane {
            ExecutionLane::MlxGpu => {
                self.metrics
                    .metal_in_flight
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            ExecutionLane::CoreMlAne => {
                self.metrics
                    .ane_in_flight
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            ExecutionLane::AccelerateCpu => {
                self.metrics
                    .accelerate_in_flight
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }
            _ => {}
        }
    }

    fn decrement_lane_in_flight(&self, lane: ExecutionLane) {
        match lane {
            ExecutionLane::MlxGpu => {
                self.metrics
                    .metal_in_flight
                    .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            }
            ExecutionLane::CoreMlAne => {
                self.metrics
                    .ane_in_flight
                    .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            }
            ExecutionLane::AccelerateCpu => {
                self.metrics
                    .accelerate_in_flight
                    .fetch_sub(1, std::sync::atomic::Ordering::Relaxed);
            }
            _ => {}
        }
    }
}

// ── Public API (HeterogeneousExecutorHandle) ────────────────────────────

impl HeterogeneousExecutorHandle {
    /// Submit an epoch for execution.
    ///
    /// The future resolves when the epoch has been fully dispatched to the
    /// lane executors (not when the execution completes — that is signalled
    /// via the completion channel).
    pub async fn execute_epoch(
        &self,
        request: SessionSubmitRequest,
    ) -> Result<EpochExecutionResult, ExecutorError> {
        let (tx, rx) = oneshot::channel();
        self.cmd_tx
            .send(ExecutorCommand::ExecuteEpoch {
                request,
                response: tx,
            })
            .await
            .map_err(|_| ExecutorError::Shutdown)?;
        rx.await.map_err(|_| ExecutorError::Shutdown)?
    }

    /// Return a shared reference to the executor's metrics.
    pub fn metrics(&self) -> Arc<SchedulerMetrics> {
        self.metrics.clone()
    }
}

impl fmt::Debug for HeterogeneousExecutorHandle {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("HeterogeneousExecutorHandle").finish()
    }
}

// ── Utility free functions ─────────────────────────────────────────────

/// Convert a `std::time::Instant` to nanoseconds since `UNIX_EPOCH`.
///
/// Uses `SystemTime::now()` and subtracts the delta from the instant.
/// This is an approximation that assumes the `Instant` clock and
/// `SystemTime` clock are the same monotonic source (true on Darwin/Linux).
fn wall_instant_to_ns(instant: Instant) -> u64 {
    let now_instant = Instant::now();
    let now_system = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;
    if instant <= now_instant {
        let delta_ns = now_instant.duration_since(instant).as_nanos() as u64;
        now_system.saturating_sub(delta_ns)
    } else {
        let delta_ns = instant.duration_since(now_instant).as_nanos() as u64;
        now_system.saturating_add(delta_ns)
    }
}

// ── Fallback ActivationAbi for receipt construction ─────────────────────

/// Extension trait to provide a default/empty [`ActivationAbi`] for
/// places where the real ABI is not available (e.g. receipt construction
/// from a bare completion event).
trait ActivationAbiDefault {
    fn default_abi() -> Self;
}

impl ActivationAbiDefault for ActivationAbi {
    fn default_abi() -> Self {
        use crate::compilation::activation_abi::MetalOnlyParams;
        use crate::compilation::phase_ir::TensorDtype;
        ActivationAbi::MetalOnly(MetalOnlyParams {
            name: String::new(),
            dtype: TensorDtype::Float16,
            byte_count: 0,
        })
    }
}
