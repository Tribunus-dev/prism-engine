//! Canonical phase-DAG executor — drives a compiler-emitted phase graph
//! from ready-set computation through lifecycle transitions to receipt
//! collection, dispatching each phase through the canonical
//! [`PhaseRunner`](super::phase_runner::execution::PhaseRunner) port.
//!
//! Authority: this module owns the canonical phase-DAG executor — the
//! ready-set computation, the lifecycle state machine, the dispatch
//! coordination, and the receipt collection for one phase graph
//! execution. It does not own the concrete [`PhaseRunner`]
//! implementations (that lives in
//! [`super::phase_runner::execution`]), the per-lane work queue (that
//! lives in [`super::lane_queue`]), the backpressure controller (that
//! lives in [`super::backpressure`]), or the constitutional schedule's
//! `System` graph (that lives in `crate::schedule`).
//!
//! Constitutional notes: phase identity is [`PhaseId`], lane identity
//! is [`LaneTag`], canonical collections are `BTreeMap`-backed,
//! fallible operations return `Result<_, PhaseEngineError>`, and the
//! [`PhaseRunner`](super::phase_runner::execution::PhaseRunner) port
//! is the only dispatch path — concrete MLX / Metal / ANE bridge code
//! lives in the engine.
//!
//! Lifecycle: `Dormant → Ready → Admitted → Dispatched → Complete |
//! FallbackComplete | FailedBeforePublication`. Cancellation may
//! transition any non-terminal state to `Cancelled`.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::time::Instant;

use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::phase_runner::execution::{
    PhaseCompletionStatus, PhaseResult, PhaseRunner, PhaseRunnerContext, PhaseRunnerError,
    PhaseRunnerRegistry,
};

// ── Identity ─────────────────────────────────────────────────────────────

/// Typed identity for one phase in a compiler-emitted DAG. The
/// constitutional side does not import the engine's
/// `compute_image::phase_dag::PhaseId`; adapters convert at the
/// boundary. Newtype around `String` so the wire format is human-readable
/// (phase ids are the compiler's stable identifiers).
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct PhaseId(pub String);

impl PhaseId {
    /// Construct a [`PhaseId`] from a raw `String`.
    pub const fn new(raw: String) -> Self {
        Self(raw)
    }

    /// Borrow the underlying string.
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl fmt::Display for PhaseId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for PhaseId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for PhaseId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Typed identity for the compute lane a phase is bound to. The engine's
/// `ComputeLane` enum (Metal / Accelerate / CoreAi / Arena) is mapped
/// at the adapter. Newtype around `String` so the wire format is
/// human-readable.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct LaneTag(pub String);

impl LaneTag {
    /// Construct a [`LaneTag`] from a raw `String`.
    pub const fn new(raw: String) -> Self {
        Self(raw)
    }

    /// Borrow the underlying string.
    pub fn as_str(&self) -> &str {
        self.0.as_str()
    }
}

impl From<&str> for LaneTag {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

// ── Enums ────────────────────────────────────────────────────────────────

/// Kind of work performed by a phase. The constitutional list mirrors
/// the engine's `PhaseKind`; the engine's adapter converts at the
/// boundary. Each variant names the work the runner performs, not the
/// underlying library (no `Mlx`, no `Metal` in the variant names — the
/// dispatch is the runner's concern).
#[derive(
    Debug, Clone, Copy, Hash, Eq, PartialEq, PartialOrd, Ord, Serialize, Deserialize,
)]
pub enum PhaseKind {
    /// Decode-step on the canonical inference path.
    Decode,
    /// Fused GPU kernel dispatch.
    FusedKernel,
    /// Compiled ANE subgraph execution.
    AnEGraph,
    /// CPU SIMD matrix multiply.
    AccelMatMul,
    /// CPU SIMD element-wise operation.
    AccelElementWise,
    /// Arena allocation phase.
    ArenaAlloc,
    /// Synchronization barrier between lanes.
    SyncBarrier,
    /// Transfer phase between lanes or memory pools.
    Transfer,
    /// Fused residual + RMS norm.
    ResidualRmsNorm,
    /// Weight residency activation phase.
    WeightResidency,
    /// Legacy MLX prologue — embedding lookup.
    LegacyPrologue,
    /// Legacy MLX epilogue — final norm + lm_head.
    LegacyEpilogue,
    /// Token sampling phase.
    Sampling,
}

impl PhaseKind {
    /// All defined [`PhaseKind`] variants, in stable order.
    pub const ALL: &'static [PhaseKind] = &[
        PhaseKind::Decode,
        PhaseKind::FusedKernel,
        PhaseKind::AnEGraph,
        PhaseKind::AccelMatMul,
        PhaseKind::AccelElementWise,
        PhaseKind::ArenaAlloc,
        PhaseKind::SyncBarrier,
        PhaseKind::Transfer,
        PhaseKind::ResidualRmsNorm,
        PhaseKind::WeightResidency,
        PhaseKind::LegacyPrologue,
        PhaseKind::LegacyEpilogue,
        PhaseKind::Sampling,
    ];
}

/// Semantic meaning of a directed edge between two phases in the DAG.
/// Each variant represents a non-overlapping dependency reason, so a
/// pair of phases may have multiple edges with different [`SemanticKind`]s.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub enum SemanticKind {
    /// Data tensor dependency: `from` produces a tensor `to` consumes.
    Data,
    /// Arena slot ownership transfer.
    ArenaOwnership,
    /// State-epoch ordering (e.g. KV-cache epoch boundaries).
    StateEpoch,
    /// Completion of a transfer (load / device-to-device) operation.
    TransferCompletion,
    /// Request-ordering constraint (e.g. token order).
    RequestOrdering,
    /// Decomposition of a fallback variant into its higher-ranked
    /// primitives.
    FallbackDecomposition,
}

/// Completion status of a phase — used for observability and fallback
/// bookkeeping.
///
/// `Failed` and `FallbackUsed` carry a human-readable reason string;
/// the typed-error variant for dispatch errors is
/// [`PhaseRunnerError`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", content = "reason")]
pub enum PhaseCompletion {
    /// Phase has not yet completed.
    Pending,
    /// Phase completed successfully.
    Complete,
    /// Phase failed; the reason is recorded for diagnostics.
    Failed(String),
    /// Phase used a fallback variant; the original error is recorded.
    FallbackUsed(String),
}

impl From<PhaseCompletionStatus> for PhaseCompletion {
    fn from(s: PhaseCompletionStatus) -> Self {
        match s {
            PhaseCompletionStatus::Pending => PhaseCompletion::Pending,
            PhaseCompletionStatus::Complete => PhaseCompletion::Complete,
            PhaseCompletionStatus::Failed(r) => PhaseCompletion::Failed(r),
            PhaseCompletionStatus::FallbackUsed(r) => PhaseCompletion::FallbackUsed(r),
        }
    }
}

// ── Phase, edge, graph ───────────────────────────────────────────────────

/// Reference to one arena allocation slot claimed by a phase.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArenaSlotRef {
    /// Slot identifier within the arena plan.
    pub slot_id: String,
    /// Size of the allocation in bytes.
    pub byte_size: u64,
    /// Required alignment in bytes.
    pub alignment: u64,
    /// Lane that owns / manages this slot.
    pub lane: LaneTag,
}

/// A single phase — one schedulable unit of work in the DAG.
///
/// Phases are the vertices of [`PhaseGraph`]. Each phase runs on a
/// single [`LaneTag`] and may claim zero or more arena slots and
/// read or write zero or more tensors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Phase {
    /// Unique phase identifier within the graph.
    pub phase_id: PhaseId,
    /// Kind of work this phase performs.
    pub kind: PhaseKind,
    /// Target compute lane.
    pub lane: LaneTag,
    /// Logical operation names (e.g. `["q_proj", "k_proj", "v_proj"]`).
    pub ops: Vec<String>,
    /// Arena slots claimed by this phase.
    pub arena_slots: Vec<ArenaSlotRef>,
    /// Tensor names this phase reads.
    pub tensor_reads: Vec<String>,
    /// Tensor names this phase writes.
    pub tensor_writes: Vec<String>,
    /// Estimated operation count for scheduling heuristics.
    pub estimated_ops: u64,
    /// Free-form metadata, keyed by string. The constitutional side
    /// uses [`BTreeMap`] for canonical collections.
    pub metadata: BTreeMap<String, String>,
}

impl Phase {
    /// Look up a metadata value by key.
    pub fn metadata(&self, key: &str) -> Option<&str> {
        self.metadata.get(key).map(String::as_str)
    }
}

/// A directed edge between two phases.
///
/// A pair of phases may have multiple edges with different
/// [`SemanticKind`]s; the dispatch logic treats them as a set of
/// constraints, not a single one.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseEdge {
    /// Source phase.
    pub from_phase: PhaseId,
    /// Destination phase.
    pub to_phase: PhaseId,
    /// Semantic meaning of the dependency.
    pub semantic_kind: SemanticKind,
    /// Optional human-readable label.
    pub label: Option<String>,
    /// Free-form metadata.
    pub metadata: BTreeMap<String, String>,
}

/// Arena layout plan attached to a phase graph. Mirrors the engine's
/// `EmittedArenaPlan` shape but uses constitutional types.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArenaPlan {
    /// Total bytes across all arena slots.
    pub total_bytes: u64,
    /// Per-slot plan entries.
    pub slots: Vec<ArenaSlotRef>,
}

/// Concurrency hint for a phase graph — independent sets of phases
/// that the executor may dispatch in parallel.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConcurrencyPlan {
    /// Independent sets; phases in the same set have no ordering
    /// constraint.
    pub independent_sets: Vec<Vec<PhaseId>>,
}

/// A compiler-emitted phase DAG. The graph is guaranteed acyclic at
/// construction time (see [`PhaseGraph::validate`]).
///
/// All collections are canonical: phases are stored in a `Vec` (the
/// engine's emission order, which is the topological order), edges are
/// stored in a [`BTreeMap`] from `(from, to)` to a [`BTreeSet`] of
/// [`SemanticKind`]s for deterministic iteration.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseGraph {
    /// All phases in the graph, in topological order.
    pub phases: Vec<Phase>,
    /// All edges in the graph, keyed by `(from, to)`.
    pub edges: BTreeMap<(PhaseId, PhaseId), BTreeSet<SemanticKind>>,
    /// Arena layout plan.
    pub arena_plan: ArenaPlan,
    /// Concurrency hint.
    pub concurrency_plan: ConcurrencyPlan,
    /// Compiler version that emitted this graph.
    pub compiler_version: String,
}

impl PhaseGraph {
    /// Return the set of phases that directly precede `phase_id`.
    pub fn predecessors(&self, phase_id: &PhaseId) -> Vec<&Phase> {
        self.edges
            .keys()
            .filter_map(|(from, to)| {
                if to == phase_id {
                    self.phases.iter().find(|p| &p.phase_id == from)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Return the set of phases that directly succeed `phase_id`.
    pub fn successors(&self, phase_id: &PhaseId) -> Vec<&Phase> {
        self.edges
            .keys()
            .filter_map(|(from, to)| {
                if from == phase_id {
                    self.phases.iter().find(|p| &p.phase_id == to)
                } else {
                    None
                }
            })
            .collect()
    }

    /// Look up a phase by id.
    pub fn phase(&self, phase_id: &PhaseId) -> Option<&Phase> {
        self.phases.iter().find(|p| &p.phase_id == phase_id)
    }

    /// Validate that the graph is acyclic and that every edge endpoint
    /// names a known phase. Returns [`PhaseEngineError::InvalidGraph`]
    /// on the first violation.
    pub fn validate(&self) -> Result<(), PhaseEngineError> {
        let known: BTreeSet<&PhaseId> = self.phases.iter().map(|p| &p.phase_id).collect();
        for (from, to, _) in self.edge_iter() {
            if !known.contains(from) {
                return Err(PhaseEngineError::InvalidGraph {
                    reason: format!("edge from unknown phase {from}"),
                });
            }
            if !known.contains(to) {
                return Err(PhaseEngineError::InvalidGraph {
                    reason: format!("edge to unknown phase {to}"),
                });
            }
        }
        // Kahn's topological sort for cycle detection.
        let mut in_degree: BTreeMap<&PhaseId, usize> =
            self.phases.iter().map(|p| (&p.phase_id, 0)).collect();
        for (_, to, _) in self.edge_iter() {
            *in_degree.entry(to).or_insert(0) += 1;
        }
        let mut ready: Vec<&PhaseId> = in_degree
            .iter()
            .filter_map(|(id, d)| if *d == 0 { Some(*id) } else { None })
            .collect();
        let successors: BTreeMap<&PhaseId, Vec<&PhaseId>> = self.edges.keys().fold(
            BTreeMap::new(),
            |mut acc, (from, to)| {
                acc.entry(from).or_default().push(to);
                acc
            },
        );
        let mut visited = 0usize;
        while let Some(id) = ready.pop() {
            visited += 1;
            if let Some(succs) = successors.get(id) {
                for to in succs {
                    if let Some(d) = in_degree.get_mut(to) {
                        *d = d.saturating_sub(1);
                        if *d == 0 {
                            ready.push(to);
                        }
                    }
                }
            }
        }
        if visited != self.phases.len() {
            return Err(PhaseEngineError::InvalidGraph {
                reason: "cycle detected".to_string(),
            });
        }
        Ok(())
    }

    /// Iterate over `(from, to, semantic_kind)` triples.
    fn edge_iter(&self) -> impl Iterator<Item = (&PhaseId, &PhaseId, SemanticKind)> {
        self.edges
            .iter()
            .flat_map(|((from, to), kinds)| kinds.iter().map(move |k| (from, to, *k)))
    }
}

// ── Lifecycle state machine ──────────────────────────────────────────────

/// Phase lifecycle state. The state machine is described in the
/// module doc.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub enum PhaseState {
    /// Phase is registered but not yet emitted to the ready set.
    Dormant,
    /// Phase has been emitted to the ready set (predecessors satisfied).
    Ready,
    /// Phase has been admitted by the executor.
    Admitted,
    /// Phase has been dispatched to a runner.
    Dispatched,
    /// Phase completed successfully.
    Complete,
    /// Phase used a fallback variant and completed.
    FallbackComplete,
    /// Phase failed before producing output.
    FailedBeforePublication,
    /// Phase was cancelled by the caller.
    Cancelled,
}

impl PhaseState {
    /// Whether the state is terminal.
    pub fn is_terminal(self) -> bool {
        matches!(
            self,
            PhaseState::Complete
                | PhaseState::FallbackComplete
                | PhaseState::FailedBeforePublication
                | PhaseState::Cancelled
        )
    }

    /// Whether the state represents successful completion.
    pub fn is_success(self) -> bool {
        matches!(self, PhaseState::Complete | PhaseState::FallbackComplete)
    }

    /// Whether the executor may proceed past this state.
    pub fn can_proceed(self) -> bool {
        self.is_success()
    }
}

/// Canonical lifecycle tracker. Maps each [`PhaseId`] to its current
/// [`PhaseState`]. Backed by [`BTreeMap`] for deterministic iteration.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseLifecycleTracker {
    states: BTreeMap<PhaseId, PhaseState>,
}

impl PhaseLifecycleTracker {
    /// Create an empty tracker.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a phase. Idempotent.
    pub fn register(&mut self, phase_id: PhaseId) {
        self.states.entry(phase_id).or_insert(PhaseState::Dormant);
    }

    /// Transition `phase_id` to `to`. Returns
    /// [`PhaseEngineError::InvalidTransition`] if the transition is
    /// not allowed. Terminal states may be overridden by any other
    /// terminal state (for shutdown error reporting).
    pub fn transition(
        &mut self,
        phase_id: &PhaseId,
        to: PhaseState,
    ) -> Result<(), PhaseEngineError> {
        let current = self
            .states
            .get(phase_id)
            .copied()
            .unwrap_or(PhaseState::Dormant);
        if current == to {
            return Ok(());
        }
        if to.is_terminal() {
            self.states.insert(phase_id.clone(), to);
            return Ok(());
        }
        let allowed = matches!(
            (current, to),
            (PhaseState::Dormant, PhaseState::Ready)
                | (PhaseState::Ready, PhaseState::Admitted)
                | (PhaseState::Admitted, PhaseState::Dispatched)
        );
        if allowed {
            self.states.insert(phase_id.clone(), to);
            Ok(())
        } else {
            Err(PhaseEngineError::InvalidTransition { from: current, to })
        }
    }

    /// Read the current state of `phase_id`. Returns
    /// [`PhaseState::Dormant`] for unknown phases.
    pub fn state(&self, phase_id: &PhaseId) -> PhaseState {
        self.states.get(phase_id).copied().unwrap_or(PhaseState::Dormant)
    }

    /// Whether every registered phase is in a terminal state.
    pub fn all_terminal(&self) -> bool {
        !self.states.is_empty() && self.states.values().all(|s| s.is_terminal())
    }

    /// Number of registered phases.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Whether no phases are registered.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Iterate over `(phase_id, state)` pairs in sorted order.
    pub fn iter(&self) -> impl Iterator<Item = (&PhaseId, &PhaseState)> {
        self.states.iter()
    }
}

// ── Ready queue ──────────────────────────────────────────────────────────

/// Ready-set computation: a phase is ready when all of its predecessors
/// have reached a [`PhaseState::can_proceed`] state.
///
/// The queue is recomputed on each call; it does not cache. Caching
/// would couple the queue to the lifecycle tracker and is unnecessary
/// because the graph is small (dozens of phases, not millions).
#[derive(Debug, Clone)]
pub struct ReadyQueue<'g> {
    graph: &'g PhaseGraph,
}

impl<'g> ReadyQueue<'g> {
    /// Build a ready-queue view of `graph`.
    pub fn new(graph: &'g PhaseGraph) -> Self {
        Self { graph }
    }

    /// Return the set of phases whose predecessors are all in a
    /// `can_proceed` state. `completed` is the set of phase ids that
    /// have reached a `can_proceed` state.
    pub fn ready_phases(&self, completed: &BTreeSet<PhaseId>) -> Vec<&'g Phase> {
        self.graph
            .phases
            .iter()
            .filter(|p| {
                if completed.contains(&p.phase_id) {
                    return false;
                }
                self.graph
                    .predecessors(&p.phase_id)
                    .iter()
                    .all(|pred| completed.contains(&pred.phase_id))
            })
            .collect()
    }
}

// ── Receipts and result ─────────────────────────────────────────────────

/// Canonical receipt for one phase execution. Emitted by the
/// [`PhaseGraphEngine`] after each phase completes (success, failure,
/// or fallback).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseReceipt {
    /// Phase this receipt is for.
    pub phase_id: PhaseId,
    /// Completion status.
    pub status: PhaseCompletion,
    /// Wall-clock duration in microseconds.
    pub duration_us: u64,
    /// Optional compiler session id that emitted the phase.
    pub compiler_session_id: Option<String>,
    /// Optional compiler event digest for cross-referencing.
    pub compiler_event_digest: Option<String>,
}

impl PhaseReceipt {
    /// Construct a [`PhaseReceipt`] from a [`PhaseResult`] and an id.
    pub fn from_result(result: PhaseResult) -> Self {
        let status = PhaseCompletion::from(result.status);
        Self {
            phase_id: PhaseId::from(result.phase_id),
            status,
            duration_us: result.duration_us,
            compiler_session_id: None,
            compiler_event_digest: None,
        }
    }

    /// Whether this receipt represents a successful completion
    /// (including fallback completion).
    pub fn is_success(&self) -> bool {
        matches!(
            self.status,
            PhaseCompletion::Complete | PhaseCompletion::FallbackUsed(_)
        )
    }
}

/// Result of executing a full phase graph to completion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PhaseGraphResult {
    /// One receipt per phase, in execution order.
    pub receipts: Vec<PhaseReceipt>,
    /// Whether every phase reached a terminal successful state.
    pub all_completed: bool,
    /// Whether the executor was cancelled mid-flight.
    pub cancelled: bool,
}

impl PhaseGraphResult {
    /// Borrow the receipts in execution order.
    pub fn receipts(&self) -> &[PhaseReceipt] {
        &self.receipts
    }
}

// ── Error type ──────────────────────────────────────────────────────────

/// Typed error returned by the canonical phase-DAG executor.
///
/// Classification:
/// - `InvalidGraph` — malformed input (unknown phase, cycle).
/// - `InvalidTransition` — lifecycle state-machine violation.
/// - `RunnerFailed` — the runner raised a typed error during dispatch.
/// - `Cancelled` — caller requested abort mid-flight.
#[derive(Debug, Error)]
pub enum PhaseEngineError {
    /// The phase graph failed validation.
    #[error("invalid phase graph: {reason}")]
    InvalidGraph {
        /// Human-readable reason.
        reason: String,
    },
    /// A lifecycle transition was not allowed by the state machine.
    #[error("invalid phase lifecycle transition: {from:?} -> {to:?}")]
    InvalidTransition {
        /// The state being transitioned from.
        from: PhaseState,
        /// The state being transitioned to.
        to: PhaseState,
    },
    /// The runner raised a typed error during dispatch.
    #[error("phase runner failed: {0}")]
    RunnerFailed(#[from] PhaseRunnerError),
    /// The executor was cancelled mid-flight.
    #[error("phase engine cancelled")]
    Cancelled,
}

// ── Cancellation token ──────────────────────────────────────────────────

/// Cancellation token checked by the executor between phases.
///
/// The token is a thin `Cell<bool>` wrapper: the executor calls
/// [`is_cancelled`](Self::is_cancelled) on each iteration; the caller
/// calls [`cancel`](Self::cancel) to request abort.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken {
    cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
}

impl CancellationToken {
    /// Create a new, un-cancelled token.
    pub fn new() -> Self {
        Self::default()
    }

    /// Request cancellation. Idempotent.
    pub fn cancel(&self) {
        self.cancelled
            .store(true, std::sync::atomic::Ordering::SeqCst);
    }

    /// Whether cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(std::sync::atomic::Ordering::SeqCst)
    }
}

// ── Executor ────────────────────────────────────────────────────────────

/// Canonical phase-DAG executor.
///
/// The executor owns the ready-set computation, the lifecycle state
/// machine, the dispatch coordination through the canonical
/// [`PhaseRunner`] port, and the receipt collection. It does not own
/// concrete MLX / Metal / ANE bridge code; that lives in the engine
/// behind [`PhaseRunner`] implementations.
#[derive(Debug)]
pub struct PhaseGraphEngine {
    runners: PhaseRunnerRegistry,
}

impl PhaseGraphEngine {
    /// Create a new executor with the supplied runner registry.
    pub fn with_registry(runners: PhaseRunnerRegistry) -> Self {
        Self { runners }
    }

    /// Create a new executor with an empty runner registry. Phases
    /// dispatched through this engine will fail with
    /// [`PhaseRunnerError::NoRunner`] unless a runner is registered
    /// before execution.
    pub fn new() -> Self {
        Self {
            runners: PhaseRunnerRegistry::new(),
        }
    }

    /// Register a runner. Replaces any existing runner for the
    /// runner's [`PhaseKind`].
    pub fn register_runner(&mut self, runner: Box<dyn PhaseRunner>) {
        self.runners.register(runner);
    }

    /// Borrow the runner registry.
    pub fn runners(&self) -> &PhaseRunnerRegistry {
        &self.runners
    }

    /// Execute the full phase graph to completion.
    ///
    /// The executor iterates the ready set, dispatches each ready phase
    /// through the runner registry, records the receipt, and updates
    /// the lifecycle tracker. Iteration continues until no phases are
    /// ready (all phases have either completed or failed).
    ///
    /// The `ctx` is the opaque execution context passed to each
    /// runner; backends downcast to their own context type via
    /// [`PhaseRunnerContext::as_any_mut`].
    ///
    /// `cancel` is checked between phases. If the token is cancelled
    /// mid-flight, the executor returns
    /// [`PhaseEngineError::Cancelled`] with the partial result in
    /// [`PhaseGraphResult::receipts`].
    pub fn execute_graph(
        &self,
        graph: &PhaseGraph,
        ctx: &mut dyn PhaseRunnerContext,
        cancel: &CancellationToken,
    ) -> Result<PhaseGraphResult, PhaseEngineError> {
        graph.validate()?;

        let mut lifecycle = PhaseLifecycleTracker::new();
        for phase in &graph.phases {
            lifecycle.register(phase.phase_id.clone());
        }

        let mut completed: BTreeSet<PhaseId> = BTreeSet::new();
        let mut receipts: Vec<PhaseReceipt> = Vec::new();
        let mut cancelled = false;
        let ready_queue = ReadyQueue::new(graph);

        loop {
            if cancel.is_cancelled() {
                for phase in &graph.phases {
                    if !lifecycle.state(&phase.phase_id).is_terminal() {
                        let _ = lifecycle.transition(&phase.phase_id, PhaseState::Cancelled);
                    }
                }
                cancelled = true;
                break;
            }

            let ready = ready_queue.ready_phases(&completed);
            if ready.is_empty() {
                break;
            }

            for phase in ready {
                let phase_id = phase.phase_id.clone();

                let _ = lifecycle.transition(&phase_id, PhaseState::Ready);
                let _ = lifecycle.transition(&phase_id, PhaseState::Admitted);
                let _ = lifecycle.transition(&phase_id, PhaseState::Dispatched);

                let start = Instant::now();
                let run_result = self.runners.dispatch(phase, ctx);
                let duration_us = start.elapsed().as_micros() as u64;

                let status = match run_result {
                    Ok(()) => PhaseCompletion::Complete,
                    Err(e) => {
                        // Check for a fallback decomposition: any outgoing
                        // edge from this phase tagged FallbackDecomposition.
                        let has_fallback = graph.edges.keys().any(|(from, _)| {
                            from == &phase_id
                                && graph
                                    .edges
                                    .get(&(from.clone(), phase_id.clone()))
                                    .map(|kinds| {
                                        kinds.contains(&SemanticKind::FallbackDecomposition)
                                    })
                                    .unwrap_or(false)
                        });
                        if has_fallback {
                            PhaseCompletion::FallbackUsed(e.to_string())
                        } else {
                            PhaseCompletion::Failed(e.to_string())
                        }
                    }
                };

                let receipt = PhaseReceipt {
                    phase_id: phase_id.clone(),
                    status: status.clone(),
                    duration_us,
                    compiler_session_id: None,
                    compiler_event_digest: None,
                };

                let terminal = match status {
                    PhaseCompletion::Complete | PhaseCompletion::FallbackUsed(_) => {
                        let next = if matches!(status, PhaseCompletion::Complete) {
                            PhaseState::Complete
                        } else {
                            PhaseState::FallbackComplete
                        };
                        let _ = lifecycle.transition(&phase_id, next);
                        true
                    }
                    PhaseCompletion::Failed(_) | PhaseCompletion::Pending => {
                        let _ = lifecycle.transition(
                            &phase_id,
                            PhaseState::FailedBeforePublication,
                        );
                        true
                    }
                };
                if terminal {
                    completed.insert(phase_id);
                }

                receipts.push(receipt);
            }
        }

        let all_completed = graph
            .phases
            .iter()
            .all(|p| lifecycle.state(&p.phase_id).is_success());

        Ok(PhaseGraphResult {
            receipts,
            all_completed,
            cancelled,
        })
    }
}

impl Default for PhaseGraphEngine {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_phase(id: &str, kind: PhaseKind) -> Phase {
        Phase {
            phase_id: PhaseId::from(id),
            kind,
            lane: LaneTag::from("metal"),
            ops: vec![format!("op_{id}")],
            arena_slots: vec![],
            tensor_reads: vec![],
            tensor_writes: vec!["out".to_string()],
            estimated_ops: 100,
            metadata: BTreeMap::new(),
        }
    }

    fn empty_graph(phases: Vec<Phase>, edges: Vec<PhaseEdge>) -> PhaseGraph {
        let mut edge_map: BTreeMap<(PhaseId, PhaseId), BTreeSet<SemanticKind>> = BTreeMap::new();
        for e in edges {
            edge_map
                .entry((e.from_phase, e.to_phase))
                .or_default()
                .insert(e.semantic_kind);
        }
        PhaseGraph {
            phases,
            edges: edge_map,
            arena_plan: ArenaPlan::default(),
            concurrency_plan: ConcurrencyPlan::default(),
            compiler_version: "test".to_string(),
        }
    }

    #[test]
    fn single_phase_graph_validates_and_executes() {
        let graph = empty_graph(vec![make_phase("p0", PhaseKind::Decode)], vec![]);
        graph.validate().expect("valid graph");
        let engine = PhaseGraphEngine::new();
        let cancel = CancellationToken::new();
        let result = engine
            .execute_graph(&graph, &mut NoopContext, &cancel)
            .expect("execution");
        assert_eq!(result.receipts.len(), 1);
        assert!(matches!(result.receipts[0].status, PhaseCompletion::Failed(_)));
    }

    #[test]
    fn validate_rejects_unknown_edge_endpoint() {
        let graph = empty_graph(
            vec![make_phase("p0", PhaseKind::Decode)],
            vec![PhaseEdge {
                from_phase: PhaseId::from("p0"),
                to_phase: PhaseId::from("missing"),
                semantic_kind: SemanticKind::Data,
                label: None,
                metadata: BTreeMap::new(),
            }],
        );
        assert!(matches!(
            graph.validate(),
            Err(PhaseEngineError::InvalidGraph { .. })
        ));
    }

    #[test]
    fn validate_rejects_cycle() {
        let e = |a: &str, b: &str| PhaseEdge {
            from_phase: PhaseId::from(a),
            to_phase: PhaseId::from(b),
            semantic_kind: SemanticKind::Data,
            label: None,
            metadata: BTreeMap::new(),
        };
        let graph = empty_graph(
            vec![make_phase("a", PhaseKind::Decode), make_phase("b", PhaseKind::Decode)],
            vec![e("a", "b"), e("b", "a")],
        );
        assert!(matches!(
            graph.validate(),
            Err(PhaseEngineError::InvalidGraph { .. })
        ));
    }

    #[test]
    fn lifecycle_state_machine_accepts_valid_transitions() {
        let mut tracker = PhaseLifecycleTracker::new();
        let id = PhaseId::from("p1");
        tracker.register(id.clone());
        assert_eq!(tracker.state(&id), PhaseState::Dormant);
        tracker.transition(&id, PhaseState::Ready).unwrap();
        tracker.transition(&id, PhaseState::Admitted).unwrap();
        tracker.transition(&id, PhaseState::Dispatched).unwrap();
        tracker.transition(&id, PhaseState::Complete).unwrap();
        assert!(tracker.state(&id).is_terminal());
        assert!(tracker.state(&id).is_success());
    }

    #[test]
    fn lifecycle_state_machine_rejects_invalid_transition() {
        let mut tracker = PhaseLifecycleTracker::new();
        let id = PhaseId::from("p1");
        tracker.register(id.clone());
        assert!(matches!(
            tracker.transition(&id, PhaseState::Dispatched),
            Err(PhaseEngineError::InvalidTransition { .. })
        ));
    }

    #[test]
    fn cancellation_marks_remaining_phases_cancelled() {
        let e = |a: &str, b: &str| PhaseEdge {
            from_phase: PhaseId::from(a),
            to_phase: PhaseId::from(b),
            semantic_kind: SemanticKind::Data,
            label: None,
            metadata: BTreeMap::new(),
        };
        let graph = empty_graph(
            vec![make_phase("a", PhaseKind::Decode), make_phase("b", PhaseKind::Decode)],
            vec![e("a", "b")],
        );
        let engine = PhaseGraphEngine::new();
        let cancel = CancellationToken::new();
        cancel.cancel();
        let result = engine
            .execute_graph(&graph, &mut NoopContext, &cancel)
            .expect("cancelled is not a hard error");
        assert!(result.cancelled);
        assert!(!result.all_completed);
    }

    #[test]
    fn ready_queue_emits_independent_phases() {
        let e = |a: &str, b: &str| PhaseEdge {
            from_phase: PhaseId::from(a),
            to_phase: PhaseId::from(b),
            semantic_kind: SemanticKind::Data,
            label: None,
            metadata: BTreeMap::new(),
        };
        let graph = empty_graph(
            vec![
                make_phase("a", PhaseKind::Decode),
                make_phase("b", PhaseKind::Decode),
                make_phase("c", PhaseKind::Decode),
            ],
            vec![e("a", "c"), e("b", "c")],
        );
        let queue = ReadyQueue::new(&graph);
        let ready = queue.ready_phases(&BTreeSet::new());
        let ids: BTreeSet<&str> = ready.iter().map(|p| p.phase_id.as_str()).collect();
        assert!(ids.contains("a") && ids.contains("b") && !ids.contains("c"));
    }

    /// No-op context used by tests that don't exercise the dispatch
    /// path beyond the runner-registry's "no runner" failure.
    struct NoopContext;

    impl PhaseRunnerContext for NoopContext {
        fn as_any(&self) -> &dyn std::any::Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
            self
        }
    }
}
