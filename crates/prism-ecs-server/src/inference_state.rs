//! Inference state types (constitutional home).
//!
//! This module owns the canonical authority for the per-image, per-session,
//! and per-step inference state containers that drive the phase-engine
//! execution path. The legacy home was
//! `compute-core/src/ecs/inference/{execution_image_state,inference_session_state,inference_step_state,phase_engine_adapter}.rs`,
//! absorbed during the engine-subsystem deletion pass (see
//! `changelogs/2026-07-27-engine-subsystem-deletion-inference.md`).
//!
//! The types are organised along three axis-stable domains:
//!
//! - [`ComputeImageState`] — immutable per-image state, shareable across
//!   sessions, never mutated after construction.
//! - [`InferenceSessionState`] — mutable per-session state (KV caches,
//!   working set, cancellation flag, receipt ledger).
//! - [`InferenceStepState`] — mutable per-step state (activations, sampling,
//!   receipts, deadline).
//!
//! [`PhaseEngineAdapter`] is the bridge that turns these into
//! `prism_ecs_runtime::scheduling::systems::phase_engine::PhaseEngine`
//! invocations.
//!
//! # Authority boundary
//!
//! These are pure server-side data carriers. They are *not* the canonical
//! world; they are the typed shapes the constitutional commands mutate.
//! Receipt emission, scheduled dispatch, and durable event application
//! happen in the runtime and constitutional crates, not here.

use prism_ecs_runtime::scheduling::evidence::scheduling_receipts::PhaseReceipt;
use prism_ecs_runtime::scheduling::state::activation_binding::CurrentActivation;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;

// ---------------------------------------------------------------------------
// Per-image identifiers and placeholders
// ---------------------------------------------------------------------------

/// Stable, content-addressed identifier for a ComputeImage.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ComputeImageId(pub String);

/// Identifier for a target hardware profile (e.g. "apple-m1-pro-16",
/// "apple-m4-max-64").
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct TargetProfileId(pub String);

/// Compiled phase program version identifier.
///
/// The engine's `crate::ecs::compute_image::phase_program_version`
/// module supplies the canonical implementation; the constitutional
/// home ships a typed version carrier until the
/// `phase_program_version` migration lands.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct PhaseProgramVersion(pub u32);

/// Pre-computed RoPE trigonometric tables for the model.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RopeTables {
    /// Cosine table for rotary positions.
    pub cos: Vec<f32>,
    /// Sine table for rotary positions.
    pub sin: Vec<f32>,
    /// Full cosine table covering max sequence length.
    pub full_cos: Vec<f32>,
    /// Full sine table covering max sequence length.
    pub full_sin: Vec<f32>,
}

/// Placeholder for the engine's fusion binding registry. The canonical
/// fusion binding type moves with the `fusion_abi` migration; this
/// minimal carrier preserves the public surface until then.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FusionBindingRegistry {
    /// All sealed fusion artifacts available to this image.
    pub artifacts: Vec<FusionBindingArtifact>,
}

/// One sealed fusion binding artifact, content-addressed.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionBindingArtifact {
    /// Stable identifier of the binding.
    pub binding_id: String,
    /// Digest of the binding's binary payload.
    pub payload_digest: String,
}

// ---------------------------------------------------------------------------
// ComputeImageState
// ---------------------------------------------------------------------------

/// Immutable state derived from a sealed ComputeImage.
///
/// This is the canonical source of truth for model architecture, weights,
/// fusion artifacts, and compilation decisions for the duration of an
/// image's lifetime. It is shareable across concurrent sessions and is
/// never mutated after construction.
///
/// # Send + Sync
///
/// All heap-allocated fields use `Arc`, making the entire struct
/// `Send + Sync`.
#[derive(Debug, Clone)]
pub struct ComputeImageState {
    /// Content-addressed identifier for this image.
    pub image_id: ComputeImageId,
    /// The target hardware profile this image was compiled for.
    pub target_profile: TargetProfileId,
    /// Resolved fusion binding registry (placeholder until the
    /// `fusion_abi` migration lands).
    pub fusion_bindings: Arc<FusionBindingRegistry>,
    /// Pre-computed RoPE trigonometric tables.
    pub rope_tables: Arc<RopeTables>,
}

impl ComputeImageState {
    /// Construct an empty `ComputeImageState`. Real construction logic
    /// is filled in by the `phase_program_version` / `fusion_abi`
    /// migration.
    pub fn empty() -> Self {
        Self {
            image_id: ComputeImageId(String::new()),
            target_profile: TargetProfileId(String::new()),
            fusion_bindings: Arc::new(FusionBindingRegistry::default()),
            rope_tables: Arc::new(RopeTables::default()),
        }
    }
}

// ---------------------------------------------------------------------------
// Per-session identifiers
// ---------------------------------------------------------------------------

/// Unique identifier for an inference session.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct InferenceSessionId(pub String);

// ---------------------------------------------------------------------------
// InferenceSessionState
// ---------------------------------------------------------------------------

/// Mutable per-session state owned by the runtime.
///
/// Contains the session's KV caches, working set, cancellation flag,
/// and the receipt ledger accumulated during the session.
#[derive(Debug)]
pub struct InferenceSessionState {
    /// Stable session identifier.
    pub session_id: InferenceSessionId,
    /// Working set of weights currently resident for this session.
    pub working_set: Option<WorkingSetManager>,
    /// Cooperative cancellation flag.
    pub cancellation: Arc<AtomicBool>,
    /// Monotonic session epoch, advanced on each epoch-bumping event.
    pub session_epoch: AtomicU64,
    /// Per-step phase receipts accumulated during the session.
    pub receipt_ledger: StepReceiptLedger,
}

/// Working set manager — placeholder for the engine's
/// `WorkingSetManager`. Holds the weight tensors currently resident in
/// device memory for the active session.
#[derive(Debug, Default, Clone)]
pub struct WorkingSetManager {
    /// Weight tensor names currently resident.
    pub resident: BTreeMap<String, WeightResidencyToken>,
}

/// A residency token indicating a weight tensor is pinned for the
/// session. The token's `Drop` releases the residency lease.
#[derive(Debug, Clone)]
pub struct WeightResidencyToken {
    /// Name of the pinned tensor.
    pub tensor_name: String,
}

impl InferenceSessionState {
    /// Build a fresh per-session state.
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: InferenceSessionId(session_id.into()),
            working_set: None,
            cancellation: Arc::new(AtomicBool::new(false)),
            session_epoch: AtomicU64::new(0),
            receipt_ledger: StepReceiptLedger::new(),
        }
    }

    /// `true` if cancellation has been requested.
    pub fn is_cancelled(&self) -> bool {
        self.cancellation.load(Ordering::Relaxed)
    }

    /// Request cancellation.
    pub fn cancel(&self) {
        self.cancellation.store(true, Ordering::Relaxed);
    }

    /// Increment and return the next session epoch.
    pub fn next_epoch(&self) -> u64 {
        self.session_epoch.fetch_add(1, Ordering::Relaxed)
    }

    /// Record a phase receipt in the session receipt ledger.
    pub fn push_receipt(&mut self, receipt: PhaseReceipt) {
        self.receipt_ledger.push(receipt);
    }
}

// ---------------------------------------------------------------------------
// Per-step state
// ---------------------------------------------------------------------------

/// Unique request identifier.
#[derive(Debug, Clone, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct RequestId(pub u64);

/// Unique execution identifier for this step.
#[derive(Debug, Clone, Copy, Hash, Eq, PartialEq, Serialize, Deserialize)]
pub struct ExecutionId(pub u64);

/// Inference mode for the current step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InferenceMode {
    /// Prefill — initial prompt ingestion.
    Prefill,
    /// Decode — autoregressive token-by-token generation.
    Decode,
}

/// Token input for a step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenInput {
    /// Input token IDs.
    pub token_ids: Vec<u32>,
}

impl TokenInput {
    /// Build a `TokenInput` from a single token.
    pub fn from_token(token: u32) -> Self {
        Self {
            token_ids: vec![token],
        }
    }
}

/// Status table tracking phase completion for the current step.
#[derive(Debug, Clone, Default)]
pub struct PhaseStatusTable {
    statuses: BTreeMap<String, PhaseStatus>,
}

/// One phase's status within a step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PhaseStatus {
    /// Not yet started.
    Pending,
    /// Ready to be picked up by a runner.
    Ready,
    /// Currently executing.
    Running,
    /// Completed successfully.
    Complete,
    /// Phase failed.
    Failed,
}

impl PhaseStatusTable {
    /// Construct an empty table.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set a phase's status.
    pub fn set(&mut self, phase_id: impl Into<String>, status: PhaseStatus) {
        self.statuses.insert(phase_id.into(), status);
    }

    /// Read a phase's status. Missing phases read as `Pending`.
    pub fn get(&self, phase_id: &str) -> PhaseStatus {
        self.statuses.get(phase_id).copied().unwrap_or(PhaseStatus::Pending)
    }
}

/// Ledger of phase receipts for the current step.
#[derive(Debug, Clone, Default)]
pub struct StepReceiptLedger {
    /// Ordered receipts.
    pub receipts: Vec<PhaseReceipt>,
}

impl StepReceiptLedger {
    /// Construct an empty ledger.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a receipt.
    pub fn push(&mut self, receipt: PhaseReceipt) {
        self.receipts.push(receipt);
    }

    /// Drain the ledger, returning all accumulated receipts.
    pub fn take(&mut self) -> Vec<PhaseReceipt> {
        std::mem::take(&mut self.receipts)
    }
}

/// Output of an inference step.
#[derive(Debug, Clone, Default)]
pub struct InferenceStepOutput {
    /// Terminal token, if the step produced one.
    pub token: Option<u32>,
    /// Logits, if the step produced logits.
    pub logits: Option<Vec<f32>>,
    /// Phase receipts collected during the step.
    pub receipts: Vec<PhaseReceipt>,
}

/// Mutable per-step state.
///
/// Created fresh for each prefill chunk or decode step.
#[derive(Debug)]
pub struct InferenceStepState {
    /// The request this step serves.
    pub request_id: RequestId,
    /// The unique execution identifier for this step.
    pub execution_id: ExecutionId,
    /// Whether this step is a prefill or a decode.
    pub mode: InferenceMode,
    /// Position of the current token in the sequence.
    pub token_position: usize,
    /// Input tokens for this step.
    pub input_tokens: TokenInput,
    /// Current input activation (placeholder until the tensor-ABI
    /// migration lands).
    pub current_activation: Option<CurrentActivation>,
    /// Output logits from the most recent run.
    pub logits: Option<Vec<f32>>,
    /// Output activation (placeholder).
    pub output_activation: Option<CurrentActivation>,
    /// Per-phase status table.
    pub phase_status: PhaseStatusTable,
    /// Step receipt ledger.
    pub receipt_ledger: StepReceiptLedger,
    /// Optional deadline for this step.
    pub deadline: Option<std::time::Instant>,
    /// Terminal output, if the step reached its terminal phase.
    pub terminal_output: Option<InferenceStepOutput>,
}

impl InferenceStepState {
    /// Construct a `Prefill` step.
    pub fn new_prefill(request_id: u64, execution_id: u64, tokens: Vec<u32>) -> Self {
        Self {
            request_id: RequestId(request_id),
            execution_id: ExecutionId(execution_id),
            mode: InferenceMode::Prefill,
            token_position: 0,
            input_tokens: TokenInput { token_ids: tokens },
            current_activation: None,
            logits: None,
            output_activation: None,
            phase_status: PhaseStatusTable::new(),
            receipt_ledger: StepReceiptLedger::new(),
            deadline: None,
            terminal_output: None,
        }
    }

    /// Construct a `Decode` step.
    pub fn new_decode(request_id: u64, execution_id: u64, token: u32, position: usize) -> Self {
        Self {
            request_id: RequestId(request_id),
            execution_id: ExecutionId(execution_id),
            mode: InferenceMode::Decode,
            token_position: position,
            input_tokens: TokenInput::from_token(token),
            current_activation: None,
            logits: None,
            output_activation: None,
            phase_status: PhaseStatusTable::new(),
            receipt_ledger: StepReceiptLedger::new(),
            deadline: None,
            terminal_output: None,
        }
    }
}

// ---------------------------------------------------------------------------
// PhaseEngineAdapter
// ---------------------------------------------------------------------------

/// Bridge that turns server-side state into
/// `prism_ecs_runtime::scheduling::systems::phase_engine::PhaseEngine`
/// invocations.
///
/// The adapter holds no per-call state; it is a pure dispatcher from
/// the typed server state into the engine's phase runner.
pub struct PhaseEngineAdapter;

impl Default for PhaseEngineAdapter {
    fn default() -> Self {
        Self::new()
    }
}

impl PhaseEngineAdapter {
    /// Build a new adapter.
    pub fn new() -> Self {
        Self
    }

    /// Execute a prefill step through the phase engine.
    ///
    /// Sets the step mode to `Prefill` before dispatching. Returns the
    /// engine's `Result` and the `InferenceStepOutput` it produced.
    pub fn execute_prefill(
        &self,
        image: &ComputeImageState,
        session: &mut InferenceSessionState,
        step: &mut InferenceStepState,
    ) -> Result<InferenceStepOutput, String> {
        step.mode = InferenceMode::Prefill;
        // The phase engine itself runs the dispatch loop; the adapter
        // here only stamps the step and runs the placeholder dispatch
        // until the engine's runtime caller is ported.
        let _ = (image, session);
        Ok(InferenceStepOutput {
            token: step.terminal_output.as_ref().and_then(|o| o.token),
            logits: step.logits.clone(),
            receipts: step.receipt_ledger.take(),
        })
    }

    /// Execute a decode step through the phase engine.
    pub fn execute_decode(
        &self,
        image: &ComputeImageState,
        session: &mut InferenceSessionState,
        step: &mut InferenceStepState,
    ) -> Result<InferenceStepOutput, String> {
        step.mode = InferenceMode::Decode;
        let _ = (image, session);
        Ok(InferenceStepOutput {
            token: step.terminal_output.as_ref().and_then(|o| o.token),
            logits: step.logits.clone(),
            receipts: step.receipt_ledger.take(),
        })
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn compute_image_state_empty_builds() {
        let _ = ComputeImageState::empty();
    }

    #[test]
    fn session_state_lifecycle() {
        let mut s = InferenceSessionState::new("session-1");
        assert_eq!(s.session_id.0, "session-1");
        assert!(!s.is_cancelled());
        s.cancel();
        assert!(s.is_cancelled());
        assert_eq!(s.next_epoch(), 0);
        assert_eq!(s.next_epoch(), 1);
    }

    #[test]
    fn step_state_prefill_and_decode_constructors() {
        let prefill = InferenceStepState::new_prefill(1, 10, vec![100, 200, 300]);
        assert_eq!(prefill.request_id.0, 1);
        assert_eq!(prefill.execution_id.0, 10);
        assert_eq!(prefill.mode, InferenceMode::Prefill);
        assert_eq!(prefill.input_tokens.token_ids, vec![100, 200, 300]);

        let decode = InferenceStepState::new_decode(2, 11, 7, 5);
        assert_eq!(decode.mode, InferenceMode::Decode);
        assert_eq!(decode.token_position, 5);
        assert_eq!(decode.input_tokens.token_ids, vec![7]);
    }

    #[test]
    fn phase_status_table_defaults_to_pending() {
        let mut table = PhaseStatusTable::new();
        assert_eq!(table.get("missing"), PhaseStatus::Pending);
        table.set("phase-1", PhaseStatus::Running);
        assert_eq!(table.get("phase-1"), PhaseStatus::Running);
    }

    #[test]
    fn step_receipt_ledger_push_and_take() {
        let mut ledger = StepReceiptLedger::new();
        let r = PhaseReceipt::completed("phase-1", 1_000);
        ledger.push(r);
        assert_eq!(ledger.receipts.len(), 1);
        let drained = ledger.take();
        assert_eq!(drained.len(), 1);
        assert!(ledger.receipts.is_empty());
    }

    #[test]
    fn phase_engine_adapter_stamps_mode() {
        let adapter = PhaseEngineAdapter::new();
        let image = ComputeImageState::empty();
        let mut session = InferenceSessionState::new("s");
        let mut step = InferenceStepState::new_prefill(1, 1, vec![1, 2]);
        let _ = adapter
            .execute_prefill(&image, &mut session, &mut step)
            .expect("prefill returns");
        assert_eq!(step.mode, InferenceMode::Prefill);

        let mut step = InferenceStepState::new_decode(1, 1, 7, 1);
        let _ = adapter
            .execute_decode(&image, &mut session, &mut step)
            .expect("decode returns");
        assert_eq!(step.mode, InferenceMode::Decode);
    }
}
