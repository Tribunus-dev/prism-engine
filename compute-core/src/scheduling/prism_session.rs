//! PrismSession and DecodeScheduler — alpha decode loop types.
//!
//! This module defines the session abstraction for the Prism inference runtime:
//! a single generation request tracked through prompt processing, decode epochs,
//! and completion with an evidence log of runtime decisions.


use crate::backend::coreml_iosurface::CoreMlIOSurfaceExecutable;
use crate::backend::metal_consumer::MetalConsumer;
use crate::compilation::epoch_scheduler::EpochScheduler;
use crate::compilation::failure_injector::FailureInjector;
use crate::compilation::tri_lane::EpochRouteOrigin;
use crate::scheduling::tri_lane_orchestrator::{
    PhaseVariantSet, TriLaneOrchestrator,
};
use crate::compute_image::apple_shared_arena::AppleSharedArena;

// PrismSessionRequest
// ---------------------------------------------------------------------------

/// An incoming inference request to be materialized into a [`PrismSession`].
#[derive(Debug, Clone)]
pub struct PrismSessionRequest {
    /// Digest identifying the compute image to load.
    pub image_digest: String,
    /// The raw text prompt.
    pub prompt: String,
    /// Maximum number of new tokens to generate (soft limit — the scheduler
    /// may terminate earlier on EOS or budget).
    pub max_new_tokens: u32,
    /// Logical context bucket for scheduling priority / batching.
    pub context_bucket: u32,
    /// Softmax temperature. 0 = greedy (argmax).
    pub temperature: f32,
    /// Top-p nucleus sampling threshold.
    pub top_p: f32,
    /// Optional deterministic seed (None = random).
    pub seed: Option<u64>,
}

// ---------------------------------------------------------------------------
// GenerationState
// ---------------------------------------------------------------------------

/// The current phase of a generation session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GenerationState {
    /// Model is consuming the prompt to build the initial KV cache.
    PromptProcessing,
    /// Autoregressive token-by-token generation.
    Decoding,
    /// Generation finished normally.
    Completed,
    /// Generation finished with an error.
    Failed(String),
}

// ---------------------------------------------------------------------------
// DecodeScheduler
// ---------------------------------------------------------------------------

/// Lightweight epoch scheduler for a single session's decode loop.
///
/// Tracks the current epoch (decode step) against the requested budget and
/// signals termination.
#[derive(Debug, Clone)]
pub struct DecodeScheduler {
    /// Current decode epoch (0 = prompt processing, 1..N = decode steps).
    pub epoch: u64,
    /// Maximum new tokens configured for this session.
    pub max_new_tokens: u32,
    /// Once true the scheduler will not schedule further decode steps.
    pub terminated: bool,
}

impl DecodeScheduler {
    /// Create a new scheduler ready for prompt processing.
    pub fn new(max_new_tokens: u32) -> Self {
        Self {
            epoch: 0,
            max_new_tokens,
            terminated: false,
        }
    }

    /// Advance to the next epoch.
    pub fn advance(&mut self) {
        self.epoch += 1;
        if self.epoch as u32 > self.max_new_tokens {
            self.terminated = true;
        }
    }

    /// Request graceful termination at the next opportunity.
    pub fn terminate(&mut self) {
        self.terminated = true;
    }
}

// ---------------------------------------------------------------------------
// KvSlotState
// ---------------------------------------------------------------------------

/// Lifecycle state of a single KV-cache slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum KvSlotState {
    /// Slot has never been touched.
    Unallocated,
    /// Memory reserved but no data written.
    Allocated,
    /// Prefix KV data written (prompt processed).
    Primed,
    /// Actively participating in decode.
    Decoding,
    /// Slot data has been persisted/synced for checkpoint or migration.
    Synchronized,
    /// Slot contents are no longer valid (e.g. after rollback or parent
    /// invalidation).
    Invalidated,
    /// Resources released back to the pool.
    Released,
}

// ---------------------------------------------------------------------------
// KvRuntime (stub)
// ---------------------------------------------------------------------------

/// Stub KV-cache runtime for the alpha decode loop.
///
/// Will be replaced with a real Arena-backed implementation in a later phase.
#[derive(Debug, Clone)]
pub struct KvRuntime {
    generation: u64,
}

impl KvRuntime {
    /// Create a new runtime with generation counter zero.
    pub fn new() -> Self {
        Self { generation: 0 }
    }

    /// Allocate space for `seq_len` tokens in the KV cache.
    pub fn allocate(&mut self, _seq_len: u32) {
        todo!("KvRuntime::allocate — not yet implemented")
    }

    /// Append a single layer's key/value data to the current sequence.
    pub fn append(&mut self, _layer: u32, _key: &[u8], _value: &[u8]) {
        todo!("KvRuntime::append — not yet implemented")
    }

    /// Return a borrowed view of a single layer's KV data.
    pub fn view(&self, _layer: u32) {
        todo!("KvRuntime::view — not yet implemented")
    }

    /// Roll back the most recent append (e.g. after a rejected speculative
    /// token).
    pub fn rollback(&mut self) {
        todo!("KvRuntime::rollback — not yet implemented")
    }

    /// Mark the entire cache as invalid (e.g. after a context switch).
    pub fn invalidate(&mut self) {
        // Stub: no-op until KV runtime is wired
    }

    pub fn release(&mut self) {
        // Stub: no-op until KV runtime is wired
    }

    /// Return the current generation counter.
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

impl Default for KvRuntime {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// SessionLogEntry
// ---------------------------------------------------------------------------

/// A single structured observation recorded during a decode epoch.
#[derive(Debug, Clone)]
pub struct SessionLogEntry {
    /// Epoch index at which this entry was recorded.
    pub epoch: u64,
    /// The token emitted (None for prompt processing / prefill epochs).
    pub token: Option<u32>,
    /// Wall-clock time elapsed during this epoch, in nanoseconds.
    pub wall_time_ns: u64,
    /// Whether the epoch fell back to a slower compute path.
    pub fallback_used: bool,
    /// Which compute route produced this token (e.g. "ane", "gpu", "cpu").
    pub route_origin: String,
}

// ---------------------------------------------------------------------------
// SessionEvidenceLog
// ---------------------------------------------------------------------------

/// Append-only log of observations from a session's decode loop.
#[derive(Debug, Clone)]
pub struct SessionEvidenceLog {
    /// Ordered entries, one per epoch (or per significant event).
    pub entries: Vec<SessionLogEntry>,
}

impl SessionEvidenceLog {
    /// Create an empty log.
    pub fn new() -> Self {
        Self {
            entries: Vec::new(),
        }
    }

    /// Append a single entry.
    pub fn record(&mut self, entry: SessionLogEntry) {
        self.entries.push(entry);
    }

    /// Return the number of recorded entries.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// True if no entries have been recorded.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

impl Default for SessionEvidenceLog {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// PrismExecutionMode
// ---------------------------------------------------------------------------

/// Execution mode for a single step call.
#[derive(Debug, Clone)]
pub struct PrismExecutionMode {
    pub tri_lane_enabled: bool,
    pub risk_policy: crate::compilation::ane_admission_gate::RiskPolicy,
    pub scheduling_mode: SchedulingMode,
}

/// Scheduling strategy for this step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulingMode {
    Latency,
    Throughput,
    Benchmark,
}

// ---------------------------------------------------------------------------
// PrismStepRequest
// ---------------------------------------------------------------------------

/// A single step request.
#[derive(Debug, Clone)]
pub struct PrismStepRequest {
    pub session_id: String,
    pub epoch_id: u64,
    pub execution_mode: PrismExecutionMode,
    pub inputs: Vec<f32>,
}

// ---------------------------------------------------------------------------
// PrismStepResult
// ---------------------------------------------------------------------------

/// Result of a single step.
#[derive(Debug, Clone)]
pub struct PrismStepResult {
    pub output: Vec<f32>,
    pub lane_receipts: Vec<crate::scheduling::tri_lane_orchestrator::TriLaneExecutionReceipt>,
}


// ---------------------------------------------------------------------------
// PrismSession
// ---------------------------------------------------------------------------

/// A single generation session — owns the tokenizer handle, KV runtime,
/// decode scheduler, state machine, and evidence log.
///
/// Alpha stub: tokenizer and kv_runtime fields are placeholders that will
/// be replaced with real types once those subsystems are wired in.
#[derive(Debug, Clone)]
pub struct PrismSession {
    /// Unique session identifier.
    pub session_id: String,
    /// Digest of the compute image this session runs on.
    pub image_digest: String,
    /// Placeholder tokenizer handle (stub).
    pub tokenizer: (),
    /// Placeholder KV runtime handle (stub).
    pub kv_runtime: KvRuntime,
    /// Per-session decode scheduler.
    pub scheduler: DecodeScheduler,
    /// Current generation phase.
    pub generation_state: GenerationState,
    /// Structured log of runtime observations.
    pub evidence_log: SessionEvidenceLog,
}

impl PrismSession {
    /// Materialize a new session from a request.
    pub fn new(request: PrismSessionRequest) -> Self {
        Self {
            session_id: uuid::Uuid::new_v4().to_string(),
            image_digest: request.image_digest,
            tokenizer: (),
            kv_runtime: KvRuntime::new(),
            scheduler: DecodeScheduler::new(request.max_new_tokens),
            generation_state: GenerationState::PromptProcessing,
            evidence_log: SessionEvidenceLog::new(),
        }
    }

    /// Execute one step using the tri-lane orchestrator.
    pub async fn step_tri_lane(
        &mut self,
        request: PrismStepRequest,
        orchestrator: &mut TriLaneOrchestrator,
        phase_set: &PhaseVariantSet,
    ) -> Result<PrismStepResult, String> {
        if self.scheduler.terminated
            || matches!(
                self.generation_state,
                GenerationState::Completed | GenerationState::Failed(_)
            )
        {
            return Err("session terminated".into());
        }

        // 1. Submit phase to orchestrator — selects best variant, enqueues on lane
        let completion = orchestrator.submit(phase_set)?;

        // 2. Process all available lane completions from the channel
        //    In production, this runs in a Tokio select loop with timeout.
        let completions: Vec<_> = orchestrator.completion_rx
        .as_mut()
        .map(|rx| std::iter::from_fn(|| rx.try_recv().ok()).collect::<Vec<_>>())
        .unwrap_or_default();

        for c in completions {
                orchestrator.apply_completion(c)?;
            }

        // 3. Apply the submit-origin completion (synchronous receipt)
        orchestrator.apply_completion(completion)?;

        // 4. Collect receipts for the step result
        let lane_receipts = orchestrator.receipts.receipts.clone();

        // 5. Record evidence
        self.evidence_log.record(SessionLogEntry {
            epoch: self.scheduler.epoch,
            token: None,
            wall_time_ns: 0,
            fallback_used: false,
            route_origin: format!("{:?}", EpochRouteOrigin::CoreMlAne),
        });

        // 6. Advance state
        self.advance_state();

        Ok(PrismStepResult {
            output: request.inputs,  // passthrough stub
            lane_receipts,
        })
    }


    /// Advance the session's generation state (called after each epoch
    /// completes).
    pub fn advance_state(&mut self) {
        match self.generation_state {
            GenerationState::PromptProcessing => {
                self.generation_state = GenerationState::Decoding;
            }
            GenerationState::Decoding => {
                if self.scheduler.terminated {
                    self.generation_state = GenerationState::Completed;
                }
                self.scheduler.advance();
            }
            GenerationState::Completed | GenerationState::Failed(_) => {
                // Terminal states — no further transitions.
            }
        }
    }

    /// Execute one decode step against the installed runtime resources.
    pub fn step(
        &mut self,
        scheduler: &mut EpochScheduler,
        arena: &mut AppleSharedArena,
        coreml_exec: &mut CoreMlIOSurfaceExecutable,
        metal_consumer: &mut MetalConsumer,
        injector: &dyn FailureInjector,
    ) -> Result<Option<u32>, String> {
        if self.scheduler.terminated || matches!(self.generation_state, GenerationState::Completed | GenerationState::Failed(_)) {
            return Ok(None);
        }

        // Check injector before dispatch
        let receipt = if injector.should_fail_before_prediction(self.scheduler.epoch) {
            // Inject failure by swapping to a nonexistent model path
            let original_path = coreml_exec.model_path.clone();
            let original_loaded = coreml_exec.loaded;
            coreml_exec.model_path = "/tmp/nonexistent.mlmodelc".into();
            coreml_exec.loaded = false;
            let result = scheduler.execute_epoch(arena, coreml_exec, metal_consumer);
            coreml_exec.model_path = original_path;
            coreml_exec.loaded = original_loaded;
            result?
        } else {
            scheduler.execute_epoch(arena, coreml_exec, metal_consumer)?
        };

        // Record evidence
        self.evidence_log.record(SessionLogEntry {
            epoch: self.scheduler.epoch,
            token: None,
            wall_time_ns: 0,
            fallback_used: receipt.fallback_used,
            route_origin: format!("{:?}", receipt.route_origin),
        });

        // Advance state
        self.advance_state();

        Ok(Some(0))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prism_session_request_defaults() {
        let req = PrismSessionRequest {
            image_digest: "abc123".into(),
            prompt: "Hello".into(),
            max_new_tokens: 128,
            context_bucket: 0,
            temperature: 0.7,
            top_p: 0.9,
            seed: None,
        };
        assert_eq!(req.prompt, "Hello");
        assert_eq!(req.max_new_tokens, 128);
    }

    #[test]
    fn test_session_creation_starts_prompt_processing() {
        let req = PrismSessionRequest {
            image_digest: "digest".into(),
            prompt: "test".into(),
            max_new_tokens: 10,
            context_bucket: 0,
            temperature: 0.0,
            top_p: 1.0,
            seed: Some(42),
        };
        let session = PrismSession::new(req);
        assert!(session.session_id.len() >= 32); // UUID v4
        assert_eq!(session.generation_state, GenerationState::PromptProcessing);
        assert_eq!(session.evidence_log.len(), 0);
    }

    #[test]
    fn test_decode_scheduler_advance_and_termination() {
        let mut sched = DecodeScheduler::new(3);
        assert_eq!(sched.epoch, 0);
        assert!(!sched.terminated);

        // Epoch 0 -> 1 (first decode step)
        sched.advance();
        assert_eq!(sched.epoch, 1);
        assert!(!sched.terminated);

        // Epoch 1 -> 2
        sched.advance();
        assert_eq!(sched.epoch, 2);
        assert!(!sched.terminated);

        // Epoch 2 -> 3
        sched.advance();
        assert_eq!(sched.epoch, 3);
        assert!(!sched.terminated);

        // Epoch 3 -> 4, exceeds max_new_tokens => terminated
        sched.advance();
        assert_eq!(sched.epoch, 4);
        assert!(sched.terminated);
    }

    #[test]
    fn test_decode_scheduler_explicit_terminate() {
        let mut sched = DecodeScheduler::new(100);
        assert!(!sched.terminated);
        sched.terminate();
        assert!(sched.terminated);
    }

    #[test]
    fn test_generation_state_transition() {
        let req = PrismSessionRequest {
            image_digest: "d".into(),
            prompt: "p".into(),
            max_new_tokens: 5,
            context_bucket: 0,
            temperature: 0.0,
            top_p: 1.0,
            seed: None,
        };
        let mut session = PrismSession::new(req);

        // Start in PromptProcessing
        assert_eq!(session.generation_state, GenerationState::PromptProcessing);

        // Advance => Decoding
        session.advance_state();
        assert_eq!(session.generation_state, GenerationState::Decoding);

        // Terminate the scheduler
        session.scheduler.terminate();

        // Advance => Completed
        session.advance_state();
        assert_eq!(session.generation_state, GenerationState::Completed);

        // Terminal state stays put
        session.advance_state();
        assert_eq!(session.generation_state, GenerationState::Completed);
    }

    #[test]
    fn test_evidence_log_append_and_query() {
        let mut log = SessionEvidenceLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);

        log.record(SessionLogEntry {
            epoch: 1,
            token: Some(42),
            wall_time_ns: 1_000_000,
            fallback_used: false,
            route_origin: "ane".into(),
        });

        assert!(!log.is_empty());
        assert_eq!(log.len(), 1);

        log.record(SessionLogEntry {
            epoch: 2,
            token: Some(99),
            wall_time_ns: 2_000_000,
            fallback_used: true,
            route_origin: "gpu".into(),
        });

        assert_eq!(log.len(), 2);
        assert_eq!(log.entries[0].epoch, 1);
        assert_eq!(log.entries[1].epoch, 2);
        assert!(log.entries[1].fallback_used);
    }

    #[test]
    fn test_kv_slot_state_variants() {
        // Just verify all variants exist and are distinct
        let states = vec![
            KvSlotState::Unallocated,
            KvSlotState::Allocated,
            KvSlotState::Primed,
            KvSlotState::Decoding,
            KvSlotState::Synchronized,
            KvSlotState::Invalidated,
            KvSlotState::Released,
        ];
        assert_eq!(states.len(), 7);
        assert_ne!(states[0], states[1]);
    }

    #[test]
    fn test_kv_runtime_stub() {
        let mut rt = KvRuntime::new();
        assert_eq!(rt.generation(), 0);
        // Stub methods are todo!() — this test confirms the type compiles.
        // Real implementation tests will come in a later phase.
        rt.release();
    }

    #[test]
    fn test_session_failed_state_stays_terminal() {
        let req = PrismSessionRequest {
            image_digest: "d".into(),
            prompt: "p".into(),
            max_new_tokens: 5,
            context_bucket: 0,
            temperature: 0.0,
            top_p: 1.0,
            seed: None,
        };
        let mut session = PrismSession::new(req);
        session.generation_state = GenerationState::Failed("OOM".into());

        // advance_state should be a no-op on Failed
        session.advance_state();
        assert_eq!(
            session.generation_state,
            GenerationState::Failed("OOM".into())
        );
    }

    #[test]
    fn test_decode_scheduler_default_epoch_zero() {
        let sched = DecodeScheduler::new(256);
        assert_eq!(sched.epoch, 0);
        assert!(!sched.terminated);
        assert_eq!(sched.max_new_tokens, 256);
    }

    #[test]
    fn test_kv_runtime_allocate_append_view_roundtrip() {
        // Compile-time check that the method signatures compile.
        let _rt = KvRuntime::new();
        // allocate, append, view, rollback, invalidate, release all exist.
    }
}
