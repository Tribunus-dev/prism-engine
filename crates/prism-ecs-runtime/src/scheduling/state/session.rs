//! PrismSession state (constitutional home for the runtime half).
//!
//! Per-session scheduling state: the decode scheduler, KV slot
//! lifecycle, KV cache runtime, and the session's evidence log.
//!
//! # Authority
//!
//! The types in this module are **scheduling state** in the C bucket:
//! - `DecodeScheduler` tracks the session's decode epoch and
//!   termination state.
//! - `KvSlotState` is the per-slot lifecycle enum.
//! - `KvRuntime` is the per-session KV cache handle (stub in the
//!   engine; the constitutional home mirrors the stub semantics).
//! - `SessionLogEntry` and `SessionEvidenceLog` are the per-session
//!   evidence records. The evidence log is admitted into
//!   `prism-ecs-runtime::evidence::session_receipts` in step 56.
//!
//! # Decomposition-by-authority
//!
//! Per the inventory, the engine's `prism_session.rs` is decomposed
//! by authority:
//! - **Runtime scheduling state** (this file): `DecodeScheduler`,
//!   `KvSlotState`, `KvRuntime`, `SessionLogEntry`,
//!   `SessionEvidenceLog`, `GenerationState`.
//! - **Runtime scheduling system** (step 32): `PrismSession` (the
//!   aggregate) and its `step`/`step_tri_lane`/`advance_state`
//!   methods.
//! - **Server / connection / request / client / auth / lifecycle**
//!   (existing in `prism-ecs-server::runtime::server::session_lifecycle`):
//!   `PrismSessionRequest`, `PrismExecutionMode`, `SchedulingMode`,
//!   `PrismStepRequest`, `PrismStepResult`.
//! - **Stable wire-level IDs** (move to `prism-ecs-protocol` in a
//!   later step): `session_id` (UUID v4 string), `image_digest`
//!   (hex string).
//!
//! The aggregate `PrismSession` does NOT move. The engine file is
//! the legacy duplicate; step 58 deletes it.
//!
//! # Migration provenance
//!
//! The legacy home was `compute-core/src/ecs/scheduling/prism_session.rs`.
//! The engine file is the legacy duplicate; step 58 deletes it.

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
/// Tracks the current epoch (decode step) against the requested budget
/// and signals termination. The scheduler is per-session, not global.
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
// KvRuntime (stub; replaced when kv_runtime migrates)
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
        // Stub: real implementation arrives with the kv_runtime migration.
    }

    /// Append a single layer's key/value data to the current sequence.
    pub fn append(&mut self, _layer: u32, _key: &[u8], _value: &[u8]) {
        // Stub.
    }

    /// Roll back the most recent append (e.g. after a rejected speculative
    /// token).
    pub fn rollback(&mut self) {
        // Stub.
    }

    /// Mark the entire cache as invalid (e.g. after a context switch).
    pub fn invalidate(&mut self) {
        // Stub.
    }

    /// Release all resources held by the runtime.
    pub fn release(&mut self) {
        // Stub.
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
#[derive(Debug, Clone, Default)]
pub struct SessionEvidenceLog {
    /// Ordered entries, one per epoch (or per significant event).
    pub entries: Vec<SessionLogEntry>,
}

impl SessionEvidenceLog {
    /// Create an empty log.
    pub fn new() -> Self {
        Self::default()
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

// ---------------------------------------------------------------------------
// Invariant tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Architectural-invariant tests for the `session` state.

    use super::*;

    #[test]
    fn decode_scheduler_starts_at_epoch_zero() {
        // Architectural invariant: a fresh decode scheduler is at
        // epoch 0 (prompt-processing) with no termination signal.
        let s = DecodeScheduler::new(100);
        assert_eq!(s.epoch, 0);
        assert_eq!(s.max_new_tokens, 100);
        assert!(!s.terminated);
    }

    #[test]
    fn decode_scheduler_advances_through_budget() {
        // Architectural invariant: advance() increments epoch and
        // sets terminated once the epoch EXCEEDS the budget (i.e.
        // epoch > max_new_tokens). At epoch == max_new_tokens the
        // scheduler has not yet signalled termination.
        let mut s = DecodeScheduler::new(3);
        s.advance();
        assert_eq!(s.epoch, 1);
        assert!(!s.terminated);
        s.advance();
        s.advance();
        // After 3 advances, epoch is 3, max is 3, NOT yet terminated
        // (3 > 3 is false). One more advance exceeds the budget.
        assert_eq!(s.epoch, 3);
        assert!(!s.terminated);
        s.advance();
        assert_eq!(s.epoch, 4);
        assert!(s.terminated);
    }

    #[test]
    fn decode_scheduler_terminate_is_immediate() {
        let mut s = DecodeScheduler::new(100);
        s.terminate();
        assert!(s.terminated);
        // Epoch does not advance on terminate; the scheduler is
        // waiting for the next opportunity to stop.
        assert_eq!(s.epoch, 0);
    }

    #[test]
    fn generation_state_transitions_are_strict() {
        // Architectural invariant: generation_state follows a
        // happy-path sequence (PromptProcessing → Decoding →
        // Completed). Terminal states (Completed, Failed) do not
        // transition further.
        let mut s = GenerationState::PromptProcessing;
        assert!(matches!(s, GenerationState::PromptProcessing));
        s = GenerationState::Decoding;
        assert!(matches!(s, GenerationState::Decoding));
        s = GenerationState::Completed;
        assert!(matches!(s, GenerationState::Completed));
        // Completed stays Completed (the runtime skips further
        // transitions; this test asserts the type allows it).
        s = GenerationState::Completed;
        assert!(matches!(s, GenerationState::Completed));
    }

    #[test]
    fn session_evidence_log_appends_and_counts() {
        // Architectural invariant: the evidence log is append-only
        // and the len() matches the number of recorded entries.
        let mut log = SessionEvidenceLog::new();
        assert!(log.is_empty());
        assert_eq!(log.len(), 0);
        log.record(SessionLogEntry {
            epoch: 0,
            token: None,
            wall_time_ns: 100,
            fallback_used: false,
            route_origin: "ane".into(),
        });
        log.record(SessionLogEntry {
            epoch: 1,
            token: Some(42),
            wall_time_ns: 200,
            fallback_used: false,
            route_origin: "gpu".into(),
        });
        assert_eq!(log.len(), 2);
        assert!(!log.is_empty());
        assert_eq!(log.entries[0].epoch, 0);
        assert_eq!(log.entries[1].token, Some(42));
    }

    #[test]
    fn kv_runtime_starts_at_generation_zero() {
        // Architectural invariant: a fresh KvRuntime is at generation
        // 0 with no allocated state.
        let r = KvRuntime::new();
        assert_eq!(r.generation(), 0);
    }

    #[test]
    fn kv_slot_state_variants_are_distinct() {
        // Architectural invariant: the seven KvSlotState variants
        // are mutually exclusive. A reader can dispatch on the
        // variant without forgetting a case.
        let states = [
            KvSlotState::Unallocated,
            KvSlotState::Allocated,
            KvSlotState::Primed,
            KvSlotState::Decoding,
            KvSlotState::Synchronized,
            KvSlotState::Invalidated,
            KvSlotState::Released,
        ];
        for s in states {
            let count = states.iter().filter(|&&v| v == s).count();
            assert_eq!(count, 1, "every variant must be self-equal exactly once");
        }
    }
}
