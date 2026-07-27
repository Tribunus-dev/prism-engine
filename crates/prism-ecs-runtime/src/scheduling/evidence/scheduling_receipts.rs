//! Scheduling receipts (constitutional home).
//!
//! Per the inventory v2.1 step 51-52, this is the canonical home for
//! `PhaseReceipt` and the related scheduling evidence types. Per
//! the inventory, the engine's `receipts.rs` (64 LOC) and `receipt.rs`
//! (618 LOC) merge here as a scheduling specialization of
//! `engine_receipts` (the canonical receipt shape, in
//! `prism_ecs_runtime::engine_receipts`).
//!
//! # Authority
//!
//! These receipts are admitted evidence (E bucket). A receipt is
//! emitted only for a committed transaction. The receipt is a
//! function of the committed state, not the in-flight one.
//!
//! # Placeholder engine types
//!
//! `PhaseCompletionStatus` and `FusedMetalExecutionEvidence` are
//! engine types that move with the phase_dag / fusion_receipts
//! migrations. The constitutional home has minimal placeholders.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Placeholder engine types
// ---------------------------------------------------------------------------

/// Placeholder for `compute-core::ecs::compute_image::phase_dag::PhaseCompletionStatus`.
/// Replaced when phase_dag migrates.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhaseCompletionStatus {
    Complete,
    Failed(String),
    Skipped,
}

// ---------------------------------------------------------------------------
// PhaseReceipt
// ---------------------------------------------------------------------------

/// Receipt for an executed phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseReceipt {
    pub phase_id: String,
    pub status: PhaseCompletionStatus,
    pub duration_us: u64,
    /// Fused-kernel execution evidence (present when a fused kernel ran).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub fused_evidence: Option<FusedMetalExecutionEvidence>,
    /// Compiler session that produced the artifact for this phase.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compiler_session_id: Option<String>,
    /// Digest of the compiler event stream at the time this phase
    /// was generated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compiler_event_digest: Option<String>,
}

/// Placeholder for the fused-kernel execution evidence type.
/// Replaced when fusion_receipts migrates.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FusedMetalExecutionEvidence {
    pub fused_kernel_digest: String,
    pub total_dispatch_ns: u64,
}

impl PhaseReceipt {
    pub fn completed(phase_id: &str, duration_us: u64) -> Self {
        Self {
            phase_id: phase_id.into(),
            status: PhaseCompletionStatus::Complete,
            duration_us,
            fused_evidence: None,
            compiler_session_id: None,
            compiler_event_digest: None,
        }
    }

    pub fn failed(phase_id: &str, reason: &str) -> Self {
        Self {
            phase_id: phase_id.into(),
            status: PhaseCompletionStatus::Failed(reason.into()),
            duration_us: 0,
            fused_evidence: None,
            compiler_session_id: None,
            compiler_event_digest: None,
        }
    }

    /// Attach compiler provenance linking this phase back to a compile
    /// session and its event stream digest.
    pub fn with_compiler_provenance(mut self, session_id: &str, event_digest: &str) -> Self {
        self.compiler_session_id = Some(session_id.into());
        self.compiler_event_digest = Some(event_digest.into());
        self
    }
}

// ---------------------------------------------------------------------------
// Invariant tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! Architectural-invariant tests for the `scheduling_receipts` evidence.

    use super::*;

    #[test]
    fn completed_receipt_carries_phase_id_and_duration() {
        // Architectural invariant: a completed receipt records the
        // phase id, the Complete status, and the duration. Receipts
        // are admitted evidence — they correspond to committed
        // state, not in-flight work.
        let r = PhaseReceipt::completed("p1", 1000);
        assert_eq!(r.phase_id, "p1");
        assert_eq!(r.duration_us, 1000);
        assert!(matches!(r.status, PhaseCompletionStatus::Complete));
    }

    #[test]
    fn failed_receipt_carries_reason() {
        // Architectural invariant: a failed receipt records the
        // failure reason in the status. The receipt's duration
        // is zero (the phase did not complete).
        let r = PhaseReceipt::failed("p2", "out of memory");
        assert!(matches!(r.status, PhaseCompletionStatus::Failed(_)));
        if let PhaseCompletionStatus::Failed(reason) = &r.status {
            assert_eq!(reason, "out of memory");
        }
        assert_eq!(r.duration_us, 0);
    }

    #[test]
    fn compiler_provenance_attaches_to_receipt() {
        // Architectural invariant: with_compiler_provenance
        // attaches both the session id and the event digest.
        // A reader can cross-reference a runtime receipt with
        // its compile-time evidence.
        let r = PhaseReceipt::completed("p1", 500)
            .with_compiler_provenance("session-1", "digest-1");
        assert_eq!(r.compiler_session_id.as_deref(), Some("session-1"));
        assert_eq!(r.compiler_event_digest.as_deref(), Some("digest-1"));
    }
}
