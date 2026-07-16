//! Phase receipts — evidence that a phase was executed.
//!
//! [`PhaseReceipt`] is produced by the phase engine after every
//! dispatched phase.  It records the phase ID, completion status,
//! duration, optional fused-kernel evidence, and compiler provenance
//! linking back to the compile session that produced the executable.

use crate::ecs::compute_image::fusion_receipts::FusedMetalExecutionEvidence;
use crate::ecs::compute_image::phase_dag::PhaseCompletionStatus;
use serde::{Deserialize, Serialize};

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
    /// Populated when a compile session created the phase graph.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compiler_session_id: Option<String>,
    /// Digest of the compiler event stream at the time this phase
    /// was generated.  Enables cross-referencing between runtime
    /// receipts and compile-time evidence.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub compiler_event_digest: Option<String>,
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
}

impl PhaseReceipt {
    /// Attach compiler provenance linking this phase back to a compile
    /// session and its event stream digest.
    pub fn with_compiler_provenance(mut self, session_id: &str, event_digest: &str) -> Self {
        self.compiler_session_id = Some(session_id.into());
        self.compiler_event_digest = Some(event_digest.into());
        self
    }
}
