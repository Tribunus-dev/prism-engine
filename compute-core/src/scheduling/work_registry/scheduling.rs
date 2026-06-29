//! Work scheduling identity — [`WorkKey`] for session/phase identity across
//! retries and fallback attempts.

use serde::{Deserialize, Serialize};

use crate::compilation::phase_ir::PhaseId;

use super::*;

// ── Work key ────────────────────────────────────────────────────────────────

/// Work identity across retries and fallback attempts.
///
/// Uniquely identifies a logical unit of work across multiple physical
/// submissions — when a fallback retry creates a new [`WorkId`], the
/// `WorkKey` remains the same.
///
/// The [`WorkKey`] is used for scheduling decisions such as deduplication,
/// retry tracking, and associating completion events with the original request.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WorkKey {
    /// Logical session identifier.
    pub session_id: String,
    /// Request identifier within the session.
    pub request_id: String,
    /// Sequence number within the request.
    pub sequence_id: u64,
    /// Epoch identifier (for epoch-based scheduling).
    pub epoch_id: u64,
    /// Compilation phase.
    pub phase_id: PhaseId,
    /// Attempt number (0 = original, 1+ = fallback retries).
    pub attempt: u32,
}
