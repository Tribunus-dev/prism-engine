//! Phase dependency and completion contracts.

use serde::{Deserialize, Serialize};

/// Dependency contract between two phases.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseDependencyContract {
    /// Source phase identifier.
    pub from_phase: String,
    /// Target phase identifier.
    pub to_phase: String,
    /// Tensor identifier that flows between phases.
    pub tensor_id: String,
}

/// Completion contract for a phase.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseCompletionContract {
    /// Phase identifier.
    pub phase_id: String,
    /// Whether the phase is required for correctness.
    pub required: bool,
    /// Optional fallback phase if this one cannot run.
    pub fallback: Option<String>,
}
